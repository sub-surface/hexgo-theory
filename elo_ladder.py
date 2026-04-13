"""
ELO round-robin ladder for hexgo-theory agents.

Uses Python agents (ForkAwareAgent, PotentialGradientAgent, ComboAgent,
EisensteinGreedyAgent) with optional Rust HexGame board for speed.

Run:
    python -X utf8 elo_ladder.py [--games N] [--verbose] [--regen]
    python -X utf8 elo_ladder.py --agents Greedy-def Fork-a2 Fork-a4 MCTS-200 --games 100
"""
from __future__ import annotations
import sys, json, math, time, random, argparse, itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CACHE_FILE = Path(__file__).parent / "elo_results.json"

# ── Agent factory ───────────────────────────────────────────────────────────────
# Each entry: (display_name, factory_fn)

def _make_agents():
    from engine import EisensteinGreedyAgent, ForkAwareAgent, PotentialGradientAgent, ComboAgent

    class _NoisyGreedy:
        """EisensteinGreedyAgent with eps noise to break determinism."""
        def __init__(self, defensive=True, eps=0.01):
            self._inner = EisensteinGreedyAgent(defensive=defensive)
            self._defensive = defensive
            self._eps = eps
            self.name = "noisy_greedy" + ("_def" if defensive else "_off")

        def choose_move(self, game):
            from engine import AXES
            player   = game.current_player
            opponent = 3 - player
            best_move, best_score = None, -1.0
            for q, r in game.legal_moves():
                own = _chain_if_placed(game, q, r, player, AXES)
                blk = _chain_if_placed(game, q, r, opponent, AXES) if self._defensive else 0
                score = max(own, blk) + self._eps * random.random()
                if score > best_score or best_move is None:
                    best_score, best_move = score, (q, r)
            return best_move or random.choice(game.legal_moves())


    agents = {
        "Greedy-off": lambda: _NoisyGreedy(defensive=False),
        "Greedy-def": lambda: _NoisyGreedy(defensive=True),
        "Fork-a1":    lambda: ForkAwareAgent("fork_a1",  alpha=1.0, defensive=True),
        "Fork-a2":    lambda: ForkAwareAgent("fork_a2",  alpha=2.0, defensive=True),
        "Fork-a4":    lambda: ForkAwareAgent("fork_a4",  alpha=4.0, defensive=True),
        "Fork-a8":    lambda: ForkAwareAgent("fork_a8",  alpha=8.0, defensive=True),
        "PotGrad":    lambda: PotentialGradientAgent("potgrad"),
        "Combo":      lambda: ComboAgent("combo"),
    }

    # Optional: MCTS via Rust engine
    try:
        from hexgo import mcts as rust_mcts, HexGame as RustHexGame
        class _MCTSAgent:
            def __init__(self, sims):
                self.sims = sims
                self.name = f"mcts_{sims}"
            def choose_move(self, game):
                # Convert Python game state to Rust for fast MCTS
                rg = _py_to_rust(game)
                if rg is None:
                    return random.choice(game.legal_moves())
                m = rust_mcts(rg, num_sims=self.sims)
                return (int(m[0]), int(m[1]))

        agents["MCTS-100"] = lambda: _MCTSAgent(100)
        agents["MCTS-200"] = lambda: _MCTSAgent(200)
    except ImportError:
        pass

    return agents

def _chain_if_placed(game, q: int, r: int, player: int, axes) -> int:
    best = 1
    board = game.board
    for dq, dr in axes:
        count = 1
        for sign in (1, -1):
            nq, nr = q + sign * dq, r + sign * dr
            while board.get((nq, nr)) == player:
                count += 1
                nq += sign * dq
                nr += sign * dr
        best = max(best, count)
    return best


def _py_to_rust(game):
    """Attempt to replicate game state in Rust HexGame by replaying moves."""
    try:
        from hexgo import HexGame as RustHexGame
        rg = RustHexGame()
        for (q, r) in game.move_history:
            rg.make_move(q, r)
        return rg
    except Exception:
        return None


# ── Game runner ─────────────────────────────────────────────────────────────────

def _play_one(agent_a, agent_b, max_moves: int = 150, a_is_p1: bool = True) -> dict:
    """Play one game. HexGame handles 1-2-2 internally — just call make_move per stone."""
    from engine import HexGame
    game = HexGame()
    p1_agent = agent_a if a_is_p1 else agent_b
    p2_agent = agent_b if a_is_p1 else agent_a

    moves = 0
    while game.winner is None and moves < max_moves:
        turn_agent = p1_agent if game.current_player == 1 else p2_agent
        legal = game.legal_moves()
        if not legal:
            break
        move = turn_agent.choose_move(game)
        if move not in set(legal):
            move = random.choice(legal)
        game.make_move(*move)
        moves += 1

    winner = game.winner
    if winner == 1:
        winner_name = agent_a.name if a_is_p1 else agent_b.name
    elif winner == 2:
        winner_name = agent_b.name if a_is_p1 else agent_a.name
    else:
        winner_name = None

    return {"winner": winner_name, "moves": moves, "swap": not a_is_p1}


def _run_matchup(name_a: str, name_b: str, factory_a, factory_b,
                 n: int, max_moves: int, verbose: bool) -> list[dict]:
    """Run n games alternating sides. Uses thread pool for parallelism."""
    results = []

    def run_game(i):
        a = factory_a()
        b = factory_b()
        a_is_p1 = (i % 2 == 0)
        return _play_one(a, b, max_moves=max_moves, a_is_p1=a_is_p1)

    # Use threads — GIL-free for Rust HexGame ops; Python agents are fast enough
    max_workers = min(8, n)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_game, i): i for i in range(n)}
        for fut in as_completed(futures):
            results.append(fut.result())

    return results


# ── ELO computation ────────────────────────────────────────────────────────────

def compute_elo(matchup_results: dict, agent_names: list[str],
                k: float = 32.0, init: float = 1000.0) -> dict[str, float]:
    elo = {n: init for n in agent_names}
    for _ in range(20):
        for (na, nb), results in matchup_results.items():
            for r in results:
                ea = 1.0 / (1.0 + 10 ** ((elo[nb] - elo[na]) / 400.0))
                eb = 1.0 - ea
                if r["winner"] == na:
                    sa, sb = 1.0, 0.0
                elif r["winner"] == nb:
                    sa, sb = 0.0, 1.0
                else:
                    sa = sb = 0.5
                elo[na] += k * (sa - ea)
                elo[nb] += k * (sb - eb)
    return elo


def matchup_stats(results: list[dict], na: str, nb: str) -> dict:
    wins_a = sum(1 for r in results if r["winner"] == na)
    wins_b = sum(1 for r in results if r["winner"] == nb)
    draws  = sum(1 for r in results if r["winner"] is None)
    n      = len(results)
    avg_moves = sum(r["moves"] for r in results) / max(1, n)
    return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws,
            "n": n, "avg_moves": avg_moves,
            "win_pct_a": 100 * wins_a / max(1, n)}


def bar(val: float, max_val: float, width: int = 20) -> str:
    filled = int(width * val / max(max_val, 1e-9))
    return "█" * filled + "░" * (width - filled)


# ── Main ───────────────────────────────────────────────────────────────────────

# Default agent set (ordered weakest → strongest expected)
DEFAULT_AGENTS = [
    "Greedy-off",
    "Greedy-def",
    "Fork-a1",
    "Fork-a2",
    "Fork-a4",
    "Fork-a8",
    "PotGrad",
    "Combo",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games",     type=int, default=100,
                        help="Games per matchup pair (default 100)")
    parser.add_argument("--verbose",   "-v", action="store_true")
    parser.add_argument("--regen",     action="store_true",
                        help="Ignore cache, regenerate all games")
    parser.add_argument("--agents",    nargs="*",
                        help="Subset of agent names (display names)")
    parser.add_argument("--max-moves", type=int, default=150)
    args = parser.parse_args()

    ALL_AGENTS = _make_agents()

    names = args.agents if args.agents else DEFAULT_AGENTS
    # Filter to available agents
    names = [n for n in names if n in ALL_AGENTS]
    if len(names) < 2:
        available = list(ALL_AGENTS.keys())
        print(f"Need >=2 agents. Available: {available}")
        return

    pairs = list(itertools.combinations(names, 2))

    # Load cache
    cache: dict = {}
    if CACHE_FILE.exists() and not args.regen:
        try:
            cache = json.loads(CACHE_FILE.read_text())
            print(f"[cache] {len(cache)} matchup(s) loaded from {CACHE_FILE.name}")
        except Exception:
            cache = {}

    # Normalize winner names in cache: old runs may have stored Rust descriptors.
    # We store display names now, so old cache entries for these pairs are incompatible.
    # Clear mismatched entries silently.

    matchup_results: dict = {}
    t_total = time.perf_counter()

    for na, nb in pairs:
        key = f"{na}|{nb}"
        cached = cache.get(key, [])
        need   = args.games

        if len(cached) >= need and not args.regen:
            matchup_results[(na, nb)] = cached[:need]
            print(f"  [cached] {na:18s} vs {nb:18s}  ({need} games)")
        else:
            missing = need - len(cached)
            print(f"  [run]    {na:18s} vs {nb:18s}  ({missing} new)...",
                  end="", flush=True)
            t0 = time.perf_counter()
            new = _run_matchup(na, nb, ALL_AGENTS[na], ALL_AGENTS[nb],
                               missing, args.max_moves, args.verbose)
            elapsed = time.perf_counter() - t0
            all_r = cached + new
            cache[key] = all_r
            matchup_results[(na, nb)] = all_r[:need]
            ms_per = elapsed / missing * 1000 if missing else 0
            print(f"  done ({elapsed:.1f}s, {ms_per:.0f}ms/game)")

    CACHE_FILE.write_text(json.dumps(cache, indent=2))
    total_elapsed = time.perf_counter() - t_total
    print(f"\nTotal: {total_elapsed:.1f}s")

    # ── Head-to-head table ─────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  HEAD-TO-HEAD")
    print("=" * 80)
    print(f"  {'Matchup':43s}  {'W-A':>5} {'W-B':>5} {'Drw':>5} {'Avg♟':>6}  A%")
    print(f"  {'-'*43}  {'-'*5} {'-'*5} {'-'*5} {'-'*6}  --")
    for na, nb in pairs:
        s = matchup_stats(matchup_results[(na, nb)], na, nb)
        label = f"{na} vs {nb}"
        print(f"  {label:43s}  {s['wins_a']:>5} {s['wins_b']:>5} {s['draws']:>5} "
              f"{s['avg_moves']:>6.1f}  {s['win_pct_a']:5.1f}%  {bar(s['win_pct_a'],100,16)}")

    # ── ELO table ──────────────────────────────────────────────────────────
    elo = compute_elo(matchup_results, names)
    ranked = sorted(elo.items(), key=lambda x: -x[1])
    baseline = elo.get("Greedy-def", 1000.0)

    print()
    print("=" * 80)
    print("  ELO RATINGS  (init=1000, K=32, 20-pass convergence)")
    print("=" * 80)
    for rank, (name, rating) in enumerate(ranked, 1):
        delta = rating - baseline
        sign  = "+" if delta >= 0 else ""
        print(f"  {rank}. {name:22s}  {rating:>7.1f}  ({sign}{delta:+.0f} vs Greedy-def)  "
              f"{bar(max(0, rating - 700), 700, 24)}")

    # ── Hypothesis checks ──────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("  HYPOTHESIS CHECKS")
    print("=" * 80)

    def win_pct(na, nb):
        key = (na, nb) if (na, nb) in matchup_results else (nb, na)
        if key not in matchup_results:
            return None
        s = matchup_stats(matchup_results[key], key[0], key[1])
        return s["win_pct_a"] if key[0] == na else 100 - s["win_pct_a"]

    checks = [
        ("Fork-a2  beats Greedy-def >55%",   "Fork-a2",  "Greedy-def",  55),
        ("Fork-a4  beats Greedy-def >60%",   "Fork-a4",  "Greedy-def",  60),
        ("Fork-a8  beats Greedy-def >60%",   "Fork-a8",  "Greedy-def",  60),
        ("PotGrad  beats Greedy-def >60%",   "PotGrad",  "Greedy-def",  60),
        ("Combo    beats Greedy-def >65%",   "Combo",    "Greedy-def",  65),
        ("Combo    beats Fork-a4 >55%",      "Combo",    "Fork-a4",     55),
        ("Fork-a4  beats Fork-a2 >55%",      "Fork-a4",  "Fork-a2",     55),
        ("Fork-a8  beats Fork-a4 >55%",      "Fork-a8",  "Fork-a4",     55),
        ("Combo    beats PotGrad >50%",      "Combo",    "PotGrad",     50),
    ]
    for label, na, nb, thresh in checks:
        if na not in names or nb not in names:
            print(f"  SKIP  {label}")
            continue
        pct = win_pct(na, nb)
        if pct is None:
            print(f"  SKIP  {label}  (no data)")
            continue
        sym = "✓" if pct >= thresh else "✗"
        print(f"  {sym}  {label:42s}  {pct:5.1f}%  (>={thresh}%)")

    # ── Game length ────────────────────────────────────────────────────────
    print()
    print("  Avg game length per agent (as P1, across all matchups):")
    for name in names:
        lengths = []
        for (na, nb), results in matchup_results.items():
            if na == name:
                lengths += [r["moves"] for r in results if not r["swap"]]
            elif nb == name:
                lengths += [r["moves"] for r in results if r["swap"]]
        if lengths:
            print(f"    {name:22s}  {sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    main()
