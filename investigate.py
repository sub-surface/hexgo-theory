"""
Headless investigation script — runs experiments and prints structured results.
No GUI required.

Usage:
    python investigate.py [--games N] [--question Q]

Questions:
    all          Run all investigations (default)
    forks        Fork frequency census + fork-win correlation
    correlation  Pair correlation g(r) across corpus
    patterns     Pattern type census
    spectrum     2D Fourier / diffraction analysis of move positions
"""
from __future__ import annotations
import sys, os, argparse, math, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

# Force UTF-8 on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from engine import HexGame, EisensteinGreedyAgent, RandomAgent
from engine.analysis import (
    fork_cells, threat_cells, potential_map, live_lines,
    pair_correlation, pattern_fingerprint, live_ap_count, axis_chain_lengths,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def play_game(agent_a, agent_b, swap: bool = False, max_moves: int = 300):
    game = HexGame()
    agents = {1: agent_b, 2: agent_a} if swap else {1: agent_a, 2: agent_b}
    move_players = []
    fork_counts_per_move: list[tuple[int, int]] = []   # (f1, f2) after each move
    threat_counts_per_move: list[tuple[int, int]] = []

    while game.winner is None and len(game.move_history) < max_moves:
        player = game.current_player
        move = agents[player].choose_move(game)
        game.make(*move)
        move_players.append(player)

        f1 = len(fork_cells(game, 1))
        f2 = len(fork_cells(game, 2))
        t1 = len(threat_cells(game, 1))
        t2 = len(threat_cells(game, 2))
        fork_counts_per_move.append((f1, f2))
        threat_counts_per_move.append((t1, t2))

    return game, move_players, fork_counts_per_move, threat_counts_per_move


def bar(val: float, max_val: float, width: int = 30) -> str:
    filled = int(width * val / max(max_val, 1e-9))
    return "█" * filled + "░" * (width - filled)


def hdr(title: str):
    print()
    print("═" * 70)
    print(f"  {title}")
    print("═" * 70)


def section(title: str):
    print()
    print(f"── {title} " + "─" * (66 - len(title)))


# ─── Q1: Fork census + fork-win correlation ───────────────────────────────────

def q_forks(n_games: int):
    hdr("QUESTION 1 — FORK GEOMETRY")
    print(f"  Running {n_games} Eisenstein-vs-Eisenstein games (alternating sides).")

    agent_a = EisensteinGreedyAgent("Eis-A", defensive=False)
    agent_b = EisensteinGreedyAgent("Eis-B", defensive=True)

    wins = {1: 0, 2: 0, 0: 0}
    fork_at_move: dict[int, list[int]] = defaultdict(list)   # move# → total fork count
    fork_totals_winner: list[int] = []    # total forks created by winner per game
    fork_totals_loser:  list[int] = []
    first_fork_moves: list[int] = []      # move# when first fork appears
    triple_forks_seen = 0
    games_with_greedy_miss = 0
    move_counts = []
    winner_had_more_forks = 0
    loser_had_more_forks  = 0
    equal_forks = 0

    t0 = time.perf_counter()
    for g in range(n_games):
        swap = (g % 2 == 1)
        game, players, fork_hist, threat_hist = play_game(agent_a, agent_b, swap)

        winner = game.winner or 0
        move_counts.append(len(game.move_history))
        wins[winner] = wins.get(winner, 0) + 1

        # Fork counts per move
        first_fork = None
        for i, (f1, f2) in enumerate(fork_hist):
            total = f1 + f2
            fork_at_move[i + 1].append(total)
            if total > 0 and first_fork is None:
                first_fork = i + 1
        if first_fork:
            first_fork_moves.append(first_fork)

        # Triple forks (cells on all 3 axes simultaneously)
        # fork_cells returns {cell: axis_count} where axis_count >= 2 means fork
        # axis_count == 3 means the cell extends chains on all three axes
        for player in (1, 2):
            final_forks = fork_cells(game, player, min_chain=1)
            for cell, axis_count in final_forks.items():
                if axis_count >= 3:
                    triple_forks_seen += 1

        # Fork-win correlation: did the winner accumulate more total fork cells?
        total_forks = fork_hist[-1] if fork_hist else (0, 0)
        f_p1, f_p2 = total_forks
        if winner == 1:
            if f_p1 > f_p2: winner_had_more_forks += 1
            elif f_p1 < f_p2: loser_had_more_forks += 1
            else: equal_forks += 1
        elif winner == 2:
            if f_p2 > f_p1: winner_had_more_forks += 1
            elif f_p2 < f_p1: loser_had_more_forks += 1
            else: equal_forks += 1

        # Greedy fork blindness: was there a move where a fork cell existed
        # but the greedy agent did NOT play it (it played somewhere else)?
        for i, (move, player) in enumerate(zip(game.move_history, players)):
            # Reconstruct board state before this move
            pre = HexGame()
            for m in game.move_history[:i]:
                pre.make(*m)
            fk = fork_cells(pre, player)
            if fk and move not in fk:
                games_with_greedy_miss += 1
                break

        if (g + 1) % max(1, n_games // 10) == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {g+1:>5}/{n_games}  ({elapsed:.1f}s)", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  {n_games}/{n_games}  done in {elapsed:.1f}s              ")

    # ── Results ──
    section("Win rates")
    total = sum(wins.values())
    for p, label in [(1, "P1 (Eis-A)"), (2, "P2 (Eis-B)"), (0, "timeout")]:
        pct = 100 * wins[p] / max(1, total)
        print(f"  {label:20s}  {wins[p]:>5}  {pct:5.1f}%  {bar(pct, 100, 25)}")
    avg_moves = sum(move_counts) / len(move_counts)
    print(f"  avg moves/game: {avg_moves:.1f}")

    section("First fork appearance")
    if first_fork_moves:
        avg_first = sum(first_fork_moves) / len(first_fork_moves)
        min_first  = min(first_fork_moves)
        print(f"  mean move# of first fork:  {avg_first:.1f}")
        print(f"  earliest fork ever:        move {min_first}")
        pct_w_fork = 100 * len(first_fork_moves) / n_games
        print(f"  games that had any fork:   {pct_w_fork:.1f}%")
    else:
        print("  No forks observed in any game.")

    section("Fork count profile (avg total forks at each move#)")
    sample_moves = [5, 10, 15, 20, 30, 40, 50, 60, 80]
    for m in sample_moves:
        vals = fork_at_move.get(m, [])
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  move {m:>3}:  avg forks = {avg:5.2f}  {bar(avg, 20)}")

    section("Fork-win correlation")
    total_decided = winner_had_more_forks + loser_had_more_forks + equal_forks
    if total_decided:
        pct_winner = 100 * winner_had_more_forks / total_decided
        pct_loser  = 100 * loser_had_more_forks  / total_decided
        pct_equal  = 100 * equal_forks           / total_decided
        print(f"  Winner had more forks at game end:   {winner_had_more_forks:>4}  ({pct_winner:.1f}%)")
        print(f"  Loser  had more forks at game end:   {loser_had_more_forks:>4}  ({pct_loser:.1f}%)")
        print(f"  Equal forks:                         {equal_forks:>4}  ({pct_equal:.1f}%)")
        if pct_winner > 55:
            print("  ✓ HYPOTHESIS SUPPORTED: forks are a leading indicator of win.")
        elif pct_winner < 45:
            print("  ✗ HYPOTHESIS REJECTED: forks do not predict win.")
        else:
            print("  ~ AMBIGUOUS: fork-win correlation is weak.")

    section("Triple forks (cell on all 3 axes)")
    print(f"  Triple fork cells seen across all games: {triple_forks_seen}")
    if triple_forks_seen > 0:
        print("  ✓ Triple forks ARE geometrically achievable in Z[ω].")
    else:
        print("  (No triple forks seen — may require larger N or specific positions.)")

    section("Greedy fork blindness")
    pct_miss = 100 * games_with_greedy_miss / n_games
    print(f"  Games where greedy agent ignored an available fork: {games_with_greedy_miss}/{n_games} ({pct_miss:.1f}%)")
    if pct_miss > 10:
        print("  ✓ CONFIRMED: EisensteinGreedy frequently ignores fork opportunities.")
    else:
        print("  Fork blindness is rare — forks may seldom score better than chain moves.")


# ─── Q2: Pair correlation g(r) ────────────────────────────────────────────────

def q_correlation(n_games: int):
    hdr("QUESTION 2 — PAIR CORRELATION g(r)")
    print(f"  Running {n_games} Eis-vs-Eis games and pooling all move positions.")

    agent_a = EisensteinGreedyAgent("Eis-A", defensive=False)
    agent_b = EisensteinGreedyAgent("Eis-B", defensive=True)

    all_moves: list[tuple[int, int]] = []
    all_corrs: dict[int, list[float]] = defaultdict(list)

    t0 = time.perf_counter()
    for g in range(n_games):
        game, players, _, _ = play_game(agent_a, agent_b, swap=(g % 2 == 1))
        all_moves.extend(game.move_history)
        corr = pair_correlation(game.move_history, max_r=20)
        for r, val in corr.items():
            all_corrs[r].append(val)
        if (g + 1) % max(1, n_games // 5) == 0:
            print(f"  {g+1}/{n_games}", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s              ")

    section("Average g(r) across corpus")
    print(f"  {'r':>3}   {'g(r)':>7}   chart (baseline=1.0)")
    print(f"  {'─'*3}   {'─'*7}   {'─'*40}")
    peaks = []
    troughs = []
    prev_v = None
    for r in range(1, 21):
        vals = all_corrs.get(r, [])
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        deviation = avg - 1.0
        bar_str = bar(max(0, avg), 3.0, 30)
        marker = ""
        if prev_v is not None:
            if avg > prev_v and avg > 1.05:
                peaks.append((r, avg))
                marker = " ▲ peak"
            elif avg < prev_v and avg < 0.95:
                troughs.append((r, avg))
                marker = " ▼ trough"
        print(f"  {r:>3}   {avg:>7.3f}   {bar_str}{marker}")
        prev_v = avg

    section("Interpretation")
    total_pts = len(all_moves)
    print(f"  Total move positions pooled: {total_pts}")
    if peaks:
        print(f"  g(r) peaks at r = {[p[0] for p in peaks]}")
        print(f"    Peaks suggest preferred spacing between stones.")
    if troughs:
        print(f"  g(r) troughs at r = {[t[0] for t in troughs]}")
        print(f"    Troughs suggest avoided spacing (exclusion zones).")

    # Check for quasi-periodic signature: peaks at irrational ratios
    if len(peaks) >= 2:
        ratios = []
        for i in range(len(peaks) - 1):
            ratios.append(peaks[i+1][0] / peaks[i][0])
        print(f"  Peak spacing ratios: {[f'{r:.3f}' for r in ratios]}")
        golden = 1.6180339887
        tribonacci = 1.3247179572
        for ratio in ratios:
            if abs(ratio - golden) < 0.05:
                print(f"  ✓ Ratio ≈ golden ratio (φ ≈ 1.618) — quasi-crystal signature!")
            elif abs(ratio - tribonacci) < 0.05:
                print(f"  ✓ Ratio ≈ tribonacci constant (≈ 1.325) — Pisot signature!")
            else:
                print(f"  Ratio {ratio:.3f} does not match known Pisot constants.")


# ─── Q3: Pattern type census ──────────────────────────────────────────────────

def q_patterns(n_games: int):
    hdr("QUESTION 3 — PATTERN TYPE CENSUS")
    print(f"  Running {n_games} games and collecting radius-2 local pattern fingerprints.")

    agent_a = EisensteinGreedyAgent("Eis-A", defensive=False)
    agent_b = EisensteinGreedyAgent("Eis-B", defensive=True)

    pattern_freq: dict[str, int] = defaultdict(int)
    total_patterns = 0

    t0 = time.perf_counter()
    for g in range(n_games):
        game, _, _, _ = play_game(agent_a, agent_b, swap=(g % 2 == 1))
        pats = pattern_fingerprint(game, radius=2)
        for fp in pats.values():
            pattern_freq[fp] += 1
            total_patterns += 1
        if (g + 1) % max(1, n_games // 5) == 0:
            print(f"  {g+1}/{n_games}", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s              ")

    section("Pattern frequency distribution")
    sorted_pats = sorted(pattern_freq.items(), key=lambda x: -x[1])
    unique = len(sorted_pats)
    print(f"  Unique pattern types: {unique}")
    print(f"  Total patterns seen:  {total_patterns}")
    print(f"  Coverage ratio:       {unique/max(1,total_patterns):.4f}")
    print()
    print(f"  Top 20 most frequent patterns:")
    for fp, cnt in sorted_pats[:20]:
        pct = 100 * cnt / total_patterns
        print(f"    {fp[:32]:32s}  {cnt:>6}  ({pct:4.1f}%)  {bar(pct, 10, 20)}")

    section("Power-law test (Zipf)")
    # If freq ∝ rank^(-α) then log(freq) vs log(rank) should be linear
    # Compute slope via least-squares on top 50
    log_ranks = []
    log_freqs = []
    for i, (_, cnt) in enumerate(sorted_pats[:50], 1):
        if cnt > 0:
            log_ranks.append(math.log(i))
            log_freqs.append(math.log(cnt))

    if len(log_ranks) >= 5:
        n = len(log_ranks)
        sx  = sum(log_ranks)
        sy  = sum(log_freqs)
        sxx = sum(x*x for x in log_ranks)
        sxy = sum(x*y for x, y in zip(log_ranks, log_freqs))
        slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        print(f"  Zipf exponent (slope of log-log rank-frequency): α = {-slope:.3f}")
        if 0.8 < -slope < 1.5:
            print("  ✓ Near-Zipf distribution — consistent with substitution tiling prediction.")
        else:
            print(f"  Exponent {-slope:.3f} deviates from Zipf (expected ~1.0 for substitution).")


# ─── Q4: Diffraction / Fourier spectrum ───────────────────────────────────────

def q_spectrum(n_games: int):
    hdr("QUESTION 4 — DIFFRACTION SPECTRUM")
    print(f"  Running {n_games} games and computing 2D Fourier transform of move positions.")

    agent_a = EisensteinGreedyAgent("Eis-A", defensive=False)
    agent_b = EisensteinGreedyAgent("Eis-B", defensive=True)

    all_positions: list[tuple[int, int]] = []

    t0 = time.perf_counter()
    for g in range(n_games):
        game, _, _, _ = play_game(agent_a, agent_b, swap=(g % 2 == 1))
        all_positions.extend(game.move_history)
        if (g + 1) % max(1, n_games // 5) == 0:
            print(f"  {g+1}/{n_games}", end="\r", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s              ")

    # Convert axial (q,r) → Cartesian for the flat-top hex grid
    sqrt3 = math.sqrt(3)
    positions_xy = []
    for q, r in all_positions:
        x = 1.5 * q
        y = sqrt3/2 * q + sqrt3 * r
        positions_xy.append((x, y))

    section("Point set statistics")
    n_pts = len(positions_xy)
    print(f"  Total points: {n_pts}")
    xs = [p[0] for p in positions_xy]
    ys = [p[1] for p in positions_xy]
    print(f"  x range: [{min(xs):.1f}, {max(xs):.1f}]")
    print(f"  y range: [{min(ys):.1f}, {max(ys):.1f}]")

    # Discrete Fourier amplitude |F(k)|² at selected wave-vectors
    # Sample k-vectors at angles 0°, 30°, 60° (D6 symmetry directions)
    # and several radii
    section("Fourier amplitudes at D6-symmetric k-vectors")
    print("  If the pattern has D6 symmetry, |F(k)| should be equal at")
    print("  k-vectors related by 60° rotation.")
    print()

    def fourier_amp(kx: float, ky: float) -> float:
        re = sum(math.cos(2 * math.pi * (kx * x + ky * y)) for x, y in positions_xy)
        im = sum(math.sin(2 * math.pi * (kx * x + ky * y)) for x, y in positions_xy)
        return math.sqrt(re*re + im*im) / n_pts

    k_radii = [0.1, 0.2, 0.5, 1.0]
    for kr in k_radii:
        amps = []
        for angle_deg in range(0, 360, 60):
            angle = math.radians(angle_deg)
            kx = kr * math.cos(angle)
            ky = kr * math.sin(angle)
            amps.append(fourier_amp(kx, ky))
        avg_amp  = sum(amps) / len(amps)
        max_amp  = max(amps)
        min_amp  = min(amps)
        symmetry = 1.0 - (max_amp - min_amp) / (avg_amp + 1e-9)
        print(f"  |k|={kr:.2f}  avg|F|={avg_amp:.4f}  "
              f"range=[{min_amp:.4f},{max_amp:.4f}]  D6-sym={symmetry:.3f}")

    section("Radial power spectrum P(|k|)")
    print("  Sampling |F(k)|² averaged over all angles at each |k|.")
    k_vals = [i * 0.05 for i in range(1, 25)]
    power_vals = []
    for kr in k_vals:
        n_angles = 12
        amps = []
        for j in range(n_angles):
            angle = 2 * math.pi * j / n_angles
            kx = kr * math.cos(angle)
            ky = kr * math.sin(angle)
            amps.append(fourier_amp(kx, ky) ** 2)
        power_vals.append(sum(amps) / n_angles)

    max_p = max(power_vals) or 1.0
    peaks_k = []
    for i in range(1, len(power_vals) - 1):
        if power_vals[i] > power_vals[i-1] and power_vals[i] > power_vals[i+1]:
            peaks_k.append(k_vals[i])
    print(f"  {'|k|':>5}   {'P(k)':>8}   chart")
    for k, p in zip(k_vals, power_vals):
        print(f"  {k:>5.2f}   {p:>8.5f}   {bar(p, max_p, 30)}")

    if len(peaks_k) >= 2:
        ratios = [peaks_k[i+1] / peaks_k[i] for i in range(len(peaks_k)-1)]
        print(f"\n  Power spectrum peaks at |k| = {[f'{k:.2f}' for k in peaks_k]}")
        print(f"  Peak spacing ratios: {[f'{r:.3f}' for r in ratios]}")
        golden = 1.6180339887
        for ratio in ratios:
            if abs(ratio - golden) < 0.08:
                print("  ✓ Peak spacing ratio ≈ φ — pure-point quasi-crystal signature!")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HexGo Theory — Headless Investigation")
    parser.add_argument("--games", type=int, default=200,
                        help="Games per question (default: 200)")
    parser.add_argument("--question", "-q", default="all",
                        choices=["all", "forks", "correlation", "patterns", "spectrum"],
                        help="Which question to investigate")
    args = parser.parse_args()

    n = args.games
    q = args.question

    print(f"\nHexGo Theory — Research Investigation")
    print(f"Games per question: {n}")
    print(f"Question: {q}")

    if q in ("all", "forks"):
        q_forks(n)
    if q in ("all", "correlation"):
        q_correlation(min(n, 100))  # correlation needs fewer games (expensive per-game)
    if q in ("all", "patterns"):
        q_patterns(n)
    if q in ("all", "spectrum"):
        q_spectrum(min(n, 50))   # Fourier is O(N²) in n_pts, keep manageable

    print()
    print("═" * 70)
    print("  Investigation complete.")
    print("═" * 70)


if __name__ == "__main__":
    main()
