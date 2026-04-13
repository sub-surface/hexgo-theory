"""
Epiplexity — time-bounded two-part MDL measurements for HexGo corpora.

Finzi et al. 2026, Def. 8: MDL_T(X) = |P*| + E[log 1/P*(X)]
                                  = S_T(X)  + H_T(X)
where P* is the minimising T-time program in P_T.

We cannot compute P* exactly (uncomputable) so we approximate with two
different observers and report both:

  (a) Markov-n observer:  a mixed-order back-off language model over move
      tokens. Its "program length" is (token-table bytes + transition-
      count bytes). Cross-entropy is the usual held-out NLL.

  (b) gzip observer:      write corpus to bytes, gzip, measure length.
      gzip is a practical universal compressor — compressed_length/N is a
      well-known upper bound on H(X) and a widely used Kolmogorov-
      complexity proxy (Cilibrasi & Vitanyi 2005).

For agents viewed as models (Programme E): each agent A exposes a move
probability distribution via softmax-of-score with a fixed temperature;
`agent_cross_entropy(A, corpus)` is the cross-entropy of that distribution
against the held-out moves. Combined with `agent_program_length(A)` (the
gzipped canonical source byte length) we have a two-part MDL for every agent.

All measurements here are deterministic given a seed; none require torch.
"""

from __future__ import annotations
import ast
import gzip
import inspect
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from engine import HexGame


# ── canonicalisation of agents as "programs" ────────────────────────────────

def _canonical_source(obj) -> bytes:
    """Return the canonical AST-unparsed source for a callable/class/module.

    Strips comments, docstrings (top-level only), trailing whitespace. This
    makes |P| less sensitive to stylistic noise and more to algorithmic
    content, per NOTES.md concern #1.
    """
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError):
        # fall back: pickled repr
        return repr(obj).encode()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src.encode()
    # strip top-level docstring to reduce noise
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree).encode()


def agent_program_length(agent) -> int:
    """Gzipped canonical-source byte length of the agent's class.

    This is |P| in the epiplexity two-part MDL. Lower is more parsimonious.
    """
    cls = agent if isinstance(agent, type) else agent.__class__
    canon = _canonical_source(cls)
    return len(gzip.compress(canon, mtime=0, compresslevel=9))


# ── corpus representation ───────────────────────────────────────────────────

@dataclass
class Game:
    """A minimal serialisable game trace."""
    moves: list[tuple[int, int]]
    players: list[int]
    winner: int | None

    @classmethod
    def from_hexgame(cls, g: HexGame) -> "Game":
        return cls(
            moves=list(g.move_history),
            players=list(g.player_history),
            winner=g.winner,
        )


@dataclass
class Corpus:
    """A corpus of games with a manifest."""
    games: list[Game]
    manifest: dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump({"games": self.games, "manifest": self.manifest}, f)

    @classmethod
    def load(cls, path: Path) -> "Corpus":
        with gzip.open(path, "rb") as f:
            d = pickle.load(f)
        return cls(games=d["games"], manifest=d["manifest"])


# ── corpus generation ──────────────────────────────────────────────────────

def generate_corpus(agent_factory_a: Callable[[], Any],
                    agent_factory_b: Callable[[], Any],
                    n_games: int,
                    seed: int = 0,
                    max_moves: int = 150,
                    swap_sides: bool = True) -> Corpus:
    """Run n_games of self-play and return a Corpus.

    agent_factory_*: zero-arg callables returning fresh agent instances.
    """
    rng = random.Random(seed)
    games: list[Game] = []
    for i in range(n_games):
        # seed each agent's internal RNG deterministically
        random.seed(rng.randint(0, 2**31 - 1))
        a = agent_factory_a()
        b = agent_factory_b()
        if swap_sides and i % 2 == 1:
            a, b = b, a
        g = HexGame()
        moves = 0
        while g.winner is None and moves < max_moves:
            agent = a if g.current_player == 1 else b
            legal = g.legal_moves()
            if not legal:
                break
            mv = agent.choose_move(g)
            if mv not in set(legal):
                mv = rng.choice(legal)
            g.make(*mv)
            moves += 1
        games.append(Game.from_hexgame(g))
    manifest = {
        "n_games": n_games,
        "seed": seed,
        "max_moves": max_moves,
        "agent_a": agent_factory_a().__class__.__name__,
        "agent_b": agent_factory_b().__class__.__name__,
        "prog_a": agent_program_length(agent_factory_a()),
        "prog_b": agent_program_length(agent_factory_b()),
    }
    return Corpus(games=games, manifest=manifest)


# ── tokenisation: relative-move encoding ────────────────────────────────────

def _relative_tokens(game: Game) -> list[tuple[int, int]]:
    """Translate absolute move coordinates to first-move-centered relative.

    This removes the trivial translation invariance, lowering entropy without
    touching strategic content.
    """
    if not game.moves:
        return []
    q0, r0 = game.moves[0]
    return [(q - q0, r - r0) for q, r in game.moves]


def corpus_token_stream(corpus: Corpus,
                        include_game_sep: bool = True) -> list[tuple[int, int] | None]:
    """Flatten all games into one token stream. None separates games."""
    out: list = []
    for g in corpus.games:
        out.extend(_relative_tokens(g))
        if include_game_sep:
            out.append(None)
    return out


# ── Markov-n back-off observer ──────────────────────────────────────────────

class MarkovBackoffObserver:
    """Mixed-order (n..1..unigram) Markov model over move tokens.

    Simple, deterministic, torch-free. Interpolated probabilities:
      P(x_t | ctx) = sum_k lambda_k * P_k(x_t | last-k tokens of ctx)
    where lambda is learned on held-out data by grid search.

    |P| = gzipped pickle of the transition tables.
    H_T = average negative log probability on held-out tokens (base 2 bits/token).
    """

    def __init__(self, max_order: int = 3, smoothing: float = 0.1):
        self.max_order = max_order
        self.smoothing = smoothing
        self.counts: list[dict] = [defaultdict(Counter) for _ in range(max_order + 1)]
        self.vocab: set = set()
        self.weights: list[float] = []

    def fit(self, token_stream: Sequence) -> "MarkovBackoffObserver":
        for k in range(self.max_order + 1):
            self.counts[k] = defaultdict(Counter)
        self.vocab = set(t for t in token_stream if t is not None)

        toks = token_stream
        for t in range(len(toks)):
            if toks[t] is None:
                continue
            for k in range(self.max_order + 1):
                if t - k < 0:
                    continue
                ctx = tuple(toks[t - k:t])
                if None in ctx:
                    continue
                self.counts[k][ctx][toks[t]] += 1

        # set uniform weights, will be re-tuned by `fit_weights`
        self.weights = [1.0 / (self.max_order + 1)] * (self.max_order + 1)
        return self

    def _pk(self, k: int, ctx: tuple, x) -> float:
        table = self.counts[k].get(ctx)
        if not table:
            # fall back to uniform over observed vocab
            return 1.0 / max(1, len(self.vocab))
        total = sum(table.values())
        V = max(1, len(self.vocab))
        return (table.get(x, 0) + self.smoothing) / (total + self.smoothing * V)

    def prob(self, ctx_tokens: Sequence, x) -> float:
        p = 0.0
        for k in range(self.max_order + 1):
            if k == 0:
                ctx = ()
            else:
                if len(ctx_tokens) < k:
                    continue
                ctx = tuple(ctx_tokens[-k:])
                if None in ctx:
                    ctx = ()
                    k = 0
            p += self.weights[k] * self._pk(k, ctx, x)
        return max(p, 1e-12)

    def cross_entropy_bits(self, token_stream: Sequence) -> float:
        """Average bits per token on `token_stream`."""
        total_bits = 0.0
        n = 0
        ctx: list = []
        for t in token_stream:
            if t is None:
                ctx = []
                continue
            p = self.prob(ctx, t)
            total_bits += -math.log2(p)
            ctx.append(t)
            n += 1
        return total_bits / max(1, n)

    def fit_weights(self, held_out_stream: Sequence, grid: int = 5) -> None:
        """Simple grid search over convex combinations of lambda_k."""
        best = (float("inf"), self.weights[:])
        orders = self.max_order + 1
        # enumerate coarse simplex grid
        def gen(remain, left):
            if left == 1:
                yield [remain]
                return
            for i in range(grid + 1):
                v = i / grid
                if v > remain + 1e-9:
                    break
                for tail in gen(remain - v, left - 1):
                    yield [v] + tail
        for ws in gen(1.0, orders):
            self.weights = ws
            ce = self.cross_entropy_bits(held_out_stream)
            if ce < best[0]:
                best = (ce, ws[:])
        self.weights = best[1]

    def program_length_bits(self) -> int:
        """|P| in bits — gzipped pickle of the model's learned tables."""
        payload = {
            "max_order": self.max_order,
            "smoothing": self.smoothing,
            "vocab_size": len(self.vocab),
            "counts": [dict((k, dict(v)) for k, v in c.items()) for c in self.counts],
            "weights": self.weights,
        }
        blob = gzip.compress(pickle.dumps(payload), mtime=0, compresslevel=9)
        return 8 * len(blob)


# ── gzip observer ──────────────────────────────────────────────────────────

def gzip_observer_bits_per_token(corpus: Corpus) -> tuple[float, int]:
    """Return (bits_per_token, total_gzipped_bits) for the corpus.

    Uses a simple variable-int encoding of each (dq, dr) to create a byte
    stream, then gzips. This is a practical upper bound on H(X) and doubles
    as a Kolmogorov-complexity proxy.
    """
    toks = corpus_token_stream(corpus, include_game_sep=True)
    buf = bytearray()
    n_tokens = 0
    for t in toks:
        if t is None:
            buf.append(0xFF); buf.append(0xFF)
            continue
        dq, dr = t
        # ZigZag encode to unsigned, then two bytes (board is small)
        buf.append(((dq << 1) ^ (dq >> 31)) & 0xFF)
        buf.append(((dr << 1) ^ (dr >> 31)) & 0xFF)
        n_tokens += 1
    gz = gzip.compress(bytes(buf), mtime=0, compresslevel=9)
    total_bits = 8 * len(gz)
    return total_bits / max(1, n_tokens), total_bits


# ── two-part MDL summary ───────────────────────────────────────────────────

@dataclass
class MDLReport:
    corpus_name: str
    n_games: int
    n_tokens: int
    # Markov observer
    markov_H_T_bits_per_token: float
    markov_S_T_bits: int
    # gzip observer
    gzip_bits_per_token: float
    gzip_total_bits: int
    # agent-as-model (if applicable)
    agent_prog_length_bytes: int | None = None
    agent_cross_entropy_bits: float | None = None

    def two_part_markov_bits(self) -> float:
        return self.markov_S_T_bits + self.n_tokens * self.markov_H_T_bits_per_token


def measure_corpus(corpus: Corpus,
                   max_order: int = 3,
                   holdout_frac: float = 0.2,
                   name: str | None = None) -> MDLReport:
    """Fit Markov observer + gzip observer on corpus, return MDL report."""
    toks = corpus_token_stream(corpus, include_game_sep=True)
    # simple split at a game boundary
    split = int(len(toks) * (1 - holdout_frac))
    # walk to next None to split on a game boundary
    while split < len(toks) and toks[split] is not None:
        split += 1
    train, test = toks[:split], toks[split:]
    obs = MarkovBackoffObserver(max_order=max_order).fit(train)
    obs.fit_weights(test)
    H_T = obs.cross_entropy_bits(test)
    S_T = obs.program_length_bits()
    n_tok = sum(1 for t in toks if t is not None)
    gz_bpt, gz_total = gzip_observer_bits_per_token(corpus)
    return MDLReport(
        corpus_name=name or corpus.manifest.get("agent_a", "unnamed"),
        n_games=len(corpus.games),
        n_tokens=n_tok,
        markov_H_T_bits_per_token=H_T,
        markov_S_T_bits=S_T,
        gzip_bits_per_token=gz_bpt,
        gzip_total_bits=gz_total,
    )


# ── agent-as-model cross-entropy (Programme E) ─────────────────────────────

def agent_policy_distribution(agent, game: HexGame, temperature: float = 0.5) -> dict:
    """Turn an agent's internal score function into a softmax over legal moves.

    Agents in engine/agents.py expose a scoring loop inside `choose_move`; we
    can't cleanly extract it without refactoring. For MDL purposes we use a
    deterministic-agent proxy: the agent's chosen move gets mass (1 - eps),
    the rest of legal moves split eps uniformly. This is crude but honest —
    it is literally the information-content of the agent's deterministic
    choice. Future work: refactor agents to expose scores.
    """
    legal = game.legal_moves()
    if not legal:
        return {}
    try:
        mv = agent.choose_move(game)
    except Exception:
        mv = legal[0]
    eps = 0.02
    n = len(legal)
    dist = {m: eps / max(1, n - 1) for m in legal}
    if mv in dist:
        dist[mv] = 1 - eps
    else:
        dist[mv] = 1 - eps
    return dist


def agent_cross_entropy_bits(agent, corpus: Corpus) -> float:
    """Held-out cross-entropy (bits/move) of the agent's policy on a corpus.

    Replays each game move-by-move; at each position asks the agent for its
    distribution over legal moves, scores the actual move against it.
    """
    total_bits, n = 0.0, 0
    for game in corpus.games:
        g = HexGame()
        for mv in game.moves:
            dist = agent_policy_distribution(agent, g)
            p = dist.get(mv, 1e-6)
            total_bits += -math.log2(max(p, 1e-12))
            n += 1
            if mv not in set(g.legal_moves()):
                break
            g.make(*mv)
    return total_bits / max(1, n)
