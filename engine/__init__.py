"""
Bridge to the hexgo game engine.
Adds the hexgo project root to sys.path so we can import game.py and elo.py directly.
"""
import sys
from pathlib import Path

HEXGO_ROOT = Path(__file__).parent.parent.parent / "hexgo"
if str(HEXGO_ROOT) not in sys.path:
    sys.path.insert(0, str(HEXGO_ROOT))

from game import HexGame, AXES, WIN_LENGTH  # noqa: F401
from elo import EisensteinGreedyAgent, RandomAgent  # noqa: F401
from engine.agents import ForkAwareAgent, PotentialGradientAgent, ComboAgent  # noqa: F401
