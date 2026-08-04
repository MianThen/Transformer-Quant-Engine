"""Independent research oracles that never participate in production replay."""

from .portfolio_math_oracle import build_phase1a_fixture
from .hypothesis_registry import Hypothesis, HypothesisRegistry

__all__ = ["Hypothesis", "HypothesisRegistry", "build_phase1a_fixture"]
