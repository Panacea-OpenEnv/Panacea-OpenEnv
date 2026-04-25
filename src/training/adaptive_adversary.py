"""
Adaptive deception sampler — the non-stationary half of the self-play loop.

Tracks the oversight agent's per-type detection rate over a rolling window
and reweights the next episode's deception distribution: the easier a type
is to catch, the rarer it becomes. The agent therefore cannot converge on
a single tactic — it must maintain a world model that generalizes.
"""

import random
from collections import deque
from typing import Deque, Optional


DECEPTION_TYPES = ["ghost", "inflation", "masking", "collusion", "none"]


class AdaptiveDeceptionSampler:
    """Tracks detection rates and emits deception_type for the next episode."""

    def __init__(
        self,
        window: int = 50,
        min_weight: float = 0.05,
        none_weight: float = 0.25,
        seed: Optional[int] = None,
    ):
        self.window = window
        self.min_weight = min_weight
        self.none_weight = none_weight
        # Rolling per-type results — 1 if agent caught the deception, 0 if missed
        self._history: dict[str, Deque[int]] = {
            t: deque(maxlen=window) for t in DECEPTION_TYPES
        }
        self._rng = random.Random(seed)

    # ── Update ───────────────────────────────────────────────────────────────

    def record(self, deception_type: str, detected: bool) -> None:
        """Log one episode outcome.

        For deceptive types (ghost/inflation/masking/collusion), `detected`
        means the agent emitted REJECT. For 'none', `detected` should be
        False if the agent correctly approved (we don't reweight 'none' from
        outcomes — its share is fixed by `none_weight`).
        """
        if deception_type not in self._history:
            return
        self._history[deception_type].append(1 if detected else 0)

    # ── Sample ───────────────────────────────────────────────────────────────

    def detection_rates(self) -> dict[str, float]:
        rates = {}
        for t, buf in self._history.items():
            rates[t] = (sum(buf) / len(buf)) if buf else 0.0
        return rates

    def weights(self) -> dict[str, float]:
        """Reweight types: high detection → low weight."""
        rates = self.detection_rates()
        deceptive = [t for t in DECEPTION_TYPES if t != "none"]

        raw = {t: max(self.min_weight, 1.0 - rates[t]) for t in deceptive}
        s = sum(raw.values())
        # Normalize the deceptive slice to (1 - none_weight)
        share = (1.0 - self.none_weight)
        weights = {t: (raw[t] / s) * share for t in deceptive}
        weights["none"] = self.none_weight
        return weights

    def sample(self) -> str:
        w = self.weights()
        types = list(w.keys())
        probs = [w[t] for t in types]
        return self._rng.choices(types, weights=probs, k=1)[0]

    # ── Telemetry ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "detection_rates": self.detection_rates(),
            "weights": self.weights(),
            "samples": {t: len(buf) for t, buf in self._history.items()},
        }
