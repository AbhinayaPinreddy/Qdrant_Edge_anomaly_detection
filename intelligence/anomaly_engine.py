import numpy as np
from collections import deque
from dataclasses import dataclass
import config


@dataclass
class AnomalyResult:
    step: int
    similarity: float
    is_anomaly: bool
    reason: str


class AnomalyDetector:

    def __init__(self, engine):
        self.engine = engine
        self.step = 0

        # Rolling window for dynamic baseline
        self.history = deque(maxlen=50)

        # Stabilization phase (no learning)
        self.stabilization_steps = 100

    def process(self, vector):

        self.step += 1

        similarity = self.engine.search(vector)

        # ======================
        # WARMUP PHASE
        # ======================
        if self.step <= config.WARMUP_STEPS:
            self.engine.store(vector)
            self.history.append(similarity)

            return AnomalyResult(
                self.step, similarity, False, "WARMUP"
            )

        # ======================
        # STABILIZATION PHASE
        # ======================
        if self.step <= self.stabilization_steps:
            self.history.append(similarity)

            return AnomalyResult(
                self.step, similarity, False, "STABILIZING"
            )

        # ======================
        # NEED ENOUGH DATA
        # ======================
        if len(self.history) < 10:
            self.history.append(similarity)

            return AnomalyResult(
                self.step, similarity, False, "COLLECTING"
            )

        # ======================
        # DYNAMIC BASELINE
        # ======================
        arr = np.array(self.history)

        mean = arr.mean()
        std = arr.std()

        if std < 1e-6:
            std = 1e-6

        # ======================
        # Z-SCORE DETECTION
        # ======================
        z_score = (similarity - mean) / std

        # STRONGER anomaly condition
        is_anomaly = z_score < -2.5

        # ======================
        # SAFE LEARNING LOGIC
        # ======================
        # Learn ONLY if:
        # - Not anomaly
        # - High confidence (far from boundary)
        # - After stabilization phase
        if (
            not is_anomaly
            and z_score > -1.0   # confidence filter
            and self.step > self.stabilization_steps
        ):
            self.engine.store(vector)

        # Update history AFTER decision
        self.history.append(similarity)

        reason = f"Z={z_score:.2f}"

        return AnomalyResult(
            self.step,
            similarity,
            is_anomaly,
            reason
        )