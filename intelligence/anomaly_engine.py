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
        self.history = deque(maxlen=config.BASELINE_WINDOW)

    def process(self, vector):

        self.step += 1

        similarity = self.engine.search(vector)

        # ======================
        # WARMUP PHASE
        # Store only if similarity looks normal (not a spike).
        # Uses a loose threshold: more than 2 std below current history mean.
        # On the very first few steps history is empty so we always store.
        # ======================
        if self.step <= config.WARMUP_STEPS:
            if len(self.history) >= 5:
                arr = np.array(self.history)
                mean, std = arr.mean(), max(arr.std(), 1e-6)
                is_spike = (similarity - mean) / std < config.ZSCORE_ANOMALY_THRESHOLD
            else:
                is_spike = False

            if not is_spike:
                self.engine.store(vector)

            self.history.append(similarity)

            return AnomalyResult(self.step, similarity, False, "WARMUP")

        # ======================
        # STABILIZATION PHASE
        # Observe without storing or flagging anomalies.
        # ======================
        if self.step <= config.STABILIZATION_STEPS:
            self.history.append(similarity)
            return AnomalyResult(self.step, similarity, False, "STABILIZING")

        # ======================
        # NEED ENOUGH DATA
        # ======================
        if len(self.history) < 10:
            self.history.append(similarity)
            return AnomalyResult(self.step, similarity, False, "COLLECTING")

        # ======================
        # DYNAMIC BASELINE (Z-SCORE)
        # ======================
        arr = np.array(self.history)
        mean = arr.mean()
        std = max(arr.std(), 1e-6)

        z_score = (similarity - mean) / std
        is_anomaly = z_score < config.ZSCORE_ANOMALY_THRESHOLD

        # ======================
        # SAFE LEARNING
        # Learn only confidently normal vectors after stabilization.
        # ======================
        if not is_anomaly and z_score > config.ZSCORE_LEARN_THRESHOLD:
            self.engine.store(vector)

        self.history.append(similarity)

        return AnomalyResult(self.step, similarity, is_anomaly, f"Z={z_score:.2f}")
