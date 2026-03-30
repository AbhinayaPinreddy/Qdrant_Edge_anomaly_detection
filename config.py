QDRANT_SHARD_PATH = "qdrant_data"

VECTOR_NAME = "sensor_vector"
VECTOR_SIZE = 15   # 3 sensors × 5 features

WARMUP_STEPS = 80
STABILIZATION_STEPS = 100       # must be > WARMUP_STEPS

ZSCORE_ANOMALY_THRESHOLD = -2.5 # flag anomaly if z-score drops below this
ZSCORE_LEARN_THRESHOLD = -1.0   # only learn if z-score is above this (confident normal)

BASELINE_WINDOW = 50
TOP_K_SEARCH = 5

CHART_SIMILARITY_REF_LINE = 0.97  # visual reference line on chart (not used in detection)
