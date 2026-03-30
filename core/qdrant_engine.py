import os
import shutil
import numpy as np

from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeVectorParams,
    EdgeShard,
    Point,
    UpdateOperation,
    Query,
    QueryRequest,
)

import config


def _shard_exists(path):
    return os.path.isdir(path) and any(os.scandir(path))


class QdrantEdgeEngine:

    def __init__(self, fresh: bool = True):
        path = config.QDRANT_SHARD_PATH

        cfg = EdgeConfig(
            vectors={
                config.VECTOR_NAME: EdgeVectorParams(
                    size=config.VECTOR_SIZE,
                    distance=Distance.Cosine,
                )
            }
        )

        if fresh and _shard_exists(path):
            shutil.rmtree(path)

        os.makedirs(path, exist_ok=True)

        if _shard_exists(path) and not fresh:
            self._shard = EdgeShard.load(path, cfg)
        else:
            self._shard = EdgeShard.create(path, cfg)

        self._count = 0
        self._id_counter = 0

    def store(self, vector: np.ndarray):
        self._id_counter += 1

        point = Point(
            id=self._id_counter,
            vector={config.VECTOR_NAME: vector.tolist()},
        )

        self._shard.update(UpdateOperation.upsert_points([point]))
        self._count += 1

    def search(self, vector: np.ndarray) -> float:
        if self._count == 0:
            return 1.0

        k = min(config.TOP_K_SEARCH, self._count)

        req = QueryRequest(
            query=Query.Nearest(
                query=vector.tolist(),
                using=config.VECTOR_NAME
            ),
            limit=k,
        )

        results = self._shard.query(req)

        if not results:
            return 1.0

        # Average top-k scores for a more robust similarity estimate
        return float(np.mean([r.score for r in results]))

    def flush(self):
        self._shard.flush()

    def close(self):
        self._shard.close()
