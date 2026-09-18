"""Exact k-means on Apple Silicon GPUs (MLX + custom Metal kernels).

    from mlx_kmeans import KMeans
    km = KMeans(n_clusters=64).fit(X)        # X: (rows, dims) float32
    km.cluster_centers_, km.labels_, km.inertia_

Every label is the center with the smallest float32 distance sum((x - c)^2), verified against float64 by
tests/accuracy.py. See README.md for benchmarks against scikit-learn, FAISS and PyTorch-MPS implementations.
"""
from .core import (  # noqa: F401
    assign_mlx, assign_numpy, kmeans, kmeans_pp_init, lloyd_step, make_data_mlx, make_data_numpy,
    slice_rows, take_rows,
)
from .core import _accumulate, _gpu_pass, _kernel, _mx, _nearest, _u32  # noqa: F401  (used by tests/benchmarks)
from .core import _ACCUMULATE_SRC, _KAHAN, _PAIRS_SRC, _ROWS_SRC, _SEGMENT_SRC  # noqa: F401
from .estimator import KMeans  # noqa: F401

__version__ = "0.1.0"
__all__ = ["KMeans", "kmeans", "lloyd_step", "assign_mlx", "kmeans_pp_init", "make_data_mlx"]
