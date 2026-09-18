"""scikit-learn-shaped wrapper around the GPU k-means in core.py."""
import numpy as np

from . import core


class KMeans:
    """K-means on the Apple GPU, with exact assignments.

        KMeans(n_clusters=64).fit(X).cluster_centers_

    Parameters mirror scikit-learn where the meaning is the same: n_clusters, max_iter, tol (relative inertia
    change), random_state. Empty clusters are relocated exactly as scikit-learn does.

    accumulate="atomic" sums cluster totals in threadgroup memory: ~1.7-4.5x faster at small k * dims, at the cost
    of bit-reproducibility (centers move by ~1e-8 relative between identical runs, which can change which local
    minimum a run reaches). The default is deterministic.

    X may be a (rows, dims) float32 NumPy array, an MLX array, or a list of MLX arrays (slices of at most
    ~100M rows each), which is how datasets larger than one buffer are held.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None, verbose=False, accumulate="auto"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.accumulate = accumulate

    @staticmethod
    def _parts(X):
        mx = core._mx()
        if isinstance(X, list):
            return X
        if isinstance(X, np.ndarray):
            X = np.ascontiguousarray(X, dtype=np.float32)
            per = core.slice_rows(X.shape[1])
            return [mx.array(X[s:s + per]) for s in range(0, len(X), per)]
        return [X]

    def fit(self, X, y=None):
        parts = self._parts(X)
        rows = sum(p.shape[0] for p in parts)
        rng = np.random.default_rng(self.random_state)
        C = core.kmeans_pp_init(parts, rows, self.n_clusters, rng)
        prev = None
        for it in range(self.max_iter):
            C, inertia, n_empty = core.lloyd_step(parts, C, accumulate=self.accumulate)
            if self.verbose:
                print(f"iter {it + 1}: inertia {inertia:.6e}" + (f", {n_empty} empty relocated" if n_empty else ""))
            if prev is not None and abs(prev - inertia) <= self.tol * prev:
                break
            prev = inertia
        self.cluster_centers_ = C
        self.inertia_ = inertia
        self.n_iter_ = it + 1
        self._parts_cache = parts
        return self

    @property
    def labels_(self):
        """Cluster index for every row of the fitted data."""
        return self.predict(self._parts_cache)

    def predict(self, X):
        parts = self._parts(X)
        return core.assign_mlx(parts, self.cluster_centers_, return_labels=True, accumulate=self.accumulate)[3]

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
