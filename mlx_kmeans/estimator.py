"""scikit-learn-shaped wrapper around the GPU k-means in core.py."""
import numpy as np

from . import core


class KMeans:
    """K-means on the Apple GPU, with exact assignments.

        KMeans(n_clusters=64).fit(X).cluster_centers_

    Parameters mirror scikit-learn where the meaning is the same: n_clusters, n_init, max_iter, tol (relative
    inertia change), random_state. Empty clusters are relocated exactly as scikit-learn does.

    spherical=True runs spherical k-means (centers re-normalised to unit length each iteration), which is what you
    want for text/image embeddings compared by cosine similarity - give it L2-normalised rows.

    exact=False allows an approximate assignment: distances from a matmul in the expanded form, argmin taken
    on trust. Where the exact paths are faster as well as exact the flag is ignored, and where that is depends
    on the data: if its norms let the multiply run in float16 (|x||c| well inside float16's range - true of
    normalised embeddings, not of raw SIFT) the approximate path wins from 64 dimensions (1.14x at 64, 1.34x
    at 96, 2.68x on a GIST1M fit at 960); if not, it wins from 256 (1.97x at 256, 2.16x at 960 in fp32).
    Below those the flag does nothing. The reason is that MLX's matmul reaches hardware these kernels cannot,
    at about 630x the error. Labels stop being provably optimal - about 0.15% of rows get a different
    centre in a pass, costing 1e-7 of the distance - and inertia_ is still the TRUE inertia of the labels
    returned, so an approximate run stays comparable with an exact one. On GIST1M at k=1024 it fits 2.68x
    faster for 0.003% more inertia and recall@10 through a FAISS IVF index that is the same to within noise
    (92.43% against 92.44% at nprobe 32). scripts/compare_exact_approx.py reproduces that.

    accumulate="atomic" sums cluster totals in threadgroup memory: ~1.7-4.5x faster at small k * dims, at the cost
    of bit-reproducibility (centers move by ~1e-8 relative between identical runs, which can change which local
    minimum a run reaches). The default is deterministic.

    X may be anything array-like - a NumPy array, a pandas DataFrame, a list of rows - or, for data larger than one
    GPU buffer, a list of MLX arrays of at most ~100M rows each.
    """

    def __init__(self, n_clusters=8, n_init=1, max_iter=300, tol=1e-4, random_state=None, verbose=False,
                 spherical=False, accumulate="auto", exact=True):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.spherical = spherical
        self.accumulate = accumulate
        self.exact = exact

    @property
    def _method(self):
        return "auto" if self.exact else "approx"

    @classmethod
    def from_centers(cls, centers, **kwargs):
        """A fitted model from centers you saved earlier: KMeans.from_centers(np.load("centers.npy"))."""
        centers = np.ascontiguousarray(centers, dtype=np.float32)
        if not np.isfinite(centers).all():
            raise ValueError("centers contain NaN or infinity")
        model = cls(n_clusters=len(centers), **kwargs)
        model.cluster_centers_ = centers
        return model

    @staticmethod
    def _explain_nonfinite(parts):
        """Say why a pass came back non-finite, then raise. Only called once something has already failed.

        The kernels accumulate a squared distance in float32, so a NaN or infinity in the data, or coordinates
        large enough for sum((x - c)^2) to overflow, used to surface as nan centers rather than as an error.
        Every such case makes the first pass's inertia non-finite, which costs nothing to notice; this runs
        only then, and one reduction distinguishes the three. max(|x|) is nan if any value is NaN and inf if
        any is infinite, and its size decides the overflow case: the worst a pass can produce is sum over dims
        of (2*max)^2, so the largest safe magnitude is sqrt(float32_max / (4 * dims)).
        """
        mx = core._mx()
        d = parts[0].shape[1]
        peak = max(float(mx.max(mx.abs(p))) for p in parts)
        if not np.isfinite(peak):
            raise ValueError(
                "input contains NaN or infinity, which would produce nan centers rather than an error. "
                "Drop or impute those rows first (the file workflow's missing='drop' does this).")
        limit = float(np.sqrt(np.finfo(np.float32).max / (4.0 * d)))
        if peak > limit:
            raise ValueError(
                f"largest coordinate is {peak:.3e}, above the {limit:.3e} this many dimensions allow: "
                f"a squared distance would overflow float32 and return nan. Standardise the features "
                f"(see the README) or rescale before fitting.")
        raise ValueError("clustering produced a non-finite result, but the input values look usable; "
                         "please report this with the data shape and dtype.")

    @staticmethod
    def _parts(X):
        """-> list of MLX arrays, one per slice of the data."""
        mx = core._mx()
        if isinstance(X, list) and X and type(X[0]).__module__.startswith("mlx"):
            parts = X                                     # already slices on the GPU
        elif type(X).__module__.startswith("mlx"):
            parts = [X]
        else:
            if not isinstance(X, np.ndarray):
                X = np.asarray(X.to_numpy() if hasattr(X, "to_numpy") else X)   # pandas/polars/lists
            X = np.ascontiguousarray(X, dtype=np.float32)
            if X.ndim != 2 or not all(X.shape):
                raise ValueError(f"expected a nonempty 2-D (rows, dims) array, got shape {X.shape}")
            per = core.slice_rows(X.shape[1])
            parts = [mx.array(X[s:s + per]) for s in range(0, len(X), per)]
        return parts

    @staticmethod
    def _normalise(C):
        n = np.linalg.norm(C, axis=1, keepdims=True)
        return (C / np.where(n > 0, n, 1)).astype(np.float32)

    def _one_run(self, parts, rows, seed):
        rng = np.random.default_rng(seed)
        C = core.kmeans_pp_init(parts, rows, self.n_clusters, rng)
        if self.spherical:
            C = self._normalise(C)
        prev = None
        for it in range(self.max_iter):
            C, inertia, n_empty = core.lloyd_step(parts, C, method=self._method,
                                                  accumulate=self.accumulate)
            if not np.isfinite(inertia):            # free: the data, not the algorithm, is the usual cause
                self._explain_nonfinite(parts)
            if self.spherical:
                C = self._normalise(C)
            if self.verbose:
                print(f"iter {it + 1}: inertia {inertia:.6e}" + (f", {n_empty} empty relocated" if n_empty else ""))
            if prev is not None and abs(prev - inertia) <= self.tol * prev:
                break
            prev = inertia
        return C, inertia, it + 1

    def fit(self, X, y=None):
        for name in ['n_clusters', 'n_init', 'max_iter']:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError('tol must be finite and nonnegative')
        parts = self._parts(X)
        rows = sum(p.shape[0] for p in parts)
        if rows < self.n_clusters:
            raise ValueError('n_clusters cannot exceed the number of rows')
        best = None
        for run in range(self.n_init):                  # restarts reuse the data already on the GPU
            seed = None if self.random_state is None else self.random_state + run
            C, inertia, iters = self._one_run(parts, rows, seed)
            # A Lloyd step reports inertia before its center update. Score the final centers so restarts,
            # labels_, and inertia_ all describe the same fitted model.
            _, _, inertia, labels = core.assign_mlx(parts, C, method=self._method, return_labels=True,
                                                    accumulate=self.accumulate)
            if self.verbose and self.n_init > 1:
                print(f"start {run + 1}/{self.n_init}: inertia {inertia:.6e} after {iters} iterations")
            if best is None or inertia < best[1]:
                best = (C, inertia, iters, labels)
        self.cluster_centers_, self.inertia_, self.n_iter_, self.labels_ = best
        return self

    def predict(self, X):
        """Cluster index for every row of X (any array-like)."""
        parts = self._parts(X)
        if parts[0].shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(f"X has {parts[0].shape[1]} features but this model was fitted with "
                             f"{self.cluster_centers_.shape[1]}")
        _, _, inertia, labels = core.assign_mlx(parts, self.cluster_centers_, method=self._method,
                                                return_labels=True, accumulate=self.accumulate)
        if not np.isfinite(inertia):
            self._explain_nonfinite(parts)
        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
