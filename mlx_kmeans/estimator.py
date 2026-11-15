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
    at about 630x the error. Labels stop being provably optimal: on GIST1M the float16 path gives about 1.2% of
    rows a different centre per pass, at 2.4e-5 of the inertia (the float32 path far less), and when the
    float16 path runs the inertia and the new centres are summed from the float16 copy as well (7e-7 and
    2.7e-4 relative on GIST1M; APPROX_FP16_ACCUMULATE=False in core.py sums them from the float32 data). The
    fit still lands where the exact one does: on GIST1M at k=1024 it is 2.68x faster for 0.003% more inertia
    and recall@10 through a FAISS IVF index the same to within noise (92.43% against 92.44% at nprobe 32).
    scripts/compare_exact_approx.py reproduces that. A fit at 256+ dims also keeps per-centre distance bounds
    between iterations on the exact path (core.py, section 5); labels stay exact, and only ties can resolve
    differently from a single pass.

    accumulate="atomic" sums cluster totals in threadgroup memory: ~1.7-4.5x faster at small k * dims, at the cost
    of bit-reproducibility (centers move by ~1e-8 relative between identical runs, which can change which local
    minimum a run reaches). The default is deterministic.

    X may be anything array-like - a NumPy array, a pandas DataFrame, a list of rows - or, for data larger than one
    GPU buffer, a list of MLX arrays of at most ~100M rows each.

    algorithm="emeans" (opt-in, approximate) fits the centres with e-means instead (arXiv:2611.09501; see
    docs/e-means.md and mlx_kmeans/emeans.py): mini-batch k-means with moving averages, run for a fixed number
    of `epochs` with the batch size and step from the paper's scaling laws (`batch_size` overrides the law),
    seeded with k random rows. Its centres are not a fixed point of Lloyd's iteration. labels_ and inertia_ still
    come from the same final pass over the returned centres as Lloyd's (exact, or the matmul one with
    exact=False); n_iter_ is the epoch count;
    max_iter and tol do not apply; emeans_record_ holds the run's record (batch size, step, losses, timings).
    exact=False makes its batch assignment the matmul one as well. spherical=True and accumulate="atomic" are
    not offered with it. The default, algorithm="lloyd", is the exact algorithm above and is unchanged.
    """

    def __init__(self, n_clusters=8, n_init=1, max_iter=300, tol=1e-4, random_state=None, verbose=False,
                 spherical=False, accumulate="auto", exact=True, algorithm="lloyd", epochs=3, batch_size=None):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.spherical = spherical
        self.accumulate = accumulate
        self.exact = exact
        self.algorithm = algorithm
        self.epochs = epochs
        self.batch_size = batch_size

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
            per = core.slice_rows(X.shape[1]) if X.ndim == 2 else len(X)
            parts = [X] if len(X) <= per else [X[s:s + per] for s in range(0, len(X), per)]
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

    # e-means takes its own deterministic accumulators; "sorted" maps to the same segment kernels planned on the GPU.
    _EMEANS_ACCUMULATE = {"auto": "auto", "lanes": "lanes", "blocks": "blocks", "sorted": "segments"}

    def _check_emeans(self):
        algorithm = getattr(self, "algorithm", "lloyd")      # an estimator pickled before the option existed
        if algorithm not in ("lloyd", "emeans"):
            raise ValueError(f"algorithm must be 'lloyd' or 'emeans', got {algorithm!r}")
        if algorithm != "emeans":
            return
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, (int, np.integer)) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if self.batch_size is not None and (isinstance(self.batch_size, bool) or
                                            not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1):
            raise ValueError("batch_size must be None or a positive integer")
        if self.spherical:
            raise ValueError("spherical=True is not offered with algorithm='emeans'")
        if self.accumulate not in self._EMEANS_ACCUMULATE:
            raise ValueError(f"accumulate={self.accumulate!r} is not offered with algorithm='emeans' "
                             f"(use one of {sorted(self._EMEANS_ACCUMULATE)})")

    def _one_run_emeans(self, parts, rows, seed):
        from . import emeans
        if self.batch_size is not None and self.batch_size > rows:
            raise ValueError(f"batch_size ({self.batch_size}) cannot exceed the number of rows ({rows})")
        return emeans.emeans_fit(parts, self.n_clusters, epochs=self.epochs, seed=seed, batch_size=self.batch_size,
                                 assign="exact" if self.exact else "approx",
                                 accumulate=self._EMEANS_ACCUMULATE[self.accumulate], verbose=self.verbose)

    def fit(self, X, y=None):
        for name in ['n_clusters', 'n_init', 'max_iter']:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError('tol must be finite and nonnegative')
        self._check_emeans()
        parts = self._parts(X)
        rows = sum(p.shape[0] for p in parts)
        if rows < self.n_clusters:
            raise ValueError('n_clusters cannot exceed the number of rows')
        emeans = getattr(self, "algorithm", "lloyd") == "emeans"
        best = None
        for run in range(self.n_init):                  # restarts reuse the data already on the GPU
            seed = None if self.random_state is None else self.random_state + run
            if emeans:
                C, record = self._one_run_emeans(parts, rows, seed)
                iters = self.epochs
            else:
                C, inertia, iters = self._one_run(parts, rows, seed)
            # A Lloyd step reports inertia before its center update. Score the final centers so restarts,
            # labels_, and inertia_ all describe the same fitted model.
            _, _, inertia, labels = core.assign_mlx(parts, C, method=self._method, return_labels=True,
                                                    accumulate=self.accumulate)
            if emeans and not np.isfinite(inertia):
                self._explain_nonfinite(parts)
            if self.verbose and self.n_init > 1:
                print(f"start {run + 1}/{self.n_init}: inertia {inertia:.6e} after {iters} "
                      f"{'epochs' if emeans else 'iterations'}")
            if best is None or inertia < best[1]:
                best = (C, inertia, iters, labels) + ((record,) if emeans else ())
        self.cluster_centers_, self.inertia_, self.n_iter_, self.labels_ = best[:4]
        if emeans:
            self.emeans_record_ = best[4]
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
