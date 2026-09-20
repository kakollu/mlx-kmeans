#!/usr/bin/env python3
"""RAM-budgeted real-transaction benchmark. Use --plan or --cleanup without running a benchmark."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data' / 'air-benchmark'
OWNER = 'mlx-kmeans air benchmark cache v1\n'
NAME = 'yellow_tripdata_2015-01.parquet'
URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/' + NAME
MIB, GIB = 2**20, 2**30
FIXED_BYTES = 256 * MIB
BYTES_PER_ROW = 320  # input, copies, float64 preprocessing, labels and working space, with headroom
MAX_DOWNLOAD = 256 * MIB
FEATURES = ['distance_miles', 'duration_minutes', 'fare_dollars', 'tip_percent', 'hour', 'passengers']
COLUMNS = ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'fare_amount',
           'tip_amount', 'passenger_count']


def plan_rows(total, available, gpu_limit, requested=5_000_000, memory_mib=None):
    budget = int(min(total * .25, available * .4, gpu_limit * .25, 2 * GIB))
    if memory_mib is not None:
        budget = min(budget, int(memory_mib * MIB))
    rows = min(requested, max(0, (budget - FIXED_BYTES) // BYTES_PER_ROW))
    if rows < 10_000:
        raise ValueError('Insufficient free memory for even 10,000 rows. Close apps and retry; nothing downloaded.')
    return dict(rows=rows, memory_budget_bytes=budget, estimated_working_bytes=FIXED_BYTES + rows*BYTES_PER_ROW,
                input_bytes=rows*6*4, total_ram_bytes=total, available_ram_bytes=available,
                gpu_recommended_bytes=gpu_limit)


def owned_cache(cache, create=False):
    # Resolve neither symlinks nor an arbitrary caller-supplied deletion tree.
    if any(p.is_symlink() for p in [cache, *cache.parents]):
        raise ValueError('Refusing a symlinked benchmark cache path.')
    marker = cache / '.owner'
    if not cache.exists() and create:
        cache.mkdir(parents=True)
        marker.write_text(OWNER)
    if not cache.exists():
        return False
    if marker.is_symlink() or not marker.is_file() or marker.read_text() != OWNER:
        raise ValueError(f'Unrecognized cache at {cache}; leaving it untouched.')
    return True


def cleanup(cache=CACHE):
    if not owned_cache(cache):
        print('No benchmark download to clean up.')
        return
    for name in [NAME, NAME+'.partial', 'download.json']:
        path = cache / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f'Refusing unexpected cache entry: {path}')
    removed = 0
    for name in [NAME, NAME+'.partial', 'download.json']:
        path = cache / name
        if path.exists():
            removed += path.stat().st_size
            path.unlink()
    if set(p.name for p in cache.iterdir()) == {'.owner'}:
        (cache / '.owner').unlink()
        cache.rmdir()
    print(f'Removed {removed/MIB:.1f} MiB of benchmark cache (can be downloaded again). Reports preserved.')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(MIB), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(cache=CACHE):
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    owned_cache(cache, create=True)
    dest, partial, manifest = cache/NAME, cache/(NAME+'.partial'), cache/'download.json'
    for p in [dest, partial, manifest]:
        if p.is_symlink() or (p.exists() and not p.is_file()):
            raise ValueError(f'Unexpected cache entry: {p}')
    if dest.exists():
        info = json.loads(manifest.read_text()) if manifest.exists() else {}
        if info.get('url') == URL and info.get('bytes') == dest.stat().st_size and info.get('sha256') == file_hash(dest):
            print('Using verified cached taxi file.')
            return dest, info
        raise ValueError('Cached file is incomplete or changed. Run --cleanup, then retry.')
    with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=30, context=context) as r:
        size = int(r.headers.get('Content-Length', 0))
    if not 0 < size <= MAX_DOWNLOAD:
        raise ValueError(f'Download size {size} is unknown or exceeds the {MAX_DOWNLOAD//MIB} MiB cap; stopped.')
    if shutil.disk_usage(cache).free < size*2 + GIB:
        raise ValueError('Not enough free disk space for download plus a 1 GiB reserve.')
    print(f'Downloading one taxi month ({size/MIB:.1f} MiB); it will be read in batches, not loaded in full.', flush=True)
    if partial.exists():
        partial.unlink()  # only this owned cache's known incomplete download
    count, digest = 0, hashlib.sha256()
    try:
        with urllib.request.urlopen(URL, timeout=30, context=context) as r, partial.open('xb') as f:
            while True:
                block = r.read(MIB)
                if not block:
                    break
                count += len(block)
                if count > size:
                    raise ValueError('Server sent more data than advertised; download stopped.')
                f.write(block)
                digest.update(block)
        if count != size:
            raise ValueError('Incomplete download; retry.')
        info = dict(url=URL, bytes=count, sha256=digest.hexdigest())
        partial.replace(dest)
        manifest.write_text(json.dumps(info, indent=2))
        return dest, info
    finally:
        if partial.exists():
            partial.unlink()


def load_features(path, limit):
    import numpy as np
    import pyarrow.parquet as pq
    X = np.empty((limit, 6), dtype=np.float32)
    kept, scanned = 0, 0
    with pq.ParquetFile(path) as reader:
        for batch in reader.iter_batches(batch_size=32_768, columns=COLUMNS, use_threads=False):
            values = {name: batch.column(i).to_numpy(zero_copy_only=False) for i, name in enumerate(COLUMNS)}
            pick = values[COLUMNS[0]].astype('datetime64[s]')
            drop = values[COLUMNS[1]].astype('datetime64[s]')
            duration = (drop-pick).astype('float32') / 60
            distance, fare, tip, pax = (values[n].astype('float32') for n in COLUMNS[2:])
            hour = (pick.astype('datetime64[h]').astype('int64') % 24).astype('float32')
            good = ((duration > 1) & (duration < 180) & (distance > .1) & (distance < 100) &
                    (fare > 2.5) & (fare < 500) & (tip >= 0) & (tip < 200) & (pax > 0) & (pax < 7))
            rows = np.stack([distance[good], duration[good], fare[good],
                             np.minimum(100*tip[good]/fare[good],100), hour[good], pax[good]], axis=1)
            n = min(len(rows), limit-kept)
            X[kept:kept+n] = rows[:n]
            kept += n
            scanned += len(batch)
            if kept == limit:
                break
    if kept < 8:
        raise ValueError('Dataset has fewer than eight usable rows.')
    return X[:kept], scanned


def capture(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='show memory plan; no download or benchmark')
    mode.add_argument('--cleanup', action='store_true', help='delete only this benchmark cache; preserve reports')
    parser.add_argument('--cleanup-after', action='store_true', help='remove benchmark cache after running, even on failure')
    parser.add_argument('--rows', type=int, default=5_000_000, help='maximum retained rows (default 5M; reduced to memory budget)')
    parser.add_argument('--memory-mib', type=int, help='lower the automatic working-memory budget')
    parser.add_argument('--no-compare', action='store_true', help='skip the optional sklearn comparison')
    args = parser.parse_args()
    if args.cleanup:
        cleanup()
        return
    if args.rows < 10_000 or (args.memory_mib is not None and args.memory_mib <= 0):
        parser.error('--rows must be at least 10000; --memory-mib must be positive')
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('The benchmark requires an Apple Silicon Mac.')
    try:
        import psutil
        import certifi
        import numpy as np
        import mlx.core as mx
        import pyarrow.parquet
        from mlx_kmeans import KMeans
    except ImportError as e:
        parser.error(f'{e}. Install dependencies with: python3 -m pip install ".[benchmark]"')
    mem = psutil.virtual_memory()
    gpu = mx.device_info()['max_recommended_working_set_size']
    plan = plan_rows(mem.total, mem.available, gpu, args.rows, args.memory_mib)
    environment = dict(chip=capture('sysctl','-n','machdep.cpu.brand_string'),
                       platform=platform.platform(), python=sys.version, mlx=mx.__version__,
                       power=capture('pmset','-g','batt'), power_settings=capture('pmset','-g','custom'))
    environment['source_sha256'] = {name: file_hash(ROOT/name) for name in
                                    ['air_bench.py','mlx_kmeans/core.py','mlx_kmeans/estimator.py']}
    print(environment['chip'], f'— {mem.total/GIB:.0f} GiB RAM, {mem.available/GIB:.1f} GiB currently available')
    print(environment['power'])
    print(f"Plan: up to {plan['rows']:,} valid taxi rows, 6 features, 8 clusters. "
          f"Estimated working memory {plan['estimated_working_bytes']/MIB:.0f} MiB; "
          f"input {plan['input_bytes']/MIB:.0f} MiB.")
    print('Conservative estimate, not a peak-memory guarantee. Close heavy apps for comparable timings.')
    print('One monthly file, at most 256 MiB on disk. No SIFT, GIST, embeddings, or other months downloaded.')
    if args.plan:
        return
    # Keep MLX's own allocator bounded as well; CPU allocations still rely on the conservative row plan.
    mx.set_memory_limit(plan['memory_budget_bytes'])
    mx.set_cache_limit(0)
    try:
        started = time.perf_counter()
        path, download = fetch()
        download_s = time.perf_counter()-started
        # Available memory can change during a download; shrink again before allocating feature storage.
        mem = psutil.virtual_memory()
        plan = plan_rows(mem.total, mem.available, gpu, plan['rows'], args.memory_mib)
        mx.set_memory_limit(plan['memory_budget_bytes'])
        t = time.perf_counter()
        X, scanned = load_features(path, plan['rows'])
        load_s = time.perf_counter()-t
        t = time.perf_counter()
        mean, std = X.mean(0,dtype=np.float64), X.std(0,dtype=np.float64)+1e-6
        # In-place normalization avoids another full-table output allocation.
        X -= mean
        X /= std
        prepare_s = time.perf_counter()-t
        t = time.perf_counter()
        model = KMeans(n_clusters=8, random_state=0).fit(X)
        fit_s = time.perf_counter()-t
        counts = np.bincount(model.labels_,minlength=8).tolist()
        report = dict(environment=environment, plan=plan, download=download, rows=len(X), scanned_rows=scanned,
                      selection='first valid rows of January 2015, not a random or representative sample',
                      features=FEATURES, clusters=8, seed=0, iterations=model.n_iter_,
                      download_or_cache_check_s=download_s, load_s=load_s, prepare_s=prepare_s,
                      mlx_fit_including_labels_s=fit_s, cluster_counts=counts,
                      centroids_original_units=(model.cluster_centers_*std+mean).tolist())
        print(f'MLX: {len(X):,} rows, {model.n_iter_} iterations, fit + labels {fit_s:.3f} s; '
              f'load + prepare + fit {load_s+prepare_s+fit_s:.3f} s.', flush=True)
        del model
        mx.clear_cache()
        if not args.no_compare:
            try:
                import sklearn
                from sklearn.cluster import KMeans as SK
                # Same data, default greedy sklearn seeding; different stopping rules. This is workflow timing.
                t = time.perf_counter()
                sk = SK(n_clusters=8,n_init=1,max_iter=300,random_state=0).fit(X)
                elapsed = time.perf_counter()-t
                report['sklearn'] = dict(version=sklearn.__version__,fit_s=elapsed,iterations=sk.n_iter_)
                print(f'scikit-learn: fit + labels {elapsed:.3f} s, {sk.n_iter_} iterations. '
                      'Different initialization/stopping; not an equal-work speed ratio.')
            except ImportError:
                report['sklearn'] = dict(skipped='not installed')
            except (ValueError, MemoryError, RuntimeError) as exc:
                report['sklearn'] = dict(error=str(exc))
                print(f'scikit-learn comparison failed: {exc}; preserving the MLX result.')
        report['process_peak_rss_bytes'] = __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss
        report['power_at_end'] = capture('pmset','-g','batt')
        out = ROOT/'benchmarks'/('local-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.json')
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report,indent=2))
        print(f'Report saved: {out}\nShare this JSON to compare measured Macs. One run; caches and thermals are uncontrolled.')
    finally:
        if args.cleanup_after:
            cleanup()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Benchmark stopped: {exc}', file=sys.stderr)
        sys.exit(1)
