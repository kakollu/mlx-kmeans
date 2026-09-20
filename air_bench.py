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


def inertia_for_labels(X, centers, labels):
    """Common float64 squared-distance metric, outside the timed fit and without a full-table copy."""
    import numpy as np
    total = 0.0
    for start in range(0, len(X), 65_536):
        block = X[start:start+65_536].astype(np.float64)
        block -= np.asarray(centers,dtype=np.float64)[labels[start:start+65_536]]
        total += float(np.sum(block*block,dtype=np.float64))
    return total


def timing_verdict(mlx_s, sklearn_s):
    if mlx_s <= 0 or sklearn_s <= 0:
        raise ValueError('Timings must be positive.')
    if abs(mlx_s-sklearn_s) <= .01*min(mlx_s,sklearn_s):
        return 'Result: effectively tied in this run (within 1%).'
    if mlx_s > sklearn_s:
        return f'Result: scikit-learn was faster; MLX took {(mlx_s/sklearn_s-1)*100:.0f}% longer.'
    return f'Result: MLX was faster; scikit-learn took {(sklearn_s/mlx_s-1)*100:.0f}% longer.'


def print_results(report):
    print(f"\nClustering {report['rows']:,} taxi rows, 6 features, 8 groups (lower time is better)")
    print(f"{'Method':<18} {'Fit + labels':>14} {'Iterations':>12}")
    print(f"{'MLX':<18} {report['mlx_fit_including_labels_s']:>12.3f} s {report['iterations']:>12}")
    sk = report.get('sklearn',{})
    if 'fit_s' in sk:
        print(f"{'scikit-learn':<18} {sk['fit_s']:>12.3f} s {sk['iterations']:>12}")
        print(timing_verdict(report['mlx_fit_including_labels_s'],sk['fit_s']))
        ours, theirs = report['mlx_inertia_float64'], sk['inertia_float64']
        if max(ours,theirs) == 0:
            print('Clustering error: both zero.')
        elif theirs == 0:
            print('Clustering error: scikit-learn is zero; MLX is higher.')
        else:
            difference = (ours/theirs-1)*100
            direction = 'higher' if difference >= 0 else 'lower'
            print(f'Clustering error: MLX is {abs(difference):.2f}% {direction} (lower error is better).')
        print('Both used the SAME data. Times include their own initialization, training and labels.')
        print('Iteration counts can differ because each method finds its own stopping point.')
    else:
        print('scikit-learn comparison:',sk.get('error',sk.get('skipped','disabled')))
    print(f"Shared file loading + preparation: {report['load_s']+report['prepare_s']:.3f} s (excluded above).")
    print('One run on this workload; results can vary. This does not establish a winner for all datasets.')


def controlled_comparison(X):
    """Optional one-update + final-label comparison, shared centers, warmup, three alternating repeats."""
    import numpy as np
    from sklearn.cluster import KMeans as SK
    from mlx_kmeans import core
    mx = core._mx()
    parts = [mx.array(X)]
    mx.eval(parts)
    centers = core.kmeans_pp_init(parts,len(X),8,np.random.default_rng(0))

    def mlx_run():
        C, _, _ = core.lloyd_step(parts,centers)
        _, _, _, labels = core.assign_mlx(parts,C,return_labels=True)
        return C, labels

    def sklearn_run():
        m = SK(n_clusters=8,init=centers.copy(),n_init=1,max_iter=1,tol=0,algorithm='lloyd').fit(X)
        return m.cluster_centers_, m.labels_

    functions = {'mlx':mlx_run,'sklearn':sklearn_run}
    times = {name:[] for name in functions}
    for fn in functions.values():
        fn()
    final = {}
    for repeat in range(3):
        for name in (['mlx','sklearn'] if repeat%2 == 0 else ['sklearn','mlx']):
            t = time.perf_counter()
            final[name] = functions[name]()
            times[name].append(time.perf_counter()-t)
    scores = {name:inertia_for_labels(X,*value) for name,value in final.items()}
    record = dict(description='same data and centers; one Lloyd update plus final labels; GPU input conversion '
                  'and initialization excluded; sklearn fit includes its internal preparation',
                  raw_seconds=times,median_seconds={name:float(np.median(t)) for name,t in times.items()},
                  inertia_float64=scores)
    print('\nControlled check: same starting centers, ONE update + labels (median of 3 warmed runs).')
    print(f"MLX {record['median_seconds']['mlx']*1000:.3f} ms; "
          f"scikit-learn {record['median_seconds']['sklearn']*1000:.3f} ms.")
    delta = abs(scores['mlx']-scores['sklearn'])/max(scores.values()) if max(scores.values()) else 0
    print(f'Final clustering error differs by {delta*100:.4f}%.')
    if delta <= 1e-4:
        print(timing_verdict(record['median_seconds']['mlx'],record['median_seconds']['sklearn']))
    else:
        print('Quality differs by more than 0.01%; do not interpret these times as a same-result speedup.')
    print('This isolates one update, not the complete training time reported above.')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan', action='store_true', help='show memory plan; no download or benchmark')
    mode.add_argument('--cleanup', action='store_true', help='delete only this benchmark cache; preserve reports')
    parser.add_argument('--cleanup-after', action='store_true', help='remove benchmark cache after running, even on failure')
    parser.add_argument('--rows', type=int, default=5_000_000, help='maximum retained rows (default 5M; reduced to memory budget)')
    parser.add_argument('--memory-mib', type=int, help='lower the automatic working-memory budget')
    parser.add_argument('--no-compare', action='store_true', help='skip the optional sklearn comparison')
    parser.add_argument('--controlled', action='store_true', help='also compare one update using identical starting centers')
    parser.add_argument('--details', action='store_true', help='print power and memory details (always saved in JSON)')
    args = parser.parse_args()
    if args.cleanup:
        cleanup()
        return
    if args.rows < 10_000 or (args.memory_mib is not None and args.memory_mib <= 0):
        parser.error('--rows must be at least 10000; --memory-mib must be positive')
    if args.controlled and args.no_compare:
        parser.error('--controlled requires the scikit-learn comparison; omit --no-compare')
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
    print(f"{environment['chip']} — testing up to {plan['rows']:,} taxi rows, 6 features, 8 groups.",flush=True)
    if args.details or args.plan:
        print(environment['power'])
        print(f"RAM {mem.total/GIB:.0f} GiB; available {mem.available/GIB:.1f} GiB. "
              f"Estimated working memory {plan['estimated_working_bytes']/MIB:.0f} MiB; "
              f"input {plan['input_bytes']/MIB:.0f} MiB.")
        print('Memory is a conservative estimate, not a guarantee. One monthly download, at most 256 MiB.')
    if args.plan:
        print('Preview only: no benchmark has run. Run it with: python air_bench.py --cleanup-after')
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
        print('Running MLX clustering...',flush=True)
        t = time.perf_counter()
        model = KMeans(n_clusters=8, random_state=0).fit(X)
        fit_s = time.perf_counter()-t
        counts = np.bincount(model.labels_,minlength=8).tolist()
        report = dict(environment=environment, plan=plan, download=download, rows=len(X), scanned_rows=scanned,
                      selection='first valid rows of January 2015, not a random or representative sample',
                      features=FEATURES, clusters=8, seed=0, iterations=model.n_iter_,
                      download_or_cache_check_s=download_s, load_s=load_s, prepare_s=prepare_s,
                      mlx_fit_including_labels_s=fit_s, cluster_counts=counts,
                      mlx_inertia_float64=inertia_for_labels(X,model.cluster_centers_,model.labels_),
                      centroids_original_units=(model.cluster_centers_*std+mean).tolist())
        del model
        mx.clear_cache()
        if not args.no_compare:
            try:
                import sklearn
                from sklearn.cluster import KMeans as SK
                # Same data, default greedy sklearn seeding; different stopping rules. This is workflow timing.
                print('Running scikit-learn clustering...',flush=True)
                t = time.perf_counter()
                sk = SK(n_clusters=8,n_init=1,max_iter=300,random_state=0).fit(X)
                elapsed = time.perf_counter()-t
                report['sklearn'] = dict(version=sklearn.__version__,fit_s=elapsed,iterations=sk.n_iter_,
                                         inertia_float64=inertia_for_labels(X,sk.cluster_centers_,sk.labels_))
                del sk
            except ImportError:
                report['sklearn'] = dict(skipped='not installed')
            except (ValueError, MemoryError, RuntimeError) as exc:
                report['sklearn'] = dict(error=str(exc))
                print(f'scikit-learn comparison failed: {exc}; preserving the MLX result.')
        print_results(report)
        if args.controlled:
            try:
                report['controlled'] = controlled_comparison(X)
            except (ImportError, ValueError, MemoryError, RuntimeError) as exc:
                report['controlled'] = dict(error=str(exc))
                print(f'Controlled check unavailable: {exc}. The complete-fit result above is preserved.')
        report['process_peak_rss_bytes'] = __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss
        report['power_at_end'] = capture('pmset','-g','batt')
        out = ROOT/'benchmarks'/('local-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.json')
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(report,indent=2))
        print(f'\nFull report (including power/memory details): {out}')
    finally:
        if args.cleanup_after:
            cleanup()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Benchmark stopped: {exc}', file=sys.stderr)
        sys.exit(1)
