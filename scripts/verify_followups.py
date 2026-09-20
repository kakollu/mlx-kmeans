"""Correctness, repeated SIFT training, and a billion-row regression, run serially."""
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench
import bench_ann as ann
import mlx_kmeans as km

OUT = ROOT / 'benchmarks/verification-2026-09-19-ac'

def main():
    with (OUT / 'accuracy.log').open('w') as f:
        r = subprocess.run([sys.executable, '-u', 'tests/accuracy.py'], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError('accuracy suite failed; inspect accuracy.log')
    X, parts = bench.load(bench.SUITE['sift-k1024'], 0)
    queries = bench.read_fvecs(ROOT/'data/sift/sift_query.fvecs')
    gt = ann.read_ivecs(ROOT/'data/sift/sift_groundtruth.ivecs')
    producers = {
        'mlx core (max 50)': lambda seed: ann.ours(parts,len(X),1024,seed),
        'FAISS sampled (25)': lambda seed: ann.faiss_kmeans(X,1024,seed,True),
        'FAISS full (25)': lambda seed: ann.faiss_kmeans(X,1024,seed,False),
        'sklearn full (25)': lambda seed: ann.sklearn_kmeans(X,1024,seed),
        'sklearn minibatch': lambda seed: ann.sklearn_minibatch(X,1024,seed),
    }
    with (OUT/'training.jsonl').open('w') as f:
        for name, fn in producers.items():
            for seed in range(3):
                load = bench.other_cpu_percent()
                t = time.perf_counter()
                C, iters = fn(seed)
                elapsed = time.perf_counter()-t
                inertia = ann.inertia_full(parts,C)
                recall, build = ann.ivf_recall(X,queries,gt,C,1024)
                rec=dict(method=name,seed=seed,train_s=elapsed,iters=iters,inertia=inertia,
                         recall=recall,other_cpu=load,index_build_s=build)
                f.write(json.dumps(rec)+'\n'); f.flush()
                print(name,seed,round(elapsed,3),inertia,flush=True)
    del X, parts, queries, gt
    km._mx().clear_cache()
    with (OUT/'billion.log').open('w') as f:
        r = subprocess.run([sys.executable,'-u','-m','mlx_kmeans','--rows','1000000000','--dims','8','--k','16'],
                           cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError('billion-row run failed')
    # Real business-shaped workload: one month of taxi transactions, full public API.
    from demos.taxi_segments import load
    t = time.perf_counter()
    X, zones, revenue, card, kept, total = load(1)
    load_s = time.perf_counter()-t
    t = time.perf_counter()
    mean, std = X.mean(0, dtype=np.float64), X.std(0, dtype=np.float64)+1e-6
    Z = np.ascontiguousarray((X-mean)/std, dtype=np.float32)
    prepare_s = time.perf_counter()-t
    t = time.perf_counter()
    m = km.KMeans(n_clusters=8, random_state=0).fit(Z)
    fit_s = time.perf_counter()-t
    rec = dict(rows=kept,source_rows=total,dims=6,k=8,input_bytes=Z.nbytes,
               load_s=load_s,prepare_s=prepare_s,fit_including_labels_s=fit_s,
               iterations=m.n_iter_,cluster_counts=np.bincount(m.labels_,minlength=8).tolist())
    (OUT/'taxi-business.json').write_text(json.dumps(rec,indent=2))
    print('Taxi business workload',rec,flush=True)

if __name__ == '__main__':
    main()
