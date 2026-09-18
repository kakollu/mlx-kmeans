#!/usr/bin/env python3
"""Demo: segment ~100M NYC taxi trips on a laptop.

Every yellow-cab trip in New York for 8 months of 2015 (~100M rows, public TLC data), clustered on what a business
cares about - fare, distance, duration, time of day, tip rate, party size - to find the trip archetypes that make up
demand, and where each one starts.

Writes demos/taxi_results.json (segments, timings, per-zone mix) for the presentation page.

  python3 demos/taxi_segments.py [--k 8] [--months 8] [--compare]

--compare also runs scikit-learn on the same features, which takes minutes rather than seconds.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys_path = str(Path(__file__).resolve().parents[1])
__import__("sys").path.insert(0, sys_path)
import mlx_kmeans as km
from mlx_kmeans import KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "taxi"
COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount", "tip_amount",
        "passenger_count", "PULocationID", "payment_type"]   # payment_type is profiled, not clustered
FEATURES = ["distance (mi)", "duration (min)", "fare ($)", "tip rate (%)", "hour of day", "passengers"]


def load(months):
    """Read the parquet files into a float32 feature matrix, keeping only plausible trips."""
    feats, zones, revenue, card, kept, total = [], [], [], [], 0, 0
    for path in sorted(DATA.glob("yellow_2015-*.parquet"))[:months]:
        t = pq.read_table(path, columns=COLS)
        pick = t["tpep_pickup_datetime"].to_numpy().astype("datetime64[s]")
        drop = t["tpep_dropoff_datetime"].to_numpy().astype("datetime64[s]")
        dur = (drop - pick).astype("float32") / 60.0
        dist = t["trip_distance"].to_numpy().astype("float32")
        fare = t["fare_amount"].to_numpy().astype("float32")
        tip = t["tip_amount"].to_numpy().astype("float32")
        pax = t["passenger_count"].to_numpy().astype("float32")
        hour = (pick.astype("datetime64[h]").astype("int64") % 24).astype("float32")
        zone = t["PULocationID"].to_numpy().astype("int32")
        total += len(dist)
        ok = ((dur > 1) & (dur < 180) & (dist > 0.1) & (dist < 100) & (fare > 2.5) & (fare < 500) &
              (tip >= 0) & (tip < 200) & (pax > 0) & (pax < 7))
        tip_rate = np.where(fare[ok] > 0, 100 * tip[ok] / fare[ok], 0).astype("float32")
        feats.append(np.stack([dist[ok], dur[ok], fare[ok], np.minimum(tip_rate, 100), hour[ok], pax[ok]], axis=1))
        zones.append(zone[ok])
        revenue.append((fare[ok] + tip[ok]).astype("float32"))
        card.append((t["payment_type"].to_numpy().astype("int8")[ok] == 1))   # 1 = credit card, 2 = cash
        kept += int(ok.sum())
        print(f"   {path.name}: {len(dist):,} trips, kept {int(ok.sum()):,}", flush=True)
    return np.concatenate(feats), np.concatenate(zones), np.concatenate(revenue), np.concatenate(card), kept, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--months", type=int, default=8)
    p.add_argument("--compare", action="store_true", help="also run scikit-learn on the same features")
    a = p.parse_args()

    t0 = time.perf_counter()
    X, zones, revenue, card, kept, total = load(a.months)
    load_s = time.perf_counter() - t0
    print(f"\n{kept:,} trips kept of {total:,} ({X.nbytes/1e9:.1f} GB of features) read in {load_s:.1f}s")

    # Standardise so no single feature dominates the distance, then cluster.
    mean, std = X.mean(0), X.std(0) + 1e-6
    Z = np.ascontiguousarray((X - mean) / std, dtype=np.float32)
    t = time.perf_counter()
    model = KMeans(n_clusters=a.k, random_state=0).fit(Z)
    fit_s = time.perf_counter() - t
    labels = model.labels_
    print(f"clustered {len(Z):,} x {Z.shape[1]} into {a.k} segments in {fit_s:.1f}s ({model.n_iter_} iterations)")

    counts = np.bincount(labels, minlength=a.k)
    centers = model.cluster_centers_ * std + mean                       # back to real units
    seg_revenue = np.bincount(labels, weights=revenue.astype(np.float64), minlength=a.k)
    seg_minutes = np.bincount(labels, weights=X[:, 1].astype(np.float64), minlength=a.k)
    seg_miles = np.bincount(labels, weights=X[:, 0].astype(np.float64), minlength=a.k)
    total_revenue = float(seg_revenue.sum())
    hours = X[:, 4].astype(np.int32)
    segments = []
    for i in range(a.k):
        m = labels == i
        by_hour = np.bincount(hours[m], minlength=24)
        segments.append(dict(
            id=int(i), trips=int(counts[i]), share=float(counts[i] / len(labels)),
            revenue=float(seg_revenue[i]), revenue_share=float(seg_revenue[i] / total_revenue),
            revenue_per_trip=float(seg_revenue[i] / max(counts[i], 1)),
            fare_per_mile=float(seg_revenue[i] / max(seg_miles[i], 1e-9)),
            revenue_per_minute=float(seg_revenue[i] / max(seg_minutes[i], 1e-9)),
            by_hour=(by_hour / max(by_hour.sum(), 1)).round(4).tolist(),
            card_share=float(card[m].mean()),
            **{f: float(v) for f, v in zip(FEATURES, centers[i])}))
    order = np.argsort([-s["revenue"] for s in segments])               # rank by revenue, not trip count
    segments = [segments[i] for i in order]

    # Where each segment starts: share of each pickup zone's trips by segment (top zones only, for the map).
    remap = {int(s["id"]): new for new, s in enumerate(segments)}
    lab2 = np.array([remap[int(l)] for l in range(a.k)])[labels]
    zone_mix = {}
    for z in np.unique(zones):
        m = zones == z
        if m.sum() >= 1000:
            zone_mix[int(z)] = dict(trips=int(m.sum()), mix=np.bincount(lab2[m], minlength=a.k).tolist())
    names = {}
    lookup = DATA / "taxi_zone_lookup.csv"
    if lookup.exists():
        for row in csv.DictReader(lookup.open()):
            names[int(row["LocationID"])] = f"{row['Zone']} ({row['Borough']})"
    for new, s in enumerate(segments):
        old_id = s["id"]
        s["id"] = new
        m = labels == old_id
        z, c = np.unique(zones[m], return_counts=True)
        top = np.argsort(-c)[:5]
        s["top_zones"] = [dict(zone=int(z[j]), name=names.get(int(z[j]), f"zone {int(z[j])}"),
                               trips=int(c[j]), share=float(c[j] / m.sum())) for j in top]

    out = dict(rows=int(len(Z)), dims=int(Z.shape[1]), k=a.k, months=a.months, trips_total=int(total),
               total_revenue=total_revenue,
               load_s=load_s, fit_s=fit_s, iterations=int(model.n_iter_), inertia=float(model.inertia_),
               features=FEATURES, segments=segments, zone_mix=zone_mix,
               machine=__import__("subprocess").run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                                    capture_output=True, text=True).stdout.strip())
    if a.compare:
        from sklearn.cluster import KMeans as SK
        sub = Z[:10_000_000]                                            # scikit-learn on all 100M would take ~an hour
        t = time.perf_counter()
        sk = SK(n_clusters=a.k, n_init=1, max_iter=model.n_iter_, random_state=0).fit(sub)
        out["sklearn_rows"], out["sklearn_s"], out["sklearn_inertia"] = len(sub), time.perf_counter() - t, float(sk.inertia_)
        t = time.perf_counter()
        ours_sub = KMeans(n_clusters=a.k, max_iter=model.n_iter_, tol=0.0, random_state=0).fit(sub)
        out["ours_same_rows_s"], out["ours_same_rows_inertia"] = time.perf_counter() - t, float(ours_sub.inertia_)
        print(f"scikit-learn on {len(sub):,} rows: {out['sklearn_s']:.1f}s (inertia {out['sklearn_inertia']:.4e}); "
              f"ours on the same rows: {out['ours_same_rows_s']:.1f}s (inertia {out['ours_same_rows_inertia']:.4e})")

    (ROOT / "demos").mkdir(exist_ok=True)
    (ROOT / "demos" / "taxi_results.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote demos/taxi_results.json")
    print(f"\n{'seg':>3} {'trips':>7} {'revenue':>8} {'$/trip':>7} {'$/mile':>7} {'$/min':>6} {'tip%':>5} {'card%':>6} "
          f"{'miles':>6} {'mins':>6} {'peak':>5}  top pickup zone")
    for s in segments:
        peak = int(np.argmax(s["by_hour"]))
        print(f"{s['id']:>3} {s['share']*100:6.1f}% {s['revenue_share']*100:7.1f}% {s['revenue_per_trip']:7.2f} "
              f"{s['fare_per_mile']:7.2f} {s['revenue_per_minute']:6.2f} {s['tip rate (%)']:5.1f} {s['card_share']*100:6.1f} "
              f"{s['distance (mi)']:6.1f} {s['duration (min)']:6.1f} {peak:02d}:00  {s['top_zones'][0]['name']}")


if __name__ == "__main__":
    main()
