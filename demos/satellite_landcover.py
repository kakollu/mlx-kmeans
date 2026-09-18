#!/usr/bin/env python3
"""Demo: land cover from one Sentinel-2 scene (~120M pixels) on a laptop.

Every 10 m pixel of a cloud-free scene over New York, clustered on its four 10 m bands (blue, green, red,
near-infrared) plus two indices that make the classes interpretable:
  NDVI = (NIR - red) / (NIR + red)   -> vegetation
  NDWI = (green - NIR) / (green + NIR) -> water

Unsupervised land cover: no labels, no training set, just what the pixels look like. Writes a classified PNG and
demos/satellite_results.json.

  python3 demos/satellite_landcover.py [--k 8] [--compare]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_kmeans import KMeans  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sentinel"
BANDS = {"blue": "B02.tif", "green": "B03.tif", "red": "B04.tif", "nir": "B08.tif"}


def load():
    """Stack the four bands, drop no-data pixels, and add NDVI/NDWI."""
    t = time.perf_counter()
    arrays, profile = {}, None
    for name, fn in BANDS.items():
        with rasterio.open(DATA / fn) as src:
            arrays[name] = src.read(1).astype(np.float32)
            profile = profile or src.profile
    shape = arrays["blue"].shape
    stack = np.stack([arrays[b].ravel() for b in BANDS], axis=1)
    valid = (stack > 0).all(1)
    blue, green, red, nir = (stack[:, i] for i in range(4))
    ndvi = np.where(nir + red > 0, (nir - red) / (nir + red + 1e-6), 0).astype(np.float32)
    ndwi = np.where(green + nir > 0, (green - nir) / (green + nir + 1e-6), 0).astype(np.float32)
    feats = np.column_stack([stack / 10000.0, ndvi * 2, ndwi * 2]).astype(np.float32)  # indices weighted x2
    print(f"scene {shape[0]}x{shape[1]} = {stack.shape[0]:,} pixels, {int(valid.sum()):,} valid, "
          f"{feats.nbytes/1e9:.1f} GB of features, read in {time.perf_counter()-t:.1f}s")
    return feats, valid, shape, time.perf_counter() - t


def colour_for(center):
    """A plausible colour for a class from its own reflectance, so the map reads like the ground."""
    b, g, r = center[0], center[1], center[2]
    scale = 3.2 / max(b + g + r, 1e-6)
    return [int(np.clip(v * scale * 255, 0, 255)) for v in (r, g, b)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--compare", action="store_true", help="also time scikit-learn on a 10M-pixel slice")
    a = p.parse_args()

    feats, valid, shape, read_s = load()
    X = np.ascontiguousarray(feats[valid])
    t = time.perf_counter()
    model = KMeans(n_clusters=a.k, random_state=0).fit(X)
    fit_s = time.perf_counter() - t
    labels = model.labels_
    print(f"clustered {len(X):,} pixels x {X.shape[1]} into {a.k} classes in {fit_s:.1f}s ({model.n_iter_} iterations)")

    counts = np.bincount(labels, minlength=a.k)
    centers = model.cluster_centers_
    classes = []
    for i in range(a.k):
        ndvi, ndwi = centers[i][4] / 2, centers[i][5] / 2
        kind = ("water" if ndwi > 0.15 else "dense vegetation" if ndvi > 0.55 else "vegetation" if ndvi > 0.3
                else "bare / low vegetation" if ndvi > 0.15 else "built-up / paved")
        classes.append(dict(id=int(i), pixels=int(counts[i]), share=float(counts[i] / len(labels)),
                            km2=float(counts[i] * 100 / 1e6), ndvi=float(ndvi), ndwi=float(ndwi),
                            brightness=float(centers[i][:3].mean()), kind=kind, colour=colour_for(centers[i])))
    classes.sort(key=lambda c: -c["pixels"])

    # classified image, downsampled so the page stays small
    full = np.full(valid.shape, 255, dtype=np.uint8)
    full[valid] = labels.astype(np.uint8)
    img = full.reshape(shape)[::6, ::6]
    palette = {c["id"]: c["colour"] for c in classes}
    rgb = np.zeros((*img.shape, 3), dtype=np.uint8)
    for cid, col in palette.items():
        rgb[img == cid] = col
    out_png = ROOT / "demos" / "satellite_classes.png"
    try:
        from PIL import Image
        Image.fromarray(rgb).save(out_png, optimize=True)
    except ImportError:
        import struct, zlib
        raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(rgb.shape[0]))
        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", rgb.shape[1], rgb.shape[0], 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
        out_png.write_bytes(png)
    print(f"wrote {out_png.name}: {img.shape[1]}x{img.shape[0]} px, {out_png.stat().st_size/1e6:.1f} MB")

    res = dict(pixels=int(len(X)), dims=int(X.shape[1]), k=a.k, read_s=read_s, fit_s=fit_s,
               iterations=int(model.n_iter_), inertia=float(model.inertia_), scene="S2A_18TWL_20240903_0_L2A",
               shape=[int(shape[0]), int(shape[1])], classes=classes, image=out_png.name,
               image_shape=[int(img.shape[0]), int(img.shape[1])],
               machine=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip())
    if a.compare:
        from sklearn.cluster import KMeans as SK
        sub = X[:10_000_000]
        t = time.perf_counter()
        SK(n_clusters=a.k, n_init=1, max_iter=model.n_iter_, random_state=0).fit(sub)
        res["sklearn_rows"], res["sklearn_s"] = len(sub), time.perf_counter() - t
        t = time.perf_counter()
        KMeans(n_clusters=a.k, max_iter=model.n_iter_, tol=0.0, random_state=0).fit(sub)
        res["ours_same_rows_s"] = time.perf_counter() - t
        print(f"scikit-learn on {len(sub):,} pixels: {res['sklearn_s']:.1f}s; ours: {res['ours_same_rows_s']:.2f}s")
    (ROOT / "demos" / "satellite_results.json").write_text(json.dumps(res, indent=1))
    print(f"\n{'class':>5} {'share':>7} {'km2':>8} {'NDVI':>6} {'NDWI':>6}  reads as")
    for c in classes:
        print(f"{c['id']:>5} {c['share']*100:6.1f}% {c['km2']:8.0f} {c['ndvi']:6.2f} {c['ndwi']:6.2f}  {c['kind']}")


if __name__ == "__main__":
    main()
