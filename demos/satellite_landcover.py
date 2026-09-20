#!/usr/bin/env python3
"""Demo: land cover from one Sentinel-2 scene (~120M pixels) on a laptop.

Every 10 m pixel of a cloud-free scene over New York, clustered on its four 10 m bands (blue, green, red,
near-infrared) plus two indices that make the classes interpretable:
  NDVI = (NIR - red) / (NIR + red)   -> vegetation
  NDWI = (green - NIR) / (green + NIR) -> water

Unsupervised land cover: no labels, no training set, just what the pixels look like. Writes a classified PNG and
demos/satellite_results.json.

  python3 demos/satellite_landcover.py [--k 8] [--compare] [--verbose]

Get the data first (~700 MB, no account needed):  python3 scripts/get_data.py sentinel
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
    print(f"[1/4] reading {len(BANDS)} bands from {DATA.relative_to(ROOT)}/")
    arrays, profile = {}, None
    for name, fn in BANDS.items():
        bt = time.perf_counter()
        with rasterio.open(DATA / fn) as src:
            arrays[name] = src.read(1).astype(np.float32)
            profile = profile or src.profile
        mb = (DATA / fn).stat().st_size / 1e6
        print(f"      {fn:8} {name:6} {arrays[name].shape[0]}x{arrays[name].shape[1]}  "
              f"{mb:6.0f} MB on disk  {time.perf_counter()-bt:5.2f}s")
    shape = arrays["blue"].shape
    crs, res = profile.get("crs"), abs(profile["transform"][0])
    print(f"      scene {shape[0]}x{shape[1]} = {shape[0]*shape[1]:,} pixels at {res:.0f} m, {crs}")

    print("[2/4] building features")
    stack = np.stack([arrays[b].ravel() for b in BANDS], axis=1)
    valid = (stack > 0).all(1)
    dropped = len(valid) - int(valid.sum())
    blue, green, red, nir = (stack[:, i] for i in range(4))
    ndvi = np.where(nir + red > 0, (nir - red) / (nir + red + 1e-6), 0).astype(np.float32)
    ndwi = np.where(green + nir > 0, (green - nir) / (green + nir + 1e-6), 0).astype(np.float32)
    feats = np.column_stack([stack / 10000.0, ndvi * 2, ndwi * 2]).astype(np.float32)  # indices weighted x2
    print(f"      6 features per pixel: blue, green, red, nir (reflectance), NDVI, NDWI (x2 weight)")
    print(f"      NDVI = (nir-red)/(nir+red) -> vegetation;  NDWI = (green-nir)/(green+nir) -> water")
    print(f"      dropped {dropped:,} no-data pixels ({dropped/len(valid)*100:.1f}%, the black corner off the "
          f"satellite's swath)")
    read_s = time.perf_counter() - t
    print(f"      {int(valid.sum()):,} pixels x 6 = {feats[valid].nbytes/1e9:.1f} GB to cluster "
          f"(read + prepare: {read_s:.1f}s)")
    return feats, valid, shape, read_s


# A land-cover map is read, not admired: colour carries the class, so the ramps are the conventional ones
# (water blue, vegetation green, bare tan, built-up red-grey) and each kind darkens with the index that defines
# it. Colouring each class by its own mean reflectance instead - the obvious thing - washes the map out to pastel,
# because averaging a cluster of real pixels lands near mid-grey whatever the cover type is.
RAMPS = {"water": [(24, 68, 130), (72, 132, 190), (140, 186, 224)],
         "dense vegetation": [(16, 74, 34), (30, 110, 48), (58, 145, 70)],
         "vegetation": [(104, 170, 74), (140, 192, 96)],
         "bare / low vegetation": [(186, 170, 110), (214, 200, 148)],
         "built-up / paved": [(158, 66, 58), (196, 118, 104)]}


def assign_colours(classes):
    """Darkest step of a kind's ramp to its most extreme member, so the map ranks within a kind as well as across."""
    for kind, ramp in RAMPS.items():
        members = [c for c in classes if c["kind"] == kind]
        key = (lambda c: -c["ndwi"]) if kind == "water" else (lambda c: -c["ndvi"])
        for rank, c in enumerate(sorted(members, key=key)):
            c["colour"] = list(ramp[min(rank, len(ramp) - 1)])
    return classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--compare", action="store_true", help="also time scikit-learn on a 10M-pixel slice")
    p.add_argument("--verbose", action="store_true", help="print inertia after every Lloyd iteration")
    a = p.parse_args()

    feats, valid, shape, read_s = load()
    X = np.ascontiguousarray(feats[valid])
    print(f"[3/4] clustering on the GPU: k={a.k}, seed 0, greedy k-means++ seeding, every pixel every iteration")
    t = time.perf_counter()
    model = KMeans(n_clusters=a.k, random_state=0, verbose=a.verbose).fit(X)
    fit_s = time.perf_counter() - t
    labels = model.labels_
    print(f"      {len(X):,} pixels x {X.shape[1]} -> {a.k} classes in {fit_s:.2f}s, {model.n_iter_} iterations "
          f"({fit_s/model.n_iter_*1e3:.0f} ms each, "
          f"{len(X)*a.k*model.n_iter_/fit_s/1e9:.0f}B pixel-to-center distances/s)")
    print(f"      inertia {model.inertia_:.6e}   labels for every pixel included in that time")

    counts = np.bincount(labels, minlength=a.k)
    centers = model.cluster_centers_
    classes = []
    for i in range(a.k):
        ndvi, ndwi = centers[i][4] / 2, centers[i][5] / 2
        kind = ("water" if ndwi > 0.15 else "dense vegetation" if ndvi > 0.55 else "vegetation" if ndvi > 0.3
                else "bare / low vegetation" if ndvi > 0.15 else "built-up / paved")
        classes.append(dict(id=int(i), pixels=int(counts[i]), share=float(counts[i] / len(labels)),
                            km2=float(counts[i] * 100 / 1e6), ndvi=float(ndvi), ndwi=float(ndwi),
                            brightness=float(centers[i][:3].mean()), kind=kind))
    assign_colours(classes)
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
    print(f"[4/4] wrote demos/{out_png.name}: {img.shape[1]}x{img.shape[0]} px, "
          f"{out_png.stat().st_size/1e6:.1f} MB (every 6th pixel, so the page stays small)")

    res = dict(pixels=int(len(X)), dims=int(X.shape[1]), k=a.k, read_s=read_s, fit_s=fit_s,
               iterations=int(model.n_iter_), inertia=float(model.inertia_), scene="S2A_18TWL_20240903_0_L2A",
               shape=[int(shape[0]), int(shape[1])], classes=classes, image=out_png.name,
               image_shape=[int(img.shape[0]), int(img.shape[1])],
               machine=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
               power=subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0])
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
