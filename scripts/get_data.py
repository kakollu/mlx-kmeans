#!/usr/bin/env python3
"""Download the public datasets the benchmarks and demos use, so the numbers can be checked independently.

    python3 scripts/get_data.py sift          # 1M x 128 vectors + ground truth (~550 MB) - benchmark suite
    python3 scripts/get_data.py gist          # 1M x 960 vectors (~5.4 GB)                 - benchmark suite
    python3 scripts/get_data.py taxi          # 8 months of NYC yellow-cab trips (~1.3 GB) - demo
    python3 scripts/get_data.py sentinel      # one Sentinel-2 scene, 4 bands (~700 MB)    - demo
    python3 scripts/get_data.py dbpedia       # 1M OpenAI embeddings (~8.6 GB)             - demo
    python3 scripts/get_data.py sift gist     # several at once;  --list shows sizes

Everything lands in data/ (gitignored). Nothing here needs an account or an API key.
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEXMEX = "ftp://ftp.irisa.fr/local/texmex/corpus"
TLC = "https://d37ci6vzurychx.cloudfront.net"
SENTINEL = ("https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/18/T/WL/2024/9/"
            "S2A_18TWL_20240903_0_L2A")          # cloud-free New York scene, 2024-09-03
HF = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main"

SETS = {
    "sift": ("SIFT1M: 1M x 128 vectors + 10k queries + ground truth", "~550 MB"),
    "gist": ("GIST1M: 1M x 960 vectors + queries + ground truth", "~5.4 GB"),
    "taxi": ("NYC yellow-cab trips, Jan-Aug 2015 (~99.8M) + taxi zones", "~1.3 GB"),
    "sentinel": ("Sentinel-2 L2A scene over New York, bands B02/B03/B04/B08", "~700 MB"),
    "dbpedia": ("1M DBpedia entity embeddings (OpenAI, 1536 dims)", "~8.6 GB"),
}


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"   have {dest.relative_to(ROOT)} ({dest.stat().st_size/1e6:.0f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"   {url.split('/')[-1]} -> {dest.relative_to(ROOT)}", flush=True)
    subprocess.run(["curl", "-sSL", "--fail", "-o", str(dest), url], check=True)


def sift_or_gist(name):
    tar = DATA / f"{name}.tar.gz"
    fetch(f"{TEXMEX}/{name}.tar.gz", tar)
    if not (DATA / name / f"{name}_base.fvecs").exists():
        subprocess.run(["tar", "xzf", str(tar), "-C", str(DATA)], check=True)
    print(f"   ready: data/{name}/{name}_base.fvecs")


def taxi():
    for m in range(1, 9):
        fetch(f"{TLC}/trip-data/yellow_tripdata_2015-{m:02d}.parquet", DATA / "taxi" / f"yellow_2015-{m:02d}.parquet")
    fetch(f"{TLC}/misc/taxi_zone_lookup.csv", DATA / "taxi" / "taxi_zone_lookup.csv")
    fetch(f"{TLC}/misc/taxi_zones.zip", DATA / "taxi" / "taxi_zones.zip")
    if not (DATA / "taxi" / "zones").exists():
        subprocess.run(["unzip", "-oq", str(DATA / "taxi" / "taxi_zones.zip"), "-d", str(DATA / "taxi" / "zones")], check=True)


def sentinel():
    for band in ["B02", "B03", "B04", "B08"]:
        fetch(f"{SENTINEL}/{band}.tif", DATA / "sentinel" / f"{band}.tif")


def dbpedia():
    with urllib.request.urlopen("https://huggingface.co/api/datasets/KShivendu/dbpedia-entities-openai-1M") as r:
        import json
        files = [s["rfilename"] for s in json.load(r)["siblings"] if s["rfilename"].endswith(".parquet")]
    for i, f in enumerate(sorted(files)):
        fetch(f"{HF}/{f}", DATA / "dbpedia" / f"shard{i:02d}.parquet")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sets", nargs="*", default=[], choices=list(SETS) + [[]], help="datasets to download")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list or not a.sets:
        print("dataset   size      what")
        for name, (what, size) in SETS.items():
            print(f"{name:9} {size:9} {what}")
        return
    for name in a.sets:
        print(f"\n{name}: {SETS[name][0]} ({SETS[name][1]})")
        {"sift": lambda: sift_or_gist("sift"), "gist": lambda: sift_or_gist("gist"),
         "taxi": taxi, "sentinel": sentinel, "dbpedia": dbpedia}[name]()
    print("\ndone. Verify the benchmark numbers with:  .venv/bin/python bench.py --suite")


if __name__ == "__main__":
    sys.exit(main())
