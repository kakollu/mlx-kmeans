"""Dated benchmark rerun; preserves historical benchmark records."""
import contextlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bench

OUT = ROOT / 'benchmarks' / 'verification-2026-09-19-ac'
OUT.mkdir(exist_ok=True)
bench.RESULTS = OUT / 'suite.jsonl'

def main():
    meta = dict(machine=bench.machine(), python=sys.version,
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                power=subprocess.check_output(['pmset', '-g', 'batt'], text=True),
                method='Original suite: warmup, median of three, five iterations; sklearn divided by six. No historical best selection.')
    (OUT / 'environment.json').write_text(json.dumps(meta, indent=2))
    with (OUT / 'suite.log').open('w') as log:
        with contextlib.redirect_stdout(log):
            for name, cfg in bench.SUITE.items():
                bench.bench(name, cfg, None)
                log.flush()
            for n,d,k in [(100000,8,8),(100000,32,64)]:
                bench.bench(f'small-{d}',dict(data='synthetic',rows=n,dims=d,k=k),None)

if __name__ == '__main__':
    main()
