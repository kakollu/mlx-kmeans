#!/usr/bin/env python3
"""Build the README figures from recorded measurements - no hand-entered numbers.

    python3 scripts/make_figures.py

Reads benchmarks/results.jsonl, benchmarks/scaling.json and the demo outputs; writes docs/*.svg and docs/*.png.
SVGs carry an explicit light background so they stay readable in GitHub's dark theme.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INK, INK2, INK3, RULE = "#16181d", "#4a4e58", "#7c808b", "#e2ded4"
PAPER, OURS, PUB, ACCENT = "#fbfaf7", "#1d4f8f", "#b8b2a4", "#e8a800"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"
SUITE = ["geo-trips", "satellite", "logs", "single-cell", "sift-k1024", "gist-k1024"]
LABELS = {"geo-trips": "GPS points<br>10M × 4, k=256", "satellite": "Satellite pixels<br>10M × 12, k=32",
          "logs": "Log vectors<br>10M × 32, k=256", "single-cell": "Single-cell<br>2M × 50, k=64",
          "sift-k1024": "SIFT1M<br>1M × 128, k=1024", "gist-k1024": "GIST1M<br>1M × 960, k=1024"}


def text(x, y, s, size=12, fill=INK, anchor="start", weight=400, mono=False):
    return (f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def latest_results():
    rows = [json.loads(l) for l in (ROOT / "benchmarks/results.jsonl").read_text().splitlines()]
    latest = {}
    for r in rows:
        if r.get("config") in SUITE:
            latest[(r["config"], r["impl"])] = r
    out = {}
    for cfg in SUITE:
        rs = [r for (c, _), r in latest.items() if c == cfg]
        ours = next((r for r in rs if r["impl"].startswith("ours") and "sec_per_pass" in r), None)
        pub = [r for r in rs if not r["impl"].startswith("ours") and "sec_per_pass" in r and "inertia" in r
               and abs(r["inertia"] - ours["inertia"]) / ours["inertia"] <= 1e-4]
        if ours and pub:
            best = min(pub, key=lambda r: r["sec_per_pass"])
            name = best["impl"].split(" (")[0].replace(" lloyd", "").replace(" elkan", "")
            out[cfg] = (ours["sec_per_pass"], best["sec_per_pass"], name)
    return out


def fig_speedup(res):
    """Horizontal paired bars: our pass time vs the fastest compared library that matches our clustering."""
    rowh, top, left, right, W = 62, 76, 210, 300, 900   # right margin holds the value label + ratio
    H = top + rowh * len(res) + 34
    longest = max(v[1] for v in res.values())
    track = W - left - right
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="k-means seconds per pass, this library versus the fastest compared library">',
           f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
           text(24, 34, "Seconds per k-means pass, same data and same result", 17, INK, weight=700),
           text(24, 55, "Apple M5 Max · each bar is the fastest compared library whose clustering matches ours", 12, INK3)]
    for i, (cfg, (ours, pub, lib)) in enumerate(res.items()):
        y = top + i * rowh
        name, shape = LABELS[cfg].split("<br>")
        svg += [text(24, y + 14, name, 12.5, INK, weight=600), text(24, y + 30, shape, 11, INK3, mono=True)]
        for j, (val, colour, label) in enumerate([(pub, PUB, lib), (ours, OURS, "this library")]):
            w = max(3, val / longest * track)
            yy = y + 2 + j * 17
            svg += [f'<rect x="{left}" y="{yy}" width="{w:.1f}" height="13" rx="3" fill="{colour}"/>',
                    text(left + w + 8, yy + 11, f"{val*1000:.1f} ms  {label}", 11, INK2, mono=True)]
        svg.append(text(W - 24, y + 24, f"{pub/ours:.1f}×", 19, ACCENT, anchor="end", weight=700, mono=True))
    svg.append(text(24, H - 12, "Bars are time — shorter is better. Right column: how many times faster.", 11, INK3))
    svg.append("</svg>")
    (DOCS / "speedup.svg").write_text("\n".join(svg))
    return len(res)


def fig_scaling():
    """Log-log: time per pass against rows, ours vs scikit-learn and FAISS."""
    path = ROOT / "benchmarks/scaling.json"
    if not path.exists():
        return False
    d = json.loads(path.read_text())
    pts = d["points"]
    W, H, L, R, T, B = 760, 430, 78, 150, 74, 56
    import math
    xs = [math.log10(p["rows"]) for p in pts]
    allv = [p[k] for p in pts for k in ("ours", "sklearn", "faiss")]
    ys = [math.log10(v) for v in allv]
    x0, x1, y0, y1 = min(xs), max(xs), math.floor(min(ys)), math.ceil(max(ys))
    px = lambda lx: L + (lx - x0) / (x1 - x0) * (W - L - R)
    py = lambda ly: H - B - (ly - y0) / (y1 - y0) * (H - T - B)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="Time per pass against dataset size">',
           f'<rect width="{W}" height="{H}" fill="{PAPER}"/>',
           text(24, 32, "Time per pass as the data grows", 17, INK, weight=700),
           text(24, 52, f'{d["dims"]} dimensions, k={d["k"]} · both axes logarithmic', 12, INK3)]
    for e in range(y0, y1 + 1):
        y = py(e)
        lab = f"{10**e:g} s" if e >= 0 else f"{10**(e+3):g} ms"
        svg += [f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="{RULE}" stroke-width="1"/>',
                text(L - 10, y + 4, lab, 11, INK3, anchor="end", mono=True)]
    for p in pts:
        x = px(math.log10(p["rows"]))
        svg.append(text(x, H - B + 20, f'{p["rows"]//1_000_000}M', 11, INK3, anchor="middle", mono=True))
    for key, colour, label in [("sklearn", "#b8b2a4", "scikit-learn"), ("faiss", "#8c8578", "FAISS"),
                               ("ours", OURS, "this library")]:
        pointstr = " ".join(f"{px(math.log10(p['rows'])):.1f},{py(math.log10(p[key])):.1f}" for p in pts)
        svg.append(f'<polyline points="{pointstr}" fill="none" stroke="{colour}" stroke-width="2.5" '
                   f'stroke-linejoin="round"/>')
        for p in pts:
            svg.append(f'<circle cx="{px(math.log10(p["rows"])):.1f}" cy="{py(math.log10(p[key])):.1f}" r="4.5" '
                       f'fill="{colour}" stroke="{PAPER}" stroke-width="1.5"/>')
        last = pts[-1]
        svg.append(text(px(math.log10(last["rows"])) + 12, py(math.log10(last[key])) + 4, label, 12,
                        INK if key == "ours" else INK2, weight=700 if key == "ours" else 400))
    fastest = pts[-1]["sklearn"] / pts[-1]["ours"]
    svg += [text(24, H - 16, f'At {pts[-1]["rows"]//1_000_000}M rows: {pts[-1]["ours"]*1000:.0f} ms per pass vs '
                 f'{pts[-1]["sklearn"]*1000:.0f} ms for scikit-learn ({fastest:.0f}× faster).', 11, INK3),
            "</svg>"]
    (DOCS / "scaling.svg").write_text("\n".join(svg))
    return True


def fig_landcover():
    """Classified Sentinel-2 scene beside a legend, so the colours can be read without the JSON."""
    src, res = ROOT / "demos/satellite_classes.png", ROOT / "demos/satellite_results.json"
    if not (src.exists() and res.exists()):
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    d = json.loads(res.read_text())
    im = Image.open(src).convert("RGB")
    im.thumbnail((820, 820))

    def font(size, bold=False):
        for p in ["/System/Library/Fonts/Supplemental/Helvetica.ttc", "/System/Library/Fonts/Helvetica.ttc"]:
            try:
                return ImageFont.truetype(p, size, index=1 if bold else 0)
            except OSError:
                continue
        return ImageFont.load_default()

    pad, legend_w, head = 22, 320, 92
    W, H = im.width + legend_w + pad * 3, max(im.height, 430) + head + pad * 2
    fig = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(fig)
    draw.text((pad, pad), "Land cover of the New York scene, found without labels", INK, font=font(22, True))
    draw.text((pad, pad + 32),
              f'Sentinel-2, 10 m pixels · {d["pixels"]:,} pixels × {d["dims"]} features clustered into '
              f'{d["k"]} classes in {d["fit_s"]:.1f} s', INK3, font=font(14))
    draw.text((pad, pad + 54), f'{d["machine"]} · k-means assigns every pixel; the names below are read off each '
              f"cluster's own NDVI and NDWI", INK3, font=font(14))
    fig.paste(im, (pad, head + pad))

    x, y = pad * 2 + im.width, head + pad + 4
    draw.text((x, y), "CLASSES BY AREA", INK2, font=font(12, True))
    y += 26
    for c in d["classes"]:
        draw.rounded_rectangle([x, y, x + 26, y + 26], 4, fill=tuple(c["colour"]))
        draw.text((x + 38, y - 1), f'{c["share"]*100:.1f}%   {c["km2"]:,.0f} km²', INK, font=font(14, True))
        draw.text((x + 38, y + 15), f'{c["kind"]}  ·  NDVI {c["ndvi"]:+.2f}', INK3, font=font(11))
        y += 38
    y += 6
    veg = sum(c["share"] for c in d["classes"] if "vegetation" in c["kind"])
    built = sum(c["share"] for c in d["classes"] if "built" in c["kind"])
    draw.line([x, y, x + legend_w - pad, y], fill=RULE, width=1)
    draw.text((x, y + 12), f"{veg*100:.0f}% vegetated, {built*100:.0f}% built-up", INK, font=font(14, True))
    draw.text((x, y + 32), "Three vegetation classes separate by", INK3, font=font(11))
    draw.text((x, y + 46), "canopy density, not by species.", INK3, font=font(11))
    fig.save(DOCS / "landcover.png", optimize=True)
    return True


def main():
    DOCS.mkdir(exist_ok=True)
    res = latest_results()
    n = fig_speedup(res)
    print(f"docs/speedup.svg      ({n} configs)")
    print(f"docs/scaling.svg      {'ok' if fig_scaling() else 'skipped (run scripts/scaling.py first)'}")
    print(f"docs/landcover.png    {'ok' if fig_landcover() else 'skipped (run demos/satellite_landcover.py first)'}")


if __name__ == "__main__":
    sys.exit(main())
