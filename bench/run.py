"""OCR benchmark:  uv run python -m bench.run [--only name,name] [--save label]

Scores what the checker would see: expected tokens the OCR missed, and tokens it produced that are not in the
artwork (each one would become a false finding). Tokens are compared the same way as the master-text check."""

import json
import sys
import time
from collections import Counter
from pathlib import Path

from app import checks, ocr

from .cases import cases

OCR_LANG = {"id": "id", "en": "en", "ms": "ms", "vi": "vi", "ar": "ar", "th": "th", "zh": "ch", "ja": "japan"}
OUT = Path(__file__).parent / "out"


def bag(text: str) -> Counter:
    return Counter(checks.norm(t) for t in checks.tokens(text))


def score(expected: str, words) -> dict:
    want, got = bag(expected), bag(" ".join(w.text for w in words))
    missed, extra = want - got, got - want
    n = sum(want.values())
    return {"tokens": n, "missed": sum(missed.values()), "extra": sum(extra.values()), "missed_list": sorted(missed.elements())[:12], "extra_list": sorted(extra.elements())[:12]}


def real_cases():
    out = {}
    for txt in sorted((Path(__file__).parent / "real").glob("*.txt")):
        img = next((p for p in txt.parent.glob(txt.stem + ".*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf")), None)
        if img:
            langs = txt.with_suffix(".langs").read_text().strip().split(",") if txt.with_suffix(".langs").exists() else ["id", "en"]
            out["real:" + txt.stem] = (img, txt.read_text(encoding="utf-8"), langs, True)
    return out


def main():
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
    label = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--save=")), None)
    OUT.mkdir(exist_ok=True)
    rows, tot = {}, Counter()
    for name, (image, expected, langs, rotations) in {**cases(), **real_cases()}.items():
        if only and name not in only:
            continue
        if isinstance(image, Path):  # real artwork goes through the same loader as the worker
            from app import ingest

            pages = ingest.load(str(image), 200)
            t0 = time.time()
            words = [w for p in pages for w in ocr.merge(p.words, ocr.read(p.image, [OCR_LANG[l] for l in langs], rotations))]
        else:
            image.save(OUT / f"{name}.png")
            t0 = time.time()
            words = ocr.read(image, [OCR_LANG[l] for l in langs], rotations)
        r = {**score(expected, words), "seconds": round(time.time() - t0, 1)}
        rows[name] = r
        tot.update({k: r[k] for k in ("tokens", "missed", "extra")})
        print(f"{name:22} tokens {r['tokens']:4}  missed {r['missed']:3}  extra {r['extra']:3}  {r['seconds']:5}s  miss={r['missed_list'][:6]} extra={r['extra_list'][:6]}", flush=True)
    acc = 1 - (tot["missed"] + tot["extra"]) / max(1, tot["tokens"])
    print(f"{'TOTAL':22} tokens {tot['tokens']:4}  missed {tot['missed']:3}  extra {tot['extra']:3}  token accuracy {acc:.1%}")
    if label:
        (OUT / f"result-{label}.json").write_text(json.dumps({"total": dict(tot), "accuracy": acc, "cases": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
