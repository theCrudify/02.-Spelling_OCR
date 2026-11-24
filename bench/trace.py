"""One label through the whole pipeline, recorded stage by stage (for the documentation):

    uv run --env-file .env python -m bench.trace bench/real/pocari_sachet_box.jpg id,en ../docs/ocr/pocari_sachet_box

Writes one picture per stage, hasil.md, trace.json and proses.mp4 (needs ffmpeg). Runs the same functions as the
worker with the live settings, dictionaries, glossary and country rules from the database (country: 4th argument,
default Indonesia). The master text is <image>.txt when it exists, for the label's first language."""

import itertools
import json
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app import checks, curved, ocr, worker
from app.ingest import load

FONT = next(p for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(p).exists())
SEVERITY = {"high": "#ef4444", "medium": "#f59e0b", "low": "#38bdf8"}
LANG = ["#22c55e", "#38bdf8", "#f59e0b", "#e879f9", "#f87171"]
W, H, FPS = 1280, 720, 25


def trace(path: Path, codes: list[str], country: str) -> dict:
    conn = worker.connect()
    langs = {r["code"]: r for r in conn.execute("select code, name, ocr_lang, dictionary from languages where is_active or code = any(%s)", (codes,)).fetchall()}
    cfg = worker.settings(conn)
    types = {r["code"]: r["severity"] for r in conn.execute("select code, severity from check_types where is_active").fetchall()}
    glossary = [r["term"] for r in conn.execute("select term from glossary_terms where is_active and (language is null or language = any(%s))", (codes,)).fetchall()]
    rules = conn.execute("select r.name, r.pattern, r.severity from country_rules r join countries c on c.id = r.country_id where r.is_active and c.name = %s", (country,)).fetchall()

    # Recorders wrapped round the pipeline's own functions; the pipeline itself is not changed.
    passes, born, strips, bent = [], [], [], {"on": False, "out": [], "seconds": 0.0}
    join, make, cols, ring, cread = ocr._join_spaced, ocr.Word, curved.columns, curved.ring, curved.read

    def _join(rows):
        out = join(rows)
        passes.append({"t": time.time(), "detected": len(rows), "lines": [(r[0], float(r[3]), [float(v) for v in r[4]]) for r in out]})
        return out

    def _make(*a):
        w = make(*a)
        born.append((w, "curved" if bent["on"] else len(passes) - 1))
        return w

    def _strip(fn):
        def f(bgr, poly, *a):
            r = fn(bgr, poly, *a)
            if r:
                strips.append((poly.tolist(), Image.fromarray(r[0][:, :, ::-1].copy())))
            return r

        return f

    def _cread(*a):
        bent["on"], t = True, time.time()
        out = cread(*a)
        bent.update(seconds=time.time() - t, out=[(o[0], o[1], [float(v) for v in o[2]]) for o in out])
        return out

    ocr._join_spaced, ocr.Word, curved.columns, curved.ring, curved.read = _join, _make, _strip(cols), _strip(ring), _cread
    page = load(str(path), int(cfg.get("render_dpi", 200)))[0]
    ocr_langs = [langs[c]["ocr_lang"] for c in codes]
    rotations = cfg.get("ocr_rotations", True)
    t0 = time.time()
    words = ocr.merge(page.words, ocr.read(page.image, ocr_langs, rotations, cfg.get("ocr_detection_model", ocr.DETECTION), cfg.get("ocr_curved_text", True)))
    seconds = time.time() - t0
    for p, prev, (angle, scale, rec) in zip(passes, [t0] + [p["t"] for p in passes], itertools.product((0, 90, 180, 270) if rotations else (0,), ocr.scales(page.image.size), ocr.models_for(ocr_langs))):
        p.update(angle=angle, scale=scale, rec=rec, seconds=p.pop("t") - prev)

    known = {c: r["name"] for c, r in langs.items()}
    language_findings = worker.assign_languages(words, codes, known, 1, cfg)
    ordered = [w for line in worker.lines(words) for w in line]
    master = path.with_suffix(".txt")
    masters = {codes[0]: master.read_text(encoding="utf-8")} if master.exists() else {}
    used = {w.lang for w in ordered} | set(codes)
    findings = checks.finalize(language_findings, types) + checks.run_all(
        [ordered],
        languages=codes,
        masters=masters,
        dictionaries={c: worker.dictionary(langs[c]["dictionary"]) for c in used if c in langs},
        glossary_terms=glossary,
        country_rules=rules,
        types=types,
        cfg=cfg,
    )
    return {"name": path.stem, "image": page.image, "text_layer": len(page.words), "passes": passes, "born": born, "strips": strips, "bent": bent, "words": ordered, "findings": findings, "seconds": seconds, "cfg": cfg, "master": bool(masters), "det": cfg.get("ocr_detection_model", ocr.DETECTION)}


# ---------------------------------------------------------------- drawing
def font(px: int):
    return ImageFont.truetype(FONT, px)


def fit(im: Image.Image, w: int, h: int) -> tuple[Image.Image, float]:
    s = min(w / im.width, h / im.height)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS), s


def boxes(im: Image.Image, s: float, items, width: int = 2, numbers: bool = False) -> Image.Image:
    """items = [(box in page pixels, colour)] drawn on a picture scaled by s."""
    d = ImageDraw.Draw(im)
    for i, (b, c) in enumerate(items, 1):
        d.rectangle([b[0] * s, b[1] * s, b[2] * s, b[3] * s], outline=c, width=width)
        if numbers:
            d.text((b[0] * s + 2, max(0, b[1] * s - 15)), str(i), fill=c, font=font(13), stroke_width=2, stroke_fill="black")
    return im


def size(p: dict) -> str:
    return "ukuran penuh" if p["scale"] == 1 else f"diperkecil ×{p['scale']:.2f}"


def turned(im: Image.Image, angle: int) -> Image.Image:
    return im.rotate(angle, expand=True) if angle else im


def stills(t: dict, out: Path) -> None:
    im, kept = t["image"], {id(w) for w in t["words"]}
    save = lambda pic, name: pic.convert("RGB").save(out / name, quality=82)
    base, s = fit(im, 1200, 1200)
    save(base, "1-asli.jpg")

    tiles = []
    for p in t["passes"]:
        pic, s2 = fit(turned(im, p["angle"]), 560, 560)
        tile = Image.new("RGB", (580, 610), "white")
        tile.paste(boxes(pic, s2, [(l[2], "#dc2626") for l in p["lines"]]), (10, 40))
        ImageDraw.Draw(tile).text((10, 8), f"{p['angle']}° {size(p)}: {len(p['lines'])} baris, {p['seconds']:.0f} dtk", fill="black", font=font(18))
        tiles.append(tile)
    sheet = Image.new("RGB", (580 * min(2, len(tiles)), 610 * -(-len(tiles) // 2)), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (580 * (i % 2), 610 * (i // 2)))
    save(sheet, "2-arah.jpg")

    save(boxes(base.copy(), s, [(l[2], "#dc2626") for l in t["passes"][0]["lines"]], numbers=True), "3-baris.jpg")

    if t["strips"]:
        polys = list({json.dumps(p): p for p, _ in t["strips"]}.values())
        pic, d = base.copy(), None
        d = ImageDraw.Draw(pic)
        for p in polys:
            d.polygon([(x * s, y * s) for x, y in p], outline="#d946ef", width=3)
        rows = [fit(st, 900, 90)[0] for _, st in t["strips"][:8]]
        sheet = Image.new("RGB", (max(pic.width, 920), pic.height + sum(r.height + 10 for r in rows) + 10), "white")
        sheet.paste(pic, (0, 0))
        y = pic.height + 10
        for r in rows:
            sheet.paste(r, (10, y))
            y += r.height + 10
        save(sheet, "4-lengkung.jpg")

    save(boxes(base.copy(), s, [(w.bbox, "#16a34a" if id(w) in kept else "#dc2626") for w, _ in t["born"]]), "5-pilih.jpg")
    colour = {l: LANG[i % len(LANG)] for i, l in enumerate(dict.fromkeys(w.lang for w in t["words"]))}
    save(boxes(base.copy(), s, [(w.bbox, colour[w.lang]) for w in t["words"]]), "6-bahasa.jpg")
    save(boxes(base.copy(), s, [(f.bbox, SEVERITY.get(f.severity, "#dc2626")) for f in t["findings"] if f.bbox], width=3, numbers=True), "7-temuan.jpg")


def report(t: dict, out: Path, codes: list[str]) -> None:
    """hasil.md + trace.json: every number of this run."""
    im, kept, thr = t["image"], {id(w) for w in t["words"]}, float(t["cfg"].get("ocr_min_confidence", 0.8))
    origin = Counter(str(o) for w, o in t["born"] if id(w) in kept)
    md = [f"# Jejak OCR: {t['name']}", "", f"Gambar {im.width}×{im.height} px, bahasa label `{','.join(codes)}`, model deteksi `{t['det']}`. Total OCR {t['seconds']:.0f} dtk (CPU).", ""]
    md += ["![proses](proses.mp4)", "", "Video: [proses.mp4](proses.mp4)", "", "## 1. Artwork", "", "![asli](1-asli.jpg)", "", f"Kata dari text layer: {t['text_layer']} (foto/JPG tidak punya text layer, jadi semua teks dari OCR).", ""]
    md += ["## 2. Deteksi dan pembacaan per arah", "", "![arah](2-arah.jpg)", "", "| Arah | Ukuran | Model pengenal | Baris terdeteksi | Baris setelah huruf berjarak digabung | Kata yang akhirnya dipakai | Waktu |", "|---|---|---|---|---|---|---|"]
    md += [f"| {p['angle']}° | {size(p)} | `{p['rec']}` | {p['detected']} | {len(p['lines'])} | {origin.get(str(i), 0)} | {p['seconds']:.0f} dtk |" for i, p in enumerate(t["passes"])]
    md += ["", "## 3. Baris terbaca pada arah 0°, ukuran penuh", "", "![baris](3-baris.jpg)", "", "| # | Teks terbaca | Keyakinan |", "|---|---|---|"]
    md += [f"| {i} | {text.replace('|', '¦')} | {score:.0%}{' ⚠' if score < thr else ''} |" for i, (text, score, _) in enumerate(t["passes"][0]["lines"], 1)]
    md += ["", "## 4. Teks melengkung", ""]
    if t["strips"]:
        md += ["![lengkung](4-lengkung.jpg)", "", f"{len({json.dumps(p) for p, _ in t['strips']})} garis yang belum terbaca diluruskan ({t['bent']['seconds']:.0f} dtk); {len(t['bent']['out'])} lolos:", ""]
        md += [f"- `{text}` ({score:.0%})" for text, score, _ in t["bent"]["out"]] or ["- (tidak ada yang lolos ambang)"]
    else:
        md += [f"Tidak ada garis teks yang terlewat oleh pembacaan biasa ({t['bent']['seconds']:.0f} dtk untuk memeriksanya)."]
    md += ["", "## 5. Pemilihan bacaan terbaik", "", "![pilih](5-pilih.jpg)", "", f"{len(t['born'])} kata dihasilkan semua arah; {len(kept)} dipakai (hijau), sisanya bacaan ganda atau salah arah (merah). Dari jalur melengkung: {origin.get('curved', 0)} kata.", ""]
    md += ["## 6. Bahasa per baris", "", "![bahasa](6-bahasa.jpg)", "", "| Bahasa | Kata |", "|---|---|"] + [f"| {l} | {n} |" for l, n in Counter(w.lang for w in t["words"]).items()]
    md += ["", "Teks akhir (urutan baca):", "", "```", *textwrap.wrap(" ".join(w.text for w in t["words"]), 110), "```", ""]
    md += ["## 7. Temuan", "", "![temuan](7-temuan.jpg)", "", f"Teks master: {'salinan manual teks pada foto (`bench/real/' + t['name'] + '.txt`)' if t['master'] else 'tidak ada'}.", "", "| # | Jenis | Tingkat | Ditemukan | Seharusnya | Pesan |", "|---|---|---|---|---|---|"]
    numbered = iter(range(1, 10**6))
    md += [f"| {next(numbered) if f.bbox else '–'} | {f.check_type} | {f.severity} | {(f.found or '')[:40]} | {(f.expected or '')[:40]} | {f.message[:140].replace('|', '¦')} |" for f in t["findings"]]
    (out / "hasil.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "trace.json").write_text(
        json.dumps(
            {
                "seconds": t["seconds"],
                "passes": t["passes"],
                "curved": t["bent"]["out"],
                "words": [{"text": w.text, "bbox": [float(v) for v in w.bbox], "conf": w.conf, "lang": w.lang, "source": w.source} for w in t["words"]],
                "findings": [{"type": f.check_type, "severity": f.severity, "found": f.found, "expected": f.expected, "message": f.message, "bbox": [float(v) for v in f.bbox] if f.bbox else None} for f in t["findings"]],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- video
def frame(picture: Image.Image, step: str, caption: str, panel=(), crops=()) -> Image.Image:
    """Picture left, a text panel right (panel = [(text, colour)], crops = pictures above it), step + caption below."""
    f = Image.new("RGB", (W, H), "#0f172a")
    f.paste(picture, (20 + (820 - picture.width) // 2, 20 + (600 - picture.height) // 2))
    d = ImageDraw.Draw(f)
    y = 20
    for c in crops:
        c = fit(c, 400, 90)[0]
        f.paste(c, (860, y))
        y += c.height + 8
    for text, colour in panel:
        for line in textwrap.wrap(text, 46)[:3]:
            if y < 610:
                d.text((860, y), line, fill=colour, font=font(15))
            y += 20
        y += 4
    d.rectangle((0, 636, W, H), fill="#1e293b")
    d.text((20, 644), step, fill="#38bdf8", font=font(22))
    d.text((20, 678), caption[:150], fill="white", font=font(17))
    return f


def video(t: dict, out: Path) -> None:
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], stdin=subprocess.PIPE)

    def show(f: Image.Image, seconds: float) -> None:
        raw = f.tobytes()
        for _ in range(max(1, round(seconds * FPS))):
            ff.stdin.write(raw)

    im, kept, thr = t["image"], {id(w) for w in t["words"]}, float(t["cfg"].get("ocr_min_confidence", 0.8))
    base, s = fit(im, 820, 600)
    card = Image.new("RGB", (820, 600), "#0f172a")
    d = ImageDraw.Draw(card)
    d.text((40, 220), "Label Spelling: proses OCR", fill="white", font=font(44))
    d.text((40, 290), f"Label: {t['name']}", fill="#38bdf8", font=font(26))
    d.text((40, 335), "Rekaman dari jalur kode yang sama dengan worker", fill="#94a3b8", font=font(20))
    show(frame(card, "", ""), 2.5)
    show(frame(base, "1. Artwork dimuat", f"{im.width}×{im.height} px. PDF/AI dirender {int(t['cfg'].get('render_dpi', 200))} dpi dan text layer-nya dibaca langsung; foto seperti ini seluruhnya lewat OCR."), 3.5)

    for p in t["passes"]:
        page = turned(im, p["angle"])
        pic, s2 = fit(page, 820, 600)
        step = f"2. Deteksi + pengenalan, arah {p['angle']}°, {size(p)}  ({p['rec']})"
        show(frame(pic, step, f"Halaman diputar {p['angle']}°, {size(p)} (huruf sangat besar hanya terbaca pada salinan kecil). Model deteksi {t['det']} mencari baris teks."), 1.3)
        show(frame(boxes(pic.copy(), s2, [(l[2], "#facc15") for l in p["lines"]]), step, f"{p['detected']} area teks ditemukan."), 1.5)
        done, panel = pic.copy(), []
        each = min(0.45, (7 if p["angle"] == 0 and p["scale"] == 1 else 2.5) / max(1, len(p["lines"])))
        for i, (text, score, box) in enumerate(p["lines"], 1):
            panel.append((f"{score:.0%}  {text}", "#e2e8f0" if score >= thr else "#fca5a5"))
            crop = page.crop((max(0, box[0] - 6), max(0, box[1] - 6), box[2] + 6, box[3] + 6))
            show(frame(boxes(done.copy(), s2, [(box, "#facc15")], 4), step, f"Baris {i}/{len(p['lines'])}: potongan gambar dibaca model pengenal menjadi teks + angka keyakinan.", panel[-10:], [crop]), each)
            boxes(done, s2, [(box, "#22c55e")])
        show(frame(done, step, f"{len(p['lines'])} baris terbaca dalam {p['seconds']:.0f} detik. Bacaan salah arah biasanya pendek dan ngawur; nanti disaring.", panel[-12:]), 1.8)

    pic = base.copy()
    d = ImageDraw.Draw(pic)
    for poly in {json.dumps(p): p for p, _ in t["strips"]}.values():
        d.polygon([(x * s, y * s) for x, y in poly], outline="#e879f9", width=3)
    got = [(f"{score:.0%}  {text}", "#e2e8f0") for text, score, _ in t["bent"]["out"]]
    n = len({json.dumps(p) for p, _ in t["strips"]})
    show(frame(pic, "3. Teks melengkung", f"{n} garis yang belum terbaca diluruskan (geser kolom / buka lingkaran) lalu dibaca ulang; {len(got)} lolos." if n else "Tidak ada garis teks yang terlewat pembacaan biasa.", got or [("(tidak ada tambahan)", "#94a3b8")], [st for _, st in t["strips"][:4]]), 4)

    show(frame(boxes(base.copy(), s, [(w.bbox, "#22c55e" if id(w) in kept else "#ef4444") for w, _ in t["born"]]), "4. Pilih bacaan terbaik", f"{len(t['born'])} kata dari semua arah. Baris yang sama terbaca berkali-kali: menang nilai keyakinan × panjang teks. Merah dibuang."), 4)
    show(frame(boxes(base.copy(), s, [(w.bbox, "#22c55e") for w in t["words"]]), "4. Pilih bacaan terbaik", f"{len(kept)} kata dipakai, masing-masing dengan kotak posisi untuk Label Viewer."), 2.5)

    colour = {l: LANG[i % len(LANG)] for i, l in enumerate(dict.fromkeys(w.lang for w in t["words"]))}
    count = Counter(w.lang for w in t["words"])
    show(frame(boxes(base.copy(), s, [(w.bbox, colour[w.lang]) for w in t["words"]]), "5. Bahasa per baris", "Tiap baris diberi bahasa (lingua, hanya bahasa yang aktif di aplikasi) supaya dicek dengan kamus yang benar.", [(f"{l}: {count[l]} kata", colour[l]) for l in colour]), 3.5)

    done, panel = boxes(base.copy(), s, [(w.bbox, "#334155") for w in t["words"]], 1), []
    step = "6. Pengecekan: teks master, kamus, glossary, aturan negara, keyakinan OCR"
    for f in t["findings"][:16]:
        panel.append((f"[{f.severity}] {f.check_type}: {f.message}", SEVERITY.get(f.severity, "white")))
        cur = boxes(done.copy(), s, [(f.bbox, SEVERITY.get(f.severity, "#ef4444"))], 4) if f.bbox else done
        show(frame(cur, step, f.message, panel[-5:]), 1.6)
        if f.bbox:
            boxes(done, s, [(f.bbox, SEVERITY.get(f.severity, "#ef4444"))])
    by = Counter(f.check_type for f in t["findings"])
    show(frame(done, "7. Hasil disimpan, label masuk antrean review", f"{len(kept)} kata, {len(t['findings'])} temuan, OCR {t['seconds']:.0f} detik di CPU. Checker memutuskan tiap temuan di Label Viewer.", [(f"{k}: {v}", "#e2e8f0") for k, v in by.items()] or [("Tidak ada temuan", "#22c55e")]), 5)
    ff.stdin.close()
    ff.wait()


def main():
    path, codes, out = Path(sys.argv[1]), sys.argv[2].split(","), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    t = trace(path, codes, sys.argv[4] if len(sys.argv) > 4 else "Indonesia")
    stills(t, out)
    report(t, out, codes)
    video(t, out / "proses.mp4")
    print(f"{t['name']}: {len(t['words'])} kata, {len(t['findings'])} temuan, {t['seconds']:.0f} dtk -> {out}")


if __name__ == "__main__":
    main()
