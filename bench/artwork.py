"""A flat label artwork as a designer would export it from Illustrator, to test the real use: "I finished the
design, are there typos?"

    uv run python -m bench.artwork ../docs/ocr/artwork        # needs inkscape

Writes four PDF-compatible .ai files (a PDF by content, which is what Illustrator saves by default) plus the approved
master text:
    label-v1.ai           live text, with the mistakes in TYPOS planted
    label-v1-outline.ai   the same with every text converted to outlines (no text layer: all of it goes through OCR)
    label-v2.ai / label-v2-outline.ai   corrected
shortcut: drawn with Inkscape, not Illustrator; test a real .ai from the design team before trusting these numbers."""

import subprocess
import sys
from html import escape
from pathlib import Path

# (correct, as mistyped in v1); the removed sentence and the short BPOM number are found by other checks than spelling
TYPOS = [
    ("natrium klorida", "natrium kloirda"),
    ("magnesium karbonat", "magnesium karbonnat"),
    ("elektrolit tubuh", "elektorlit tubuh"),
    ("electrolytes", "electrolites"),
    ("Isi bersih: 350 mL", "Isi bersih: 530 mL"),
    ("Karbohidrat Total 22g", "Karbohidrat Total 23g"),
    ("Tanpa pengawet dan pemanis buatan.", ""),
    ("BPOM RI MD 266610015254", "BPOM RI MD 26661001525"),
]

# (x, y, size, weight, text) in millimetres on a 220 × 80 wrap-around label; "|" separates table cells
GIZI = [
    (6, 12, 3.4, "bold", "INFORMASI NILAI GIZI"),
    (6, 18, 2.4, "normal", "Takaran saji 350 mL"),
    (6, 22, 2.4, "normal", "1 Sajian per Kemasan"),
    (6, 28, 2.6, "bold", "JUMLAH PER SAJIAN"),
    (6, 33, 2.4, "normal", "Energi Total 90 kkal"),
    (6, 39, 2.4, "normal", "Lemak Total 0g"),
    (6, 43, 2.4, "normal", "Protein 0g"),
    (6, 47, 2.4, "normal", "Karbohidrat Total 22g"),
    (6, 51, 2.4, "normal", "Gula 21g"),
    (6, 55, 2.4, "normal", "Garam (natrium) 170mg"),
    (6, 62, 1.9, "normal", "*Persen AKG berdasarkan kebutuhan energi 2150 kkal."),
    (6, 65, 1.9, "normal", "Kebutuhan energi Anda mungkin lebih tinggi atau lebih rendah."),
]
BRAND = [
    (84, 14, 3.2, "bold", "ION SUPPLY DRINK"),
    (84, 31, 15, "bold", "POCARI"),
    (84, 46, 15, "bold", "SWEAT"),
    (84, 66, 3.4, "bold", "Isi bersih: 350 mL"),
    (84, 72, 2.2, "normal", "Ion supply drink to replenish water and electrolytes."),
]
INFO = [
    (146, 10, 2.3, "normal", "POCARI SWEAT adalah minuman isotonik yang cepat"),
    (146, 13.4, 2.3, "normal", "menggantikan cairan dan elektrolit tubuh yang hilang."),
    (146, 19, 2.3, "normal", "Komposisi: Air, gula, pengatur keasaman, natrium klorida,"),
    (146, 22.4, 2.3, "normal", "kalium klorida, kalsium laktat, magnesium karbonat,"),
    (146, 25.8, 2.3, "normal", "perisa sintetik, jus sitrus, dan antioksidan asam askorbat."),
    (146, 29.2, 2.3, "normal", "Tanpa pengawet dan pemanis buatan."),
    (146, 32.6, 2.3, "normal", "Minuman rasa sitrus."),
    (146, 38, 2.3, "normal", "Lisensi dari:"),
    (146, 41.4, 2.3, "bold", "Otsuka Pharmaceutical Co., Ltd., Japan"),
    (146, 44.8, 2.3, "normal", "Diproduksi oleh:"),
    (146, 48.2, 2.3, "bold", "PT Amerta Indah Otsuka"),
    (146, 51.6, 2.3, "normal", "Sukabumi 43359, Jawa Barat, Indonesia"),
    (146, 57, 2.3, "bold", "BPOM RI MD 266610015254"),
    (146, 60.4, 2.3, "bold", "MINUMAN ISOTONIK"),
    (146, 63.8, 1.9, "normal", "Kode produksi & baik digunakan sebelum:"),
    (146, 66.4, 1.9, "normal", "tertera pada tutup botol."),
    (146, 72, 4.2, "bold", "HALAL"),
    (164, 72, 2.2, "bold", "INDONESIA"),
]
SIDE = "Simpan di tempat sejuk dan kering."  # runs bottom to top along the right edge
BADGE = "PILIHAN LEBIH SEHAT"  # on an arc


def svg(mistyped: bool) -> str:
    def fix(t: str) -> str:
        for good, bad in TYPOS if mistyped else []:
            t = t.replace(good, bad)
        return escape(t)

    text = "".join(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{fix(t)}</text>' for x, y, size, weight, t in GIZI + BRAND + INFO if fix(t))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="220mm" height="80mm" viewBox="0 0 220 80">
<rect width="220" height="80" fill="#0a4ea3"/>
<rect x="4" y="6" width="68" height="62" fill="none" stroke="white" stroke-width="0.4"/>
<line x1="6" y1="35.5" x2="70" y2="35.5" stroke="white" stroke-width="0.3"/>
<path d="M84 52 C100 49 118 60 138 52 L138 57 C118 63 100 53 84 58 Z" fill="white"/>
<path id="arc" d="M194 72 a9 9 0 0 1 18 0" fill="none"/>
<circle cx="203" cy="72" r="5" fill="#1a9b4b"/>
<g fill="white" font-family="Liberation Sans, Arial, DejaVu Sans, sans-serif">{text}
<text font-size="1.9" font-weight="bold"><textPath xlink:href="#arc" startOffset="4%">{escape(BADGE)}</textPath></text>
<text x="217" y="56" font-size="2.2" transform="rotate(-90 217 56)">{escape(SIDE)}</text>
</g></svg>"""


def master() -> str:
    return "\n".join([t for *_, t in GIZI + BRAND + INFO] + [SIDE, BADGE]) + "\n"


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, mistyped in (("label-v1", True), ("label-v2", False)):
        src = out / f"{name}.svg"
        src.write_text(svg(mistyped), encoding="utf-8")
        for suffix, flags in (("", []), ("-outline", ["--export-text-to-path"])):
            pdf = out / f"{name}{suffix}.pdf"
            subprocess.run(["inkscape", str(src), "--export-type=pdf", f"--export-filename={pdf}", *flags], check=True, capture_output=True)
            pdf.rename(pdf.with_suffix(".ai"))
            (out / f"{name}{suffix}.txt").write_text(master(), encoding="utf-8")  # master text beside every file, for bench.trace
        src.unlink()
    print(sorted(p.name for p in out.glob("*.ai")))


if __name__ == "__main__":
    main()
