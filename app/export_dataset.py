"""Exports checker corrections as a PaddleOCR recognition dataset (our own training data, stays on this machine).

    uv run --env-file .env python -m app.export_dataset ../storage/ocr-dataset

Output: <out>/images/<id>.png (word crops) + <out>/rec_gt.txt ("images/<id>.png<TAB>correct text").
Fine-tune PP-OCRv6 recognition on it with PaddleOCR's training tools, evaluate CER on a holdout,
then point text_recognition_model_dir at the new model only if it beats the current one.
shortcut: export only; scheduled fine-tune + model registry (concept §3.6) come in phase 2.
"""

import os
import sys
from pathlib import Path

import psycopg
from PIL import Image

STORAGE = Path(os.environ.get("STORAGE_DIR", "../storage")).resolve()


def main(out: Path):
    (out / "images").mkdir(parents=True, exist_ok=True)
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn:
        rows = conn.execute(
            """select c.id, p.image_path, c.bbox, c.correct_text from ocr_corrections c
               join label_pages p on p.version_id = c.version_id and p.page_no = c.page_no order by c.id"""
        ).fetchall()
    lines = []
    for cid, image_path, bbox, text in rows:
        x0, y0, x1, y1 = bbox
        pad = max(2, (y1 - y0) * 0.15)
        with Image.open(STORAGE / image_path) as img:
            img.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad)).save(out / "images" / f"{cid}.png")
        lines.append(f"images/{cid}.png\t{text}")
    (out / "rec_gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} samples -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "../storage/ocr-dataset"))
