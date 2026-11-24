#!/usr/bin/env bash
# One-time download of everything the worker needs (dictionaries + OCR models). After this the worker runs
# fully offline: only public files come IN, no label data ever goes out.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dictionaries
LO=https://raw.githubusercontent.com/LibreOffice/dictionaries/master
for d in id/id_ID vi/vi_VN ar/ar; do
  name=$(basename "$d")
  for ext in aff dic; do [ -s "dictionaries/$name.$ext" ] || curl -fsSL "$LO/$d.$ext" -o "dictionaries/$name.$ext"; done
done
for ext in aff dic; do [ -s "dictionaries/en_US.$ext" ] || curl -fsSL "$LO/en/en_US.$ext" -o "dictionaries/en_US.$ext"; done
# Tesseract data for the languages PaddleOCR cannot fully read (see TESSERACT in app/ocr.py).
mkdir -p tessdata
for l in vie; do [ -s "tessdata/$l.traineddata" ] || curl -fsSL "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/$l.traineddata" -o "tessdata/$l.traineddata"; done
# Downloads PaddleOCR models into ~/.paddlex/official_models for the languages in the database.
uv run --env-file .env python -m app.worker --download-models
echo "Offline assets ready."
