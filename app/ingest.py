"""Artwork file -> pages: a rendered image (for OCR and the viewer) plus the text-layer words when the file has them."""

from dataclasses import dataclass, field

import pdfplumber
from PIL import Image, ImageSequence

Image.MAX_IMAGE_PIXELS = 400_000_000  # print artwork is large; files only come from logged-in users


@dataclass
class Word:
    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels of the page image
    conf: float
    source: str  # 'text_layer' | 'ocr'
    lang: str | None = None


@dataclass
class Page:
    image: Image.Image
    words: list[Word] = field(default_factory=list)


def kind(path: str) -> str:
    """By magic bytes, not extension: an .ai file is a PDF when saved 'PDF compatible', else PostScript."""
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"%PDF":
        return "pdf"
    if head == b"8BPS":
        return "psd"
    if head[:2] == b"%!" or head == b"\xc5\xd0\xd3\xc6":
        return "eps"
    return "image"


def load(path: str, dpi: int) -> list[Page]:
    k = kind(path)
    if k == "pdf":
        return _pdf(path, dpi)
    if k == "psd":
        from psd_tools import PSDImage

        # shortcut: PSD text layers are not read (the composite is OCR'd); read layer.text when designers keep live text.
        return [Page(PSDImage.open(path).composite().convert("RGB"))]
    if k == "eps":
        img = Image.open(path)  # rendered by Ghostscript
        img.load(scale=max(1, round(dpi / 72)))
        return [Page(img.convert("RGB"))]
    img = Image.open(path)
    return [Page(frame.convert("RGB")) for frame in ImageSequence.Iterator(img)]  # multi-page TIFF


def _pdf(path: str, dpi: int) -> list[Page]:
    pages = []
    scale = dpi / 72
    with pdfplumber.open(path) as doc:
        for p in doc.pages:
            image = p.to_image(resolution=dpi).original.convert("RGB")
            # Only level text is taken from the text layer: sideways text and text on a curve come out of the PDF as
            # loose letters ("k", "eri", "ej"), so those are left to OCR, which reads them from the rendered page.
            level = p.filter(lambda o: o.get("object_type") != "char" or abs(o["matrix"][1]) < 0.01 * max(abs(o["matrix"][0]), abs(o["matrix"][1])))
            words = [
                Word(w["text"], (w["x0"] * scale, w["top"] * scale, w["x1"] * scale, w["bottom"] * scale), 1.0, "text_layer")
                for w in level.extract_words(use_text_flow=True)
                if w["text"].strip()
            ]
            pages.append(Page(image, words))
    return pages
