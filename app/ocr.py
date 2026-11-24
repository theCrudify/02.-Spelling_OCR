"""Local OCR with PaddleOCR. Models are loaded from ~/.paddlex/official_models (downloaded once by setup_offline.sh)."""

import logging
import os
import re
import unicodedata
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")  # no network check at startup

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

from .checks import CHAR_SCRIPT
from .ingest import Word

log = logging.getLogger("ocr")
_engines: dict[tuple[str, str], PaddleOCR] = {}

# Letters the PaddleOCR models cannot output at all (not in their character list): Vietnamese ấ ệ ổ ớ ả ể …
# For these languages every line is read again by Tesseract (bundled in the tesserocr wheel, data in tessdata/).
TESSERACT = {"vi": "vie"}  # PaddleOCR language code -> Tesseract language
PADDLE_BLIND = re.compile(r"[\u1ea0-\u1ef9]")
TESSDATA = Path(__file__).resolve().parent.parent / "tessdata"
_tess: dict[str, object] = {}


def tesseract(lang: str):
    if lang not in _tess:
        _tess[lang] = None
        try:
            import tesserocr

            _tess[lang] = tesserocr.PyTessBaseAPI(path=str(TESSDATA), lang=lang, psm=tesserocr.PSM.SINGLE_LINE)
        except Exception as e:  # the label is still read by PaddleOCR, only without those letters
            log.warning("Tesseract '%s' not available (%s): run scripts/setup_offline.sh", lang, e)
    return _tess[lang]


def _reread(image: Image.Image, line_box, lang: str) -> tuple[list[tuple[str, tuple]], float] | None:
    """Tesseract reading of one detected line, as ([(word, box)], confidence); None unless it contains a letter
    PaddleOCR cannot produce, because everywhere else PaddleOCR is the better reader."""
    api = tesseract(lang)
    if api is None:
        return None
    from tesserocr import RIL, iterate_level

    pad = 4
    x0, y0 = max(0, int(line_box[0]) - pad), max(0, int(line_box[1]) - pad)
    api.SetImage(image.crop((x0, y0, int(line_box[2]) + pad, int(line_box[3]) + pad)))
    api.Recognize()
    parts, confs = [], []
    for it in iterate_level(api.GetIterator(), RIL.WORD):
        text = unicodedata.normalize("NFC", it.GetUTF8Text(RIL.WORD) or "").strip()
        box = it.BoundingBox(RIL.WORD)
        if box and any(ch.isalnum() for ch in text):
            parts.append((text, (box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0)))
            confs.append(it.Confidence(RIL.WORD) / 100)
    if not any(PADDLE_BLIND.search(t) for t, _ in parts):
        return None
    return parts, min(confs)


def _rtl_words(text: str, line_box) -> list[tuple[str, tuple]]:
    """PaddleOCR gives no usable word boxes for right-to-left lines (its per-word text is in visual, i.e. reversed,
    order), so each word gets its share of the line by length, from the right.
    shortcut: boxes are approximate (letters are not equally wide); good enough to point at the word in the viewer."""
    words = text.split()
    x0, y0, x1, y1 = (float(v) for v in line_box)
    if (x1 - x0) < (y1 - y0) or not words:  # vertical line: keep it whole
        return [(text.strip(), line_box)] if text.strip() else []
    unit = (x1 - x0) / (sum(map(len, words)) + len(words) - 1)
    out, right = [], x1
    for w in words:
        out.append((w, (right - unit * len(w), y0, right, y1)))
        right -= unit * (len(w) + 1)
    return [(t, b) for t, b in out if any(ch.isalnum() for ch in t)]


def rec_model(lang: str) -> str:
    """PaddleOCR's own language -> recognition model table (id, ms, vi, en, ch, japan share PP-OCRv6; ar, th have their own)."""
    _, rec = PaddleOCR._get_ocr_model_names(None, lang, None)
    if rec is None:
        raise ValueError(f"PaddleOCR has no model for language code '{lang}' (check Bahasa & OCR in admin)")
    return rec


DETECTION = "PP-OCRv5_mobile_det"  # fallback; the live value is the setting ocr_detection_model


def engine(rec: str, det: str = DETECTION) -> PaddleOCR:
    if (det, rec) not in _engines:
        _engines[det, rec] = PaddleOCR(
            text_detection_model_name=det,
            text_recognition_model_name=rec,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # oneDNN crashes on paddlepaddle 3.3 CPU ("ConvertPirAttribute2RuntimeAttribute")
        )
    return _engines[det, rec]


def models_for(ocr_langs: list[str]) -> list[str]:
    """One OCR run per distinct recognition model."""
    return list(dict.fromkeys(rec_model(lang) for lang in ocr_langs))


RTL = re.compile(r"[\u0590-\u08ff]")


def _group(tokens, boxes) -> list[tuple[str, tuple]]:
    """PaddleOCR cuts a line at every accent and punctuation mark ('gi','ú','p' / '27',',','5'), but a word ends
    only at whitespace. CJK and Thai keep one token per character."""
    out: list[list] = []
    glue = False
    for tok, box in zip(tokens, boxes):
        text = tok.strip()
        if text and glue and not tok[0].isspace() and not CHAR_SCRIPT.search(out[-1][0][-1] + text[0]):
            b = out[-1][1]
            out[-1] = [out[-1][0] + text, (min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3]))]
        elif text:
            out.append([text, box])
        glue = bool(text) and not tok[-1].isspace()
    return [(t, b) for t, b in out if any(ch.isalnum() for ch in t)]


def _join_spaced(rows: list[tuple]) -> list[tuple]:
    """Letter-spaced titles ('S O Y J O Y') come back as one short line per letter or pair; three or more of those
    side by side on one baseline are one word. rows = (text, tokens, boxes, score, line box) of one pass."""
    def fragment(r) -> bool:
        t = r[0].strip()
        return 0 < len(t) <= 2 and t.isalpha() and not CHAR_SCRIPT.search(t)

    out = [r for r in rows if not fragment(r)]
    left = sorted((r for r in rows if fragment(r)), key=lambda r: r[4][0])
    while left:
        run = [left.pop(0)]
        for r in left[:]:
            a, b = run[-1][4], r[4]
            h = a[3] - a[1]
            if abs((a[1] + a[3]) - (b[1] + b[3])) < h and 0.6 < (b[3] - b[1]) / h < 1.6 and b[0] - a[2] < h:
                run.append(r)
                left.remove(r)
        if len(run) < 3:
            out += run
            continue
        text = "".join(r[0].strip() for r in run)
        box = (min(r[4][0] for r in run), min(r[4][1] for r in run), max(r[4][2] for r in run), max(r[4][3] for r in run))
        out.append((text, [text], [box], min(r[3] for r in run), box))
    return out


def _unrotate(box, angle: int, size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Box in an image rotated `angle`° counter-clockwise (PIL rotate, expand=True) -> box in the original image."""
    w, h = size
    x0, y0, x1, y1 = (float(v) for v in box)
    if angle == 90:
        return (w - y1, x0, w - y0, x1)
    if angle == 180:
        return (w - x1, h - y1, w - x0, h - y0)
    if angle == 270:
        return (y0, h - x1, y1, h - x0)
    return (x0, y0, x1, y1)


def read(image: Image.Image, ocr_langs: list[str], rotations: bool = False, det: str = DETECTION) -> list[Word]:
    """OCR of one page. With `rotations`, the page is also read turned 90/180/270° and the best reading of each
    text line wins, so sideways and upside-down text is read too (4× the time)."""
    lines: list[tuple[float, tuple, list[Word], tuple]] = []  # (rank, line box, words, pass)
    reread = [TESSERACT[lang] for lang in ocr_langs if lang in TESSERACT]
    for angle in (0, 90, 180, 270) if rotations else (0,):
        turned = image.rotate(angle, expand=True) if angle else image
        bgr = np.array(turned)[:, :, ::-1]
        for rec in models_for(ocr_langs):
            r = engine(rec, det).predict(bgr, return_word_box=True)[0]
            for text, tokens, boxes, score, line_box in _join_spaced(list(zip(r["rec_texts"], r["text_word"], r["text_word_boxes"], r["rec_scores"], r["rec_boxes"]))):
                score = float(score)
                if RTL.search(text):
                    parts = _rtl_words(text, line_box)
                else:
                    parts = _group(tokens, boxes)
                    for lang in reread:
                        again = _reread(turned, line_box, lang)
                        if again:
                            parts, score = again
                            break
                words = [Word(t, _unrotate(box, angle, image.size), score, "ocr") for t, box in parts]
                if not words:
                    continue
                # A wrong-way reading is short garbage with a deceptively high score ("ocu" 0.98), so weigh by length.
                # Prefer the pass where the line lies horizontal: its per-word boxes are the reliable ones.
                n = sum(ch.isalnum() for w in words for ch in w.text)
                horizontal = (line_box[2] - line_box[0]) >= (line_box[3] - line_box[1])
                lines.append((score * (1 - 1 / (1 + n)) + (0.02 if horizontal else 0), _unrotate(line_box, angle, image.size), words, (angle, rec)))
    kept: list[tuple[float, tuple, list[Word], tuple]] = []
    for line in sorted(lines, key=lambda x: -x[0]):
        # Lines of one pass are all different text; on a skewed photo their upright boxes overlap, so only compare across passes.
        if all(line[3] == k[3] or _overlap(line[1], k[1]) < 0.5 for k in kept):
            kept.append(line)
    return [w for _, _, words, _ in kept for w in words]


def _overlap(a, b) -> float:
    """Intersection over the smaller box: the same line seen in two passes, even when one box is a bit larger."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def merge(text_layer: list[Word], ocr_words: list[Word]) -> list[Word]:
    """Text layer is exact; OCR adds only what the text layer does not cover (outlined / rasterised text)."""
    def covered(w: Word) -> bool:
        cx, cy = (w.bbox[0] + w.bbox[2]) / 2, (w.bbox[1] + w.bbox[3]) / 2
        return any(t.bbox[0] - 2 <= cx <= t.bbox[2] + 2 and t.bbox[1] - 2 <= cy <= t.bbox[3] + 2 for t in text_layer)

    return text_layer + [w for w in ocr_words if not covered(w)]
