"""Curved text (arcs, seals, waves, a label bowed round a bottle). The normal pass drops these lines: their upright box
is mostly background. Here the detector outlines each line as a polygon; every line the normal pass did not read is
straightened, then recognised.

A line is straightened three ways and the best reading wins:
  columns  each pixel column is shifted so the line's centre becomes level (waves, bowed lines, gentle arcs)
  ring     the line is unrolled round its fitted circle, letter tops outward (top of a seal)
  ring180  the same, letter tops inward (bottom of a seal)
shortcut: only lines that run roughly left-to-right or round a circle; a wavy line that runs vertically is not read."""

import math
import shutil
from pathlib import Path

import cv2
import numpy as np

MODELS = Path.home() / ".paddlex" / "official_models"
_det: dict[str, object] = {}
_rec: dict[str, object] = {}


def detector(det: str):
    """The configured detection model, set to outline lines as polygons scored on the text itself (a private copy of
    the model folder with two extra lines in its inference.yml; PaddleOCR has no parameter for this)."""
    if det not in _det:
        from paddleocr import TextDetection

        copy = MODELS / f"{det}_poly"
        if not copy.exists():
            shutil.copytree(MODELS / det, copy)
            yml = copy / "inference.yml"
            yml.write_text(yml.read_text(encoding="utf-8").replace("  name: DBPostProcess\n", "  name: DBPostProcess\n  box_type: poly\n  score_mode: slow\n"), encoding="utf-8")
        _det[det] = TextDetection(model_name=det, model_dir=str(copy), enable_mkldnn=False)
    return _det[det]


def recogniser(rec: str):
    if rec not in _rec:
        from paddleocr import TextRecognition

        _rec[rec] = TextRecognition(model_name=rec, enable_mkldnn=False)
    return _rec[rec]


def _smooth(v: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    return np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def columns(bgr: np.ndarray, poly: np.ndarray, others: list[tuple] = ()):
    """-> (strip, box(a, b)): box maps strip columns a..b back to a page box. None when the line doubles back.
    `others` = boxes of the other text on the page: the strip is widened a little at each end where none is next to it."""
    x0, y0, x1, y1 = (int(v) for v in (*poly.min(0), *poly.max(0)))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mask, [(poly - (x0, y0)).astype(np.int32)], 1)
    filled = mask.any(0)
    if filled.sum() < 8:
        return None
    top = mask.argmax(0).astype(np.float32)
    bottom = (mask.shape[0] - 1 - mask[::-1].argmax(0)).astype(np.float32)
    thick = bottom - top + 1
    h = float(np.median(thick[filled]))
    if (thick[filled] > 2.2 * h).mean() > 0.2:  # two runs of text in one column: an arc that turns back on itself
        return None
    xs = np.flatnonzero(filled)
    centre = np.interp(np.arange(mask.shape[1]), xs, ((top + bottom) / 2)[xs])
    # The outline is often tight on the first and last letter, and the reader then drops them ("omposisi").
    pad = int(h / 2)
    centre = _smooth(centre, int(h)) + y0

    def free(x: float, y: float, side: int) -> bool:
        """No other text lies beyond this end of the line (lines above and below start at about the same x: they do not count)."""
        return not any(b[0] < x + pad and b[2] > x - pad and b[1] < y + h / 2 and b[3] > y - h / 2 and ((b[0] + b[2]) / 2 - x) * side > 0 for b in others)

    left, right = pad * free(x0, centre[0], -1), pad * free(x1, centre[-1], 1)
    centre = np.pad(centre, (left, right), mode="edge")
    H = int(h * 1.15) + 2
    map_x = np.tile(np.arange(x0 - left, x1 + right + 1, dtype=np.float32), (H, 1))
    map_y = (centre[None, :] + (np.arange(H)[:, None] - (H - 1) / 2)).astype(np.float32)
    strip = cv2.remap(bgr, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def box(a: float, b: float):
        a, b = int(max(0, a)), int(min(len(centre) - 1, max(a + 1, b)))
        c = centre[a : b + 1]
        return (float(x0 - left + a), float(c.min() - H / 2), float(x0 - left + b), float(c.max() + H / 2))

    return strip, box


def ring(bgr: np.ndarray, poly: np.ndarray, tops_outward: bool):
    """Unroll the line round the circle fitted to its outline."""
    x0, y0, x1, y1 = (int(v) for v in (*poly.min(0), *poly.max(0)))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    cv2.fillPoly(mask, [(poly - (x0, y0)).astype(np.int32)], 1)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    px, py = xs + x0, ys + y0
    # Least-squares circle through the pixels of the line: x² + y² = 2ax + 2by + c
    (a, b, c), *_ = np.linalg.lstsq(np.c_[2 * px, 2 * py, np.ones(len(px))], px**2 + py**2, rcond=None)
    R = math.sqrt(max(1.0, c + a * a + b * b))
    if R > 20 * max(x1 - x0, y1 - y0):  # practically straight
        return None
    r = np.hypot(px - a, py - b)
    r_in, r_out = np.percentile(r, [1, 99])
    ang = np.sort(np.arctan2(py - b, px - a))  # image y points down, so angles grow clockwise = reading direction of a top arc
    gaps = np.diff(np.r_[ang, ang[0] + 2 * math.pi])
    start = ang[(gaps.argmax() + 1) % len(ang)]  # the line starts after the widest empty stretch of the circle
    span = 2 * math.pi - gaps.max()
    W, H = int(R * span), int(r_out - r_in) + 4
    if W < 8 or H < 8:
        return None
    theta = start + span * np.arange(W) / W
    radius = (r_out + 2) - np.arange(H)  # row 0 = outer edge
    map_x = (a + radius[:, None] * np.cos(theta)[None, :]).astype(np.float32)
    map_y = (b + radius[:, None] * np.sin(theta)[None, :]).astype(np.float32)
    strip = cv2.remap(bgr, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if not tops_outward:
        strip = strip[::-1, ::-1]

    def box(s: float, e: float):
        if not tops_outward:
            s, e = W - e, W - s
        t = start + span * np.linspace(max(0, s), min(W, e), 8) / W
        xs_ = np.r_[a + r_in * np.cos(t), a + r_out * np.cos(t)]
        ys_ = np.r_[b + r_in * np.sin(t), b + r_out * np.sin(t)]
        return (float(xs_.min()), float(ys_.min()), float(xs_.max()), float(ys_.max()))

    return np.ascontiguousarray(strip), box


def read(bgr: np.ndarray, recs: list[str], det: str, already: list[tuple]) -> list[tuple[str, float, tuple, list[tuple[str, tuple]], str]]:
    """Lines the normal pass did not read (`already` = the line boxes it did) -> [(text, score, line box, [(word, box)], recognition model)]."""
    out = []
    polys = [np.asarray(p, dtype=np.float32) for p in detector(det).predict(bgr)[0]["dt_polys"]]
    for poly in polys:
        others = list(already) + [(*p.min(0), *p.max(0)) for p in polys if p is not poly]
        x0, y0, x1, y1 = (float(v) for v in (*poly.min(0), *poly.max(0)))
        area = (x1 - x0) * (y1 - y0)
        if len(poly) < 4 or min(x1 - x0, y1 - y0) < 10:
            continue
        if any(max(0.0, min(x1, k[2]) - max(x0, k[0])) * max(0.0, min(y1, k[3]) - max(y0, k[1])) >= 0.5 * area for k in already):
            continue
        strips = [s for s in (columns(bgr, poly, others), ring(bgr, poly, True), ring(bgr, poly, False)) if s]
        best = None
        for strip, box in strips:
            for rec in recs:
                r = recogniser(rec).predict(strip)[0]
                text, score = r["rec_text"].strip(), float(r["rec_score"])
                n = sum(ch.isalnum() for ch in text)
                rank = score * (1 - 1 / (1 + n))  # same weighing as the normal pass: short garbage scores deceptively high
                if n and (best is None or rank > best[0]):
                    best = (rank, text, score, strip.shape[1], box, rec)
        if best is None or best[0] <= 0.5:  # a logo or pattern outlined as 'text' reads as one or two unsure characters
            continue
        _, text, score, width, box, rec = best
        words = text.split()
        if any("֐" <= ch <= "ࣿ" for ch in text):  # right-to-left: the first word is at the right end of the strip
            words = words[::-1]
        unit = width / max(1, sum(map(len, words)) + len(words) - 1)
        parts, at = [], 0.0
        for w in words:  # each word gets its share of the line by length
            if any(ch.isalnum() for ch in w):
                parts.append((w, box(at, at + unit * len(w))))
            at += unit * (len(w) + 1)
        out.append((text, score, (x0, y0, x1, y1), parts, rec))
    return out
