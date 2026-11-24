"""Real OCR + language detection (needs the downloaded models, ~20 s):  uv run python -m unittest tests.test_ocr"""

import unittest

from PIL import Image, ImageDraw, ImageFont

from app import ocr, worker
from app.ingest import Word

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 40)
TEXTS = {0: "Isi bersih 500 ml horizontal", 180: "Simpan di tempat sejuk", 90: "Kocok sebelum diminum", 270: "Komposisi air gula garam"}
AT = {0: (300, 40), 180: (300, 200), 90: (60, 150), 270: (1250, 150)}


def strip(text: str) -> Image.Image:
    im = Image.new("RGB", (900, 70), "white")
    ImageDraw.Draw(im).text((10, 10), text, fill="black", font=FONT)
    return im


class Rotations(unittest.TestCase):
    def test_reads_text_in_all_four_directions_with_boxes_in_place(self):
        page = Image.new("RGB", (1400, 1100), "white")
        placed = {}
        for angle, text in TEXTS.items():
            piece = strip(text).rotate(angle, expand=True)
            page.paste(piece, AT[angle])
            placed[angle] = (*AT[angle], AT[angle][0] + piece.width, AT[angle][1] + piece.height)
        words = ocr.read(page, ["id"], rotations=True)
        got = {w.text for w in words}
        for text in TEXTS.values():
            for token in text.split():
                self.assertIn(token, got, f"missing '{token}' from: {sorted(got)}")
        # every word box must lie inside the strip its text was drawn in (boxes are mapped back from the rotated pass)
        for angle, text in TEXTS.items():
            x0, y0, x1, y1 = placed[angle]
            for w in words:
                if w.text in text.split():
                    cx, cy = (w.bbox[0] + w.bbox[2]) / 2, (w.bbox[1] + w.bbox[3]) / 2
                    self.assertTrue(x0 <= cx <= x1 and y0 <= cy <= y1, f"{w.text} at {w.bbox}, expected inside {placed[angle]}")
        self.assertLessEqual(len(words), sum(len(t.split()) for t in TEXTS.values()) + 2, f"garbage readings kept: {sorted(got)}")


def line(text: str, y: int) -> list[Word]:
    return [Word(t, (i * 50, y, i * 50 + 45, y + 20), 1.0, "text_layer") for i, t in enumerate(text.split())]


KNOWN = {"id": "Indonesia", "en": "English", "ms": "Melayu", "vi": "Vietnam", "ar": "Arab", "th": "Thai", "zh": "Mandarin", "ja": "Jepang"}


class Languages(unittest.TestCase):
    def test_declared_languages_are_assigned_without_findings(self):
        words = line("Simpan di tempat sejuk dan kering, hindarkan dari sinar matahari langsung", 0) + line("Store in a cool and dry place away from direct sunlight", 50)
        self.assertEqual(worker.assign_languages(words, ["id", "en"], KNOWN, 1), [])
        self.assertEqual({w.lang for w in words[:11]}, {"id"})
        self.assertEqual({w.lang for w in words[11:]}, {"en"})

    def test_undeclared_language_is_reported(self):
        words = line("Store in a cool and dry place away from direct sunlight", 0) + line("เก็บในที่เย็นและแห้ง หลีกเลี่ยงแสงแดดโดยตรง", 50)
        found = worker.assign_languages(words, ["id", "en"], KNOWN, 1)
        self.assertEqual([(f.check_type, f.lang) for f in found], [("language", "th")])

    def test_close_languages_do_not_false_alarm(self):
        # Malay-looking Indonesian must not be reported while Indonesian is declared
        words = line("Simpan di tempat yang sejuk dan kering selepas dibuka", 0)
        self.assertEqual(worker.assign_languages(words, ["id"], KNOWN, 1), [])


if __name__ == "__main__":
    unittest.main()
