"""Real OCR + language detection (needs the downloaded models, ~20 s):  uv run python -m unittest tests.test_ocr"""

import unittest
from pathlib import Path

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


class Lines(unittest.TestCase):
    """How PaddleOCR's raw lines become words (no model needed)."""

    def test_words_end_only_at_whitespace(self):
        tokens = ["Gula", " ", "27", ",", "5", " ", "gi", "ú", "p", " đ", "in", ", ", "mg", "/", "100"]
        boxes = [(i * 10, 0, i * 10 + 9, 20) for i in range(len(tokens))]
        got = ocr._group(tokens, boxes)
        self.assertEqual([t for t, _ in got], ["Gula", "27,5", "giúp", "đin,", "mg/100"])
        self.assertEqual(got[1][1], (20, 0, 49, 20))  # box spans the glued pieces

    def test_cjk_and_thai_stay_one_token_per_character(self):
        got = ocr._group(["饮", "料", "500", "ml"], [(i * 10, 0, i * 10 + 9, 20) for i in range(4)])
        self.assertEqual([t for t, _ in got], ["饮", "料", "500ml"])

    def test_letter_spaced_title_is_one_word_but_numbers_are_not(self):
        def row(text, x, y=0):
            return (text, [text], [(x, y, x + 80, y + 100)], 0.9, (x, y, x + 80, y + 100))

        rows = [row("S", 0), row("O", 110), row("YJ", 220), row("O", 330), row("Y", 440), row("POCARI", 0, 300), row("1", 0, 600), row("2", 110, 600), row("3", 220, 600), row("A", 0, 900), row("B", 110, 900), row("C", 0, 1200), row("D", 150, 1200), row("E", 300, 1200)]
        got = sorted(r[0] for r in ocr._join_spaced(rows))
        self.assertEqual(got, sorted(["SOYJOY", "POCARI", "1", "2", "3", "A", "B", "C", "D", "E"]))  # C D E: too far apart to be one word

    def test_right_to_left_words_are_boxed_from_the_right(self):
        got = ocr._rtl_words("مشروب متساوي", (0, 0, 120, 30))
        self.assertEqual([t for t, _ in got], ["مشروب", "متساوي"])
        self.assertEqual(got[0][1], (70.0, 0.0, 120.0, 30.0))
        self.assertEqual(got[1][1], (0.0, 0.0, 60.0, 30.0))


class Curved(unittest.TestCase):
    def test_text_on_an_arc_and_under_a_seal_is_read_with_boxes_on_the_arc(self):
        from bench.cases import arc, font, on_path

        page = Image.new("RGB", (900, 900), "white")
        top, bottom = "KUALITAS TERJAMIN", "PRODUK HALAL INDONESIA"
        on_path(page, top, "bold", 44, arc(450, 450, 300, font("bold", 44).getlength(top)))
        on_path(page, bottom, "bold", 40, arc(450, 450, 340, font("bold", 40).getlength(bottom), top=False))
        self.assertEqual([w.text for w in ocr.read(page, ["id"], curves=False)], [], "the normal pass is expected to miss arcs")
        words = {w.text: w for w in ocr.read(page, ["id"])}
        self.assertEqual(sorted(words), sorted((top + " " + bottom).split()))
        self.assertLess(words["KUALITAS"].bbox[2], words["TERJAMIN"].bbox[2])  # top arc reads left to right
        self.assertLess(words["PRODUK"].bbox[0], words["INDONESIA"].bbox[0])  # so does the bottom one
        self.assertLess(words["TERJAMIN"].bbox[3], 450)  # top arc lies above the centre, bottom arc below it
        self.assertGreater(words["HALAL"].bbox[1], 450)


class Choice(unittest.TestCase):
    """Which reading of a line wins (the engine is faked: one list of (text, score, box) per pass, in pass order)."""

    def read(self, size, passes, **kw):
        calls = iter(passes)

        class Fake:
            def predict(self, bgr, return_word_box):
                rows = next(calls)
                return [{"rec_texts": [r[0] for r in rows], "text_word": [[r[0]] for r in rows], "text_word_boxes": [[r[2]] for r in rows], "rec_scores": [r[1] for r in rows], "rec_boxes": [r[2] for r in rows]}]

        real, ocr.engine = ocr.engine, lambda rec, det: Fake()
        try:
            return sorted(w.text for w in ocr.read(Image.new("RGB", size), ["id"], curves=False, **kw))
        finally:
            ocr.engine = real

    def test_huge_title_is_taken_from_the_shrunk_copy_and_keeps_the_small_print_beside_it(self):
        s = ocr.scales((2000, 3000))[1]
        small = lambda b: tuple(v * s for v in b)
        full = [("SUPPLY", 1.0, (100, 100, 700, 200)), ("D", 0.6, (300, 300, 340, 350)), ("Isi", 0.9, (100, 2000, 400, 2100))]
        shrunk = [("POCARI", 1.0, small((90, 150, 1200, 450))), ("si", 1.0, small((100, 2000, 400, 2100)))]
        self.assertEqual(self.read((2000, 3000), [full, shrunk]), ["Isi", "POCARI", "SUPPLY"])  # 'D' (a piece of the title) and 'si' are dropped

    def test_unsure_single_characters_are_dropped_as_graphics(self):
        rows = [("Komposisi", 0.99, (100, 100, 500, 140)), ("à", 0.23, (700, 100, 710, 110)), ("电", 0.7, (700, 300, 740, 340)), ("1", 1.0, (100, 300, 110, 340))]
        self.assertEqual(self.read((1000, 800), [rows]), ["1", "Komposisi"])

    def test_short_line_is_read_in_the_direction_of_the_rest_of_the_page(self):
        upright = [("Karbohidrat", 1.0, (100, 100, 500, 140)), ("0%", 1.0, (600, 100, 660, 140))]
        upside_down = [("900", 1.0, (340, 660, 400, 700))]  # the same spot as "0%", page turned 180°
        self.assertEqual(self.read((1000, 800), [upright, [], upside_down, []], rotations=True), ["0%", "Karbohidrat"])


class TextLayer(unittest.TestCase):
    ARTWORK = Path(__file__).resolve().parents[2] / "docs/ocr/artwork/label-v2.ai"  # written by bench/artwork.py

    @unittest.skipUnless(ARTWORK.exists(), "run: python -m bench.artwork ../docs/ocr/artwork")
    def test_sideways_and_curved_text_is_not_taken_from_the_text_layer_as_loose_letters(self):
        from app import checks, ingest

        page = ingest.load(str(self.ARTWORK), 100)[0]
        texts = [w.text for w in page.words]
        master = {checks.norm(t) for t in checks.tokens(self.ARTWORK.with_suffix(".txt").read_text(encoding="utf-8"))}
        self.assertIn("Komposisi:", texts)
        self.assertEqual([t for w in texts for t in checks.tokens(w) if checks.norm(t) not in master], [])  # no "k", "eri", "ej"
        self.assertNotIn("PILIHAN", texts)  # the arc is left to OCR
