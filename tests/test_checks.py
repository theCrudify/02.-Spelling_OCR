"""Checks are pure functions: run with  uv run python -m unittest discover tests"""

import unittest

from app import checks
from app.ingest import Word


def words(text: str, lang="id", y=0) -> list[Word]:
    return [Word(t, (i * 10, y, i * 10 + 9, y + 9), 1.0, "text_layer", lang) for i, t in enumerate(text.split())]


class FakeDict:
    def __init__(self, vocab):
        self.vocab = set(vocab)

    def lookup(self, w):
        return w.lower() in self.vocab

    def suggest(self, w):
        return iter(["elektrolit"]) if w.startswith("elekt") else iter([])


MASTER = {"id": "Minuman isotonik pengganti elektrolit. Kocok sebelum diminum. Isi bersih 500 ml."}


class MasterDiff(unittest.TestCase):
    def test_typo_number_and_missing_phrase(self):
        page = words("Minuman isotonik pengganti elektorlit. Isi bersih 50 ml.")
        f = checks.master_diff([page], MASTER)
        diffs = {(x.found, x.expected) for x in f if x.check_type == "master_diff"}
        self.assertIn(("elektorlit.", "elektrolit"), diffs)
        self.assertIn(("50", "500"), diffs)
        missing = [x.expected for x in f if x.check_type == "master_missing"]
        self.assertEqual(missing, ["Kocok sebelum diminum"])

    def test_identical_text_in_other_layout_is_clean(self):
        page = words("Isi bersih 500 ml.", y=0) + words("Minuman isotonik pengganti elektrolit. Kocok sebelum diminum.", y=50)
        self.assertEqual(checks.master_diff([page], MASTER), [])

    def test_char_scripts_compare_per_character(self):
        self.assertEqual(checks.tokens("净含量500ml"), ["净", "含", "量", "500ml"])


class OtherChecks(unittest.TestCase):
    def test_spelling_skips_glossary_master_and_numbers(self):
        page = words("Pocari elektorlit 500 Sukabumi minuman")
        f = checks.spelling([page], {"id": FakeDict(["minuman"])}, {"pocari"}, {"sukabumi"}, set())
        self.assertEqual([(x.found, x.expected) for x in f], [("elektorlit", "elektrolit")])

    def test_glossary_near_miss(self):
        f = checks.glossary_near_miss([words("Pocary Sweat")], ["Pocari", "Sweat"], set())
        self.assertEqual([(x.found, x.expected) for x in f], [("Pocary", "Pocari")])
        # a real word from the master copy is not a near miss ("Isotonic" vs glossary "isotonik")
        self.assertEqual(checks.glossary_near_miss([words("Isotonic")], ["isotonik"], set(), {"isotonic"}), [])

    def test_country_rule_missing(self):
        rules = [{"name": "BPOM", "pattern": r"BPOM\s*RI\s*MD\s*\d{12}", "severity": "high"}, {"name": "Halal", "pattern": r"\bhalal\b", "severity": "high"}]
        f = checks.rules([words("BPOM RI MD 265210001234")], rules)
        self.assertEqual([x.message for x in f], ["Elemen wajib tidak ditemukan: Halal"])

    def test_low_confidence_only_for_ocr(self):
        page = [Word("abc", (0, 0, 1, 1), 0.5, "ocr"), Word("def", (0, 0, 1, 1), 0.5, "text_layer")]
        self.assertEqual([x.found for x in checks.low_confidence([page], 0.8)], ["abc"])


class Masters(unittest.TestCase):
    """Severity and on/off come from the check_types master, thresholds from settings."""

    PAGE = None

    def run_all(self, types, cfg=None):
        page = words("Minuman isotonik pengganti elektorlit. Isi bersih 50 ml.")
        return checks.run_all([page], languages=["id"], masters=MASTER, dictionaries={}, glossary_terms=[], country_rules=[{"name": "Halal", "pattern": "halal", "severity": "low"}], types=types, cfg=cfg)

    def test_severity_comes_from_the_master_and_rules_keep_their_own(self):
        f = self.run_all({"master_diff": "medium", "master_missing": "high", "rule": "high"})
        self.assertEqual({x.severity for x in f if x.check_type == "master_diff"}, {"medium"})
        self.assertEqual([x.severity for x in f if x.check_type == "rule"], ["low"])

    def test_inactive_check_type_produces_nothing(self):
        f = self.run_all({"rule": "high"})
        self.assertEqual({x.check_type for x in f}, {"rule"})

    def test_similarity_threshold_is_a_setting(self):
        strict = self.run_all({"master_diff": "high"}, {"master_diff_similarity": 0.99})
        self.assertTrue(all(x.expected is None for x in strict))


if __name__ == "__main__":
    unittest.main()
