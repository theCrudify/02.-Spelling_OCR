"""Deterministic checks. Pure functions over words (+ references) so they are testable without OCR or a database.
No generative AI here on purpose: every finding must be exact and explainable for regulated labels."""

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .ingest import Word

# Words and numbers (keeps 0,5 / 1.5 together). Units stay separate tokens: "500 ml" -> 500, ml.
TOKEN = re.compile(r"[^\W_]+(?:[.,][0-9]+)?", re.UNICODE)
# Scripts written without spaces: compare per character.
CHAR_SCRIPT = re.compile(r"[฀-๿぀-ヿ㐀-鿿가-힯]")


@dataclass
class Finding:
    check_type: str  # code of a check_types master row
    message: str
    severity: str | None = None  # None = severity of the check type (master); set only when the rule carries its own
    page_no: int | None = None
    bbox: tuple | None = None
    found: str | None = None
    expected: str | None = None
    lang: str | None = None


def norm(t: str) -> str:
    return unicodedata.normalize("NFKC", t).casefold()


def tokens(text: str) -> list[str]:
    spaced = CHAR_SCRIPT.sub(lambda m: f" {m.group()} ", unicodedata.normalize("NFKC", text))  # each CJK/Thai char is a token
    return [m.group() for m in TOKEN.finditer(spaced)]


def _at(w: Word, page_no: int) -> dict:
    return {"page_no": page_no, "bbox": w.bbox, "lang": w.lang}


# ---------------------------------------------------------------- 1. master text diff (most important)
def master_diff(pages: list[list[Word]], masters: dict[str, str], similarity: float = 0.6) -> list[Finding]:
    """Bag-of-words compare per label vs approved master copy (order-insensitive, so panel layout does not matter).
    shortcut: a word-order swap inside a sentence is not detected; add a sequence diff per text block if that is needed."""
    if not masters:
        return []
    master_seq = [(norm(t), t) for text in masters.values() for t in tokens(text)]
    in_master = {n for n, _ in master_seq}
    occurrences: dict[str, list[tuple[Word, int]]] = {}
    for page_no, words in enumerate(pages, 1):
        for w in words:
            for t in tokens(w.text):
                # Text in a language without master copy is checked by spelling only, except words the master does
                # contain: an English line on an Indonesian label ("Otsuka Pharmaceutical Co., Ltd., Japan") is
                # detected as English, yet it is in the Indonesian master and must not be reported missing.
                if w.lang is None or w.lang in masters or norm(t) in in_master:
                    occurrences.setdefault(norm(t), []).append((w, page_no))
    label_count = Counter({k: len(v) for k, v in occurrences.items()})
    master_count = Counter(n for n, _ in master_seq)
    extra = label_count - master_count
    missing = master_count - label_count

    findings: list[Finding] = []
    unmatched = Counter(missing)
    for tok, n in sorted(extra.items(), key=lambda kv: occurrences[kv[0]][-1][1]):
        best = difflib.get_close_matches(tok, [m for m, c in unmatched.items() if c > 0], n=1, cutoff=similarity)
        expected = best[0] if best else None
        if expected:
            unmatched[expected] -= 1
        for w, page_no in occurrences[tok][-n:]:
            msg = f'"{w.text}" tidak sesuai teks master' + (f', seharusnya "{expected}"' if expected else " (kata tambahan)")
            findings.append(Finding("master_diff", msg, found=w.text, expected=expected, **_at(w, page_no)))

    # Missing master words, grouped into phrases in master order ("Kocok sebelum diminum").
    left = +unmatched
    phrase: list[str] = []
    for n_tok, original in master_seq + [("", "")]:
        if n_tok and left[n_tok] > 0:
            left[n_tok] -= 1
            phrase.append(original)
        elif phrase:
            text = " ".join(phrase)
            findings.append(Finding("master_missing", f'Teks master tidak ditemukan di label: "{text}"', expected=text))
            phrase = []
    return findings


def missing_masters(languages: list[str], masters: dict[str, str]) -> list[Finding]:
    return [Finding("master_absent", f"Belum ada teks master untuk bahasa '{lang}': hanya dicek typo & glossary", lang=lang) for lang in languages if lang not in masters]


# ---------------------------------------------------------------- 2. spelling + 3. glossary
def spelling(pages: list[list[Word]], dictionaries: dict, glossary: set[str], vocabulary: set[str], skip: set[int], min_length: int = 2) -> list[Finding]:
    """Hunspell per detected language. Skips glossary terms, master-copy words, numbers and words already flagged (`skip` = id(word))."""
    findings = []
    for page_no, words in enumerate(pages, 1):
        for w in words:
            d = dictionaries.get(w.lang)
            if d is None or id(w) in skip:
                continue
            for t in tokens(w.text):
                n = norm(t)
                if len(t) < min_length or any(ch.isdigit() for ch in t) or n in glossary or n in vocabulary or CHAR_SCRIPT.search(t):
                    continue
                if d.lookup(t) or d.lookup(t.lower()):
                    continue
                suggestion = next(iter(d.suggest(t)), None)
                msg = f'Kemungkinan salah ketik: "{t}"' + (f', saran "{suggestion}"' if suggestion else "")
                findings.append(Finding("spelling", msg, found=t, expected=suggestion, **_at(w, page_no)))
    return findings


def glossary_near_miss(pages: list[list[Word]], glossary_terms: list[str], skip: set[int], vocabulary: set[str] = frozenset(), dictionaries: dict | None = None, similarity: float = 0.8, min_length: int = 4) -> list[Finding]:
    """A word very close to (but not exactly) a company term, e.g. "Pocary" vs "Pocari". Real words (master copy or dictionary) are fine."""
    terms = {norm(t): t for t in glossary_terms}
    findings = []
    for page_no, words in enumerate(pages, 1):
        for w in words:
            if id(w) in skip:
                continue
            for t in tokens(w.text):
                n = norm(t)
                if len(n) < min_length or n in terms or n in vocabulary or any(ch.isdigit() for ch in n):
                    continue
                d = (dictionaries or {}).get(w.lang)
                if d is not None and d.lookup(t):
                    continue
                close = difflib.get_close_matches(n, terms.keys(), n=1, cutoff=similarity)
                if close:
                    term = terms[close[0]]
                    findings.append(Finding("glossary", f'"{t}" mirip istilah glossary "{term}"', found=t, expected=term, **_at(w, page_no)))
    return findings


# ---------------------------------------------------------------- 4. mandatory elements per country
def rules(pages: list[list[Word]], country_rules: list[dict]) -> list[Finding]:
    text = "\n".join(" ".join(w.text for w in words) for words in pages)
    return [
        Finding("rule", f'Elemen wajib tidak ditemukan: {r["name"]}', severity=r["severity"], expected=r["pattern"])
        for r in country_rules
        if not re.search(r["pattern"], text, re.IGNORECASE | re.UNICODE)
    ]


# ---------------------------------------------------------------- 6. low OCR confidence
def low_confidence(pages: list[list[Word]], threshold: float) -> list[Finding]:
    return [
        Finding("low_confidence", f'OCR kurang yakin ({w.conf:.0%}): "{w.text}". Cek manual.', found=w.text, **_at(w, page_no))
        for page_no, words in enumerate(pages, 1)
        for w in words
        if w.source == "ocr" and w.conf < threshold
    ]


# Thresholds are settings (database); these are only the fallbacks when a row is missing.
DEFAULTS = {"master_diff_similarity": 0.6, "glossary_similarity": 0.8, "glossary_min_length": 4, "spelling_min_length": 2, "ocr_min_confidence": 0.8}


def finalize(findings: list[Finding], types: dict[str, str]) -> list[Finding]:
    """`types` = active check types (code -> severity) from the check_types master: a check that is switched off (or
    deleted) produces nothing, and each finding takes the severity configured for its type."""
    out = []
    for f in findings:
        if f.check_type in types:
            f.severity = f.severity or types[f.check_type]
            out.append(f)
    return out


def run_all(pages, *, languages, masters, dictionaries, glossary_terms, country_rules, types: dict[str, str], cfg: dict | None = None) -> list[Finding]:
    c = {**DEFAULTS, **(cfg or {})}
    glossary = {norm(t) for t in glossary_terms}
    vocabulary = {norm(t) for text in masters.values() for t in tokens(text)}
    found = master_diff(pages, masters, c["master_diff_similarity"]) + missing_masters(languages, masters)
    flagged_bboxes = {f.bbox for f in found if f.bbox}
    skip = {id(w) for words in pages for w in words if w.bbox in flagged_bboxes}  # master diff already explains these words
    found += spelling(pages, dictionaries, glossary, vocabulary, skip, int(c["spelling_min_length"]))
    skip |= {id(w) for words in pages for w in words if w.bbox in {f.bbox for f in found if f.bbox}}
    found += glossary_near_miss(pages, glossary_terms, skip, vocabulary, dictionaries, c["glossary_similarity"], int(c["glossary_min_length"]))
    found += rules(pages, country_rules)
    found += low_confidence(pages, c["ocr_min_confidence"])
    return finalize(found, types)
