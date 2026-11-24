"""OCR worker: takes queued label versions from Postgres, extracts text (text layer + local OCR), runs the checks,
writes pages / text blocks / findings back. Nothing leaves the machine.

    uv run --env-file .env python -m app.worker            # run forever
    uv run --env-file .env python -m app.worker --once     # process the queue, then exit
    uv run --env-file .env python -m app.worker --download-models
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path

import psycopg
from lingua import IsoCode639_1, LanguageDetectorBuilder
from psycopg.rows import dict_row
from spylls.hunspell import Dictionary

from . import checks, ingest, ocr

log = logging.getLogger("worker")
STORAGE = Path(os.environ.get("STORAGE_DIR", "../storage")).resolve()
DICT_DIR = Path(__file__).resolve().parent.parent / "dictionaries"
_dictionaries: dict[str, Dictionary | None] = {}
_detectors: dict[tuple, object] = {}


def dictionary(name: str | None):
    if name and name not in _dictionaries:
        path = DICT_DIR / name
        _dictionaries[name] = Dictionary.from_files(str(path)) if path.with_suffix(".dic").exists() else None
        if _dictionaries[name] is None:
            log.warning("dictionary %s not found in %s (run scripts/setup_offline.sh); spelling skipped", name, DICT_DIR)
    return _dictionaries.get(name) if name else None


def detector(codes: tuple[str, ...]):
    """lingua detector limited to the languages configured in the app (far more reliable than open detection)."""
    if codes not in _detectors:
        isos = [getattr(IsoCode639_1, c.upper()) for c in codes if hasattr(IsoCode639_1, c.upper())]
        _detectors[codes] = LanguageDetectorBuilder.from_iso_codes_639_1(*isos).build() if len(isos) > 1 else None
    return _detectors[codes]


def lines(words: list[ingest.Word]) -> list[list[ingest.Word]]:
    """Reading order: group words whose vertical centres overlap into lines, lines top to bottom, words left to right."""
    out: list[list[ingest.Word]] = []
    for w in sorted(words, key=lambda w: (w.bbox[1] + w.bbox[3]) / 2):
        cy = (w.bbox[1] + w.bbox[3]) / 2
        if out and out[-1][0].bbox[1] <= cy <= out[-1][0].bbox[3]:
            out[-1].append(w)
        else:
            out.append([w])
    return [sorted(line, key=lambda w: w.bbox[0]) for line in out]


LANGUAGE_DEFAULTS = {"language_min_letters": 20, "language_min_confidence": 0.7, "language_max_declared_confidence": 0.15}


def assign_languages(words: list[ingest.Word], registered: list[str], known: dict[str, str], page_no: int, cfg: dict | None = None) -> list[checks.Finding]:
    """Sets word.lang per text line. `registered` = languages declared on the label, `known` = every active language
    (code -> name). A line that is clearly in a known language the label does not declare gets that language (so it is
    spell-checked correctly) and a 'language' finding."""
    c = {**LANGUAGE_DEFAULTS, **(cfg or {})}  # thresholds are settings; the defaults only cover a missing row
    det = detector(tuple(sorted(known)))
    findings = []
    for line in lines(words):
        text = " ".join(w.text for w in line)
        lang = registered[0]
        if det is not None:
            conf = {v.language.iso_code_639_1.name.lower(): v.value for v in det.compute_language_confidence_values(text)}
            best = max(registered, key=lambda c: conf.get(c, 0.0))
            top = max(conf, key=conf.get)
            letters = sum(ch.isalpha() for ch in text)
            # Clear = long enough to judge, confident, and the label's own languages are practically ruled out
            # (so close pairs like Indonesian / Malay never trigger it when one of them is declared).
            if top not in registered and letters >= c["language_min_letters"] and conf[top] >= c["language_min_confidence"] and sum(conf.get(r, 0.0) for r in registered) < c["language_max_declared_confidence"]:
                lang = top
                box = (min(w.bbox[0] for w in line), min(w.bbox[1] for w in line), max(w.bbox[2] for w in line), max(w.bbox[3] for w in line))
                findings.append(checks.Finding("language", f"Teks terdeteksi berbahasa {known[top]}, yang tidak terdaftar di label ini: \"{text[:80]}\"", page_no=page_no, bbox=box, found=text[:200], lang=top))
            elif conf.get(best, 0.0) > 0:
                lang = best
        for w in line:
            w.lang = lang
    return findings


# ---------------------------------------------------------------- database
def connect():
    return psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row, autocommit=True)


def claim(conn):
    with conn.transaction():
        job = conn.execute(
            """select v.id, v.label_id, v.version, f.path, l.languages, l.country_id, l.label_type, l.product_id, p.brand
               from label_versions v
               join files f on f.id = v.file_id
               join labels l on l.id = v.label_id
               join products p on p.id = l.product_id
               where v.ocr_status = 'queued'
               order by v.id
               for update of v skip locked
               limit 1"""
        ).fetchone()
        if job:
            conn.execute("update label_versions set ocr_status = 'running', ocr_started_at = now(), ocr_error = null where id = %s", (job["id"],))
    return job


def settings(conn) -> dict:
    """All settings as a dict; numbers and booleans are converted, so every threshold of the worker is data."""
    out = {}
    for r in conn.execute("select key, value, value_type from settings").fetchall():
        try:
            out[r["key"]] = float(r["value"]) if r["value_type"] == "number" else r["value"] == "true" if r["value_type"] == "boolean" else r["value"]
        except ValueError:
            pass  # a malformed number falls back to the code default
    return out


def process(conn, job):
    codes = job["languages"]
    langs = {r["code"]: r for r in conn.execute("select code, name, ocr_lang, dictionary from languages where is_active or code = any(%s)", (codes,)).fetchall()}
    known = {c: r["name"] for c, r in langs.items()}
    masters = {
        r["language"]: r["text"]
        for r in conn.execute(
            "select language, text from master_texts where product_id = %s and country_id = %s and language = any(%s) and deleted_at is null",
            (job["product_id"], job["country_id"], codes),
        ).fetchall()
    }
    glossary_terms = [
        r["term"]
        for r in conn.execute(
            "select term from glossary_terms where is_active and (language is null or language = any(%s)) and (brand is null or brand = %s)",
            (codes, job["brand"]),
        ).fetchall()
    ]
    country_rules = conn.execute(
        "select name, pattern, severity from country_rules where is_active and country_id = %s and (label_type is null or label_type = %s)",
        (job["country_id"], job["label_type"]),
    ).fetchall()
    cfg = settings(conn)
    types = {r["code"]: r["severity"] for r in conn.execute("select code, severity from check_types where is_active").fetchall()}
    open_code = conn.execute("select code from finding_resolutions where kind = 'open'").fetchone()["code"]
    dpi = int(cfg.get("render_dpi", 200))

    t0 = time.time()
    pages = ingest.load(str(STORAGE / job["path"]), dpi)
    ocr_langs = [langs[c]["ocr_lang"] for c in codes if c in langs]
    page_words = []
    language_findings = []
    for n, p in enumerate(pages, 1):
        words = ocr.merge(p.words, ocr.read(p.image, ocr_langs, cfg.get("ocr_rotations", True), cfg.get("ocr_detection_model", ocr.DETECTION)))
        language_findings += assign_languages(words, codes, known, n, cfg)
        page_words.append([w for line in lines(words) for w in line])
    used = {w.lang for words in page_words for w in words} | set(codes)
    findings = checks.finalize(language_findings, types) + checks.run_all(
        page_words,
        languages=codes,
        masters=masters,
        dictionaries={c: dictionary(langs[c]["dictionary"]) for c in used if c in langs},
        glossary_terms=glossary_terms,
        country_rules=country_rules,
        types=types,
        cfg=cfg,
    )

    out_dir = STORAGE / "pages" / str(job["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with conn.transaction():
        # Re-check replaces the previous results of this version.
        for t in ("findings", "text_blocks", "label_pages"):
            conn.execute(f"delete from {t} where version_id = %s", (job["id"],))
        for n, (p, words) in enumerate(zip(pages, page_words), 1):
            rel = f"pages/{job['id']}/{n}.png"
            p.image.save(STORAGE / rel)
            conn.execute("insert into label_pages (version_id, page_no, image_path, width, height) values (%s, %s, %s, %s, %s)", (job["id"], n, rel, p.image.width, p.image.height))
            with conn.cursor() as cur:
                cur.executemany(
                    "insert into text_blocks (version_id, page_no, seq, text, lang, bbox, confidence, source) values (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [(job["id"], n, i, w.text, w.lang, list(w.bbox), w.conf, w.source) for i, w in enumerate(words)],
                )
        with conn.cursor() as cur:
            cur.executemany(
                "insert into findings (version_id, check_type, severity, page_no, bbox, found, expected, message, lang, resolution) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [(job["id"], f.check_type, f.severity, f.page_no, list(f.bbox) if f.bbox else None, f.found, f.expected, f.message, f.lang, open_code) for f in findings],
            )
        conn.execute("update label_versions set ocr_status = 'done', ocr_finished_at = now() where id = %s", (job["id"],))
        to_review(conn, job)
    log.info("version %s: %d page(s), %d words, %d findings in %.1fs", job["id"], len(pages), sum(map(len, page_words)), len(findings), time.time() - t0)


def to_review(conn, job):
    """Hand the label to the checker, unless a newer version was uploaded meanwhile."""
    conn.execute(
        """update labels set status = (select code from workflow_states where kind = 'review'), updated_at = now()
           where id = %s and current_version = %s and status = (select code from workflow_states where kind = 'processing')""",
        (job["label_id"], job["version"]),
    )


def fail(conn, job, err: Exception):
    log.error("version %s failed: %s", job["id"], err)
    with conn.transaction():
        conn.execute("update label_versions set ocr_status = 'failed', ocr_error = %s, ocr_finished_at = now() where id = %s", (f"{type(err).__name__}: {err}"[:1000], job["id"]))
        to_review(conn, job)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = connect()
    if "--download-models" in sys.argv:
        langs = [r["ocr_lang"] for r in conn.execute("select distinct ocr_lang from languages where is_active").fetchall()]
        for rec in ocr.models_for(langs):
            ocr.engine(rec, settings(conn).get("ocr_detection_model", ocr.DETECTION))  # creating the engine downloads its models once
            log.info("model ready: %s", rec)
        return
    log.info("worker started, storage %s", STORAGE)
    while True:
        # A job left 'running' by a worker that died is retried after the timeout. Safe with several workers:
        # nobody resets another worker's live job, and a repeated job replaces its own results.
        conn.execute(
            "update label_versions set ocr_status = 'queued' where ocr_status = 'running' and ocr_started_at < now() - make_interval(mins => %s)",
            (int(settings(conn).get("ocr_stale_minutes", 30)),),
        )
        job = claim(conn)
        if not job:
            if "--once" in sys.argv:
                return
            time.sleep(settings(conn).get("worker_poll_seconds", 2))
            continue
        try:
            process(conn, job)
        except Exception as e:  # a broken file must not stop the queue
            traceback.print_exc()
            fail(conn, job, e)


if __name__ == "__main__":
    main()
