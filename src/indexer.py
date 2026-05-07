"""
indexer.py
==========
Information Retrieval – positional inverted index builder.

Iterates over the corpus (10 Arabic + 10 English plain-text files stored
under ``data/Arabic/`` and ``data/English/``), preprocesses each document
via :func:`preprocessor.process_text`, and constructs a positional inverted
index together with document-frequency (df) and collection statistics.

Index structure (in-memory)
---------------------------
::

    {
        "index": {
            "<term>": {
                "<docID>": [<pos0>, <pos1>, ...]   # 0-based token positions
            },
            ...
        },
        "df": {
            "<term>": <int>   # number of documents containing the term
        },
        "N": <int>,           # total number of indexed documents
        "doc_meta": {
            "<docID>": {
                "path":   "<relative path to source file>",
                "lang":   "en" | "ar",
                "length": <int>   # total tokens after preprocessing
            }
        }
    }

Public API
----------
    build_index(data_dir, output_path) -> dict
    load_index(path)                   -> dict

Usage (CLI)
-----------
    python src/indexer.py
    python src/indexer.py --data data --out index.json

Dependencies
------------
    Same virtual-environment as preprocessor.py (nltk already installed).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sibling-module import – works whether run as a script or imported as a
# package member (src/indexer.py → src/preprocessor.py).
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from preprocessor import process_text  # noqa: E402  (after sys.path tweak)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Corpus layout constants
# ---------------------------------------------------------------------------
_LANG_DIRS: dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
}

_FILE_RANGE = range(1, 11)          # files 1.txt … 10.txt
_ENCODING   = "utf-8"


# ===========================================================================
# Core helpers
# ===========================================================================

def _iter_corpus(data_dir: Path) -> list[tuple[str, str, Path]]:
    """
    Yield ``(doc_id, lang, file_path)`` tuples for every corpus file.

    Doc-ID convention:   ``"en_1"``, ``"en_2"``, … ``"ar_1"``, … ``"ar_10"``
    Iteration order:     English first (en_1 … en_10), Arabic second.
    """
    entries: list[tuple[str, str, Path]] = []
    for lang, subdir in _LANG_DIRS.items():
        lang_dir = data_dir / subdir
        if not lang_dir.is_dir():
            logger.warning("Expected directory not found: %s", lang_dir)
            continue
        for i in _FILE_RANGE:
            file_path = lang_dir / f"{i}.txt"
            if not file_path.is_file():
                logger.warning("Missing corpus file: %s", file_path)
                continue
            entries.append((f"{lang}_{i}", lang, file_path))
    return entries


def _index_document(
    doc_id:          str,
    lang:            str,
    file_path:       Path,
    index:           dict[str, dict[str, list[int]]],
    df:              dict[str, int],
    doc_meta:        dict[str, dict[str, Any]],
) -> None:
    """
    Read *file_path*, preprocess its text, and update *index*, *df*,
    and *doc_meta* in-place.

    Parameters
    ----------
    doc_id    : Unique document identifier (e.g. ``"en_3"``).
    lang      : Language code passed to :func:`process_text`.
    file_path : Absolute path to the source ``.txt`` file.
    index     : Positional inverted index being built.
    df        : Document-frequency counter being updated.
    doc_meta  : Per-document metadata being populated.
    """
    text = file_path.read_text(encoding=_ENCODING)
    stems: list[str] = process_text(text, lang)

    # Positional index – record each stem with its 0-based position
    seen_in_doc: set[str] = set()
    for position, term in enumerate(stems):
        index[term][doc_id].append(position)
        seen_in_doc.add(term)

    # df – increment once per document, regardless of term frequency
    for term in seen_in_doc:
        df[term] += 1

    doc_meta[doc_id] = {
        "path":   str(file_path.relative_to(file_path.parents[2])),
        "lang":   lang,
        "length": len(stems),
    }

    logger.info(
        "Indexed  %-8s  |  lang=%-2s  |  tokens=%d  |  unique_terms=%d",
        doc_id, lang, len(stems), len(seen_in_doc),
    )


# ===========================================================================
# Public API
# ===========================================================================

def build_index(
    data_dir:    str | Path = "data",
    output_path: str | Path = "index.json",
) -> dict[str, Any]:
    """
    Build a positional inverted index from the corpus and persist it as JSON.

    Parameters
    ----------
    data_dir : str | Path
        Root directory that contains ``English/`` and ``Arabic/``
        sub-folders.  Defaults to ``"data"`` (relative to CWD).
    output_path : str | Path
        Destination for the serialised index JSON file.
        Defaults to ``"index.json"`` in the CWD.

    Returns
    -------
    dict
        The complete index payload (same structure written to disk)::

            {
                "index":    { term: { docID: [positions] } },
                "df":       { term: int },
                "N":        int,
                "doc_meta": { docID: { path, lang, length } }
            }

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist.
    """
    data_dir    = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    # Use defaultdict during construction for ergonomic updates
    raw_index: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    df:       dict[str, int]              = defaultdict(int)
    doc_meta: dict[str, dict[str, Any]]   = {}

    corpus = _iter_corpus(data_dir)
    if not corpus:
        logger.error("No corpus files discovered under %s", data_dir)
        return {}

    logger.info("Starting indexing – %d documents found", len(corpus))

    for doc_id, lang, file_path in corpus:
        _index_document(doc_id, lang, file_path, raw_index, df, doc_meta)

    N = len(doc_meta)
    logger.info("Indexing complete – N=%d documents, vocabulary=%d terms",
                N, len(raw_index))

    # Convert defaultdicts to plain dicts for clean JSON serialisation
    payload: dict[str, Any] = {
        "N":        N,
        "df":       dict(df),
        "index":    {term: dict(postings) for term, postings in raw_index.items()},
        "doc_meta": doc_meta,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding=_ENCODING,
    )
    logger.info("Index saved → %s", output_path)

    return payload


def load_index(path: str | Path = "index.json") -> dict[str, Any]:
    """
    Load a previously saved index from *path*.

    Parameters
    ----------
    path : str | Path
        Path to the JSON file produced by :func:`build_index`.

    Returns
    -------
    dict
        Parsed index payload.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Index file not found: {path}")

    payload = json.loads(path.read_text(encoding=_ENCODING))
    logger.info(
        "Index loaded from %s  (N=%s, vocab=%s terms)",
        path, payload.get("N"), len(payload.get("index", {})),
    )
    return payload


# ===========================================================================
# CLI entry-point   (python src/indexer.py [--data <dir>] [--out <file>])
# ===========================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="indexer",
        description="Build a positional inverted index from the IR corpus.",
    )
    parser.add_argument(
        "--data",
        default="data",
        metavar="DIR",
        help="Root corpus directory (default: data)",
    )
    parser.add_argument(
        "--out",
        default="index.json",
        metavar="FILE",
        help="Output JSON path (default: index.json)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    # Reconfigure stdout to UTF-8 so Arabic terms display correctly on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()

    # Resolve paths relative to the project root (parent of src/)
    project_root = Path(__file__).resolve().parents[1]
    data_dir     = project_root / args.data
    output_path  = project_root / args.out

    payload = build_index(data_dir=data_dir, output_path=output_path)

    if payload:
        print("\n── Index summary ──────────────────────────────────")
        print(f"  Total documents (N)  : {payload['N']}")
        print(f"  Vocabulary size      : {len(payload['index'])} terms")
        print(f"  Output file          : {output_path}")

        # Show 3 sample entries from each language
        print("\n  Sample English postings:")
        en_terms = [t for t in payload["index"] if all(ord(c) < 128 for c in t)][:3]
        for t in en_terms:
            print(f"    {t!r:20s} → {dict(list(payload['index'][t].items())[:2])}")

        print("\n  Sample Arabic postings:")
        ar_terms = [t for t in payload["index"] if any(ord(c) > 127 for c in t)][:3]
        for t in ar_terms:
            print(f"    {t!r:20s} → {dict(list(payload['index'][t].items())[:2])}")
        print("───────────────────────────────────────────────────\n")
