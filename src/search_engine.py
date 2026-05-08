"""
search_engine.py
================
Information Retrieval – TF-IDF vector-space ranking engine.

Loads the positional inverted index produced by :mod:`indexer`, converts a
user query and every candidate document into TF-IDF vectors, then ranks
documents by cosine similarity.

Scoring formula
---------------
For each term *t* in document *d*:

    tf_weight(t, d)  = 1 + log10(tf(t, d))     if tf > 0, else 0
    idf_weight(t)    = log10(N / df(t))
    tfidf(t, d)      = tf_weight(t, d) * idf_weight(t)

Query vector entries use the same tf-idf scheme (tf = raw count of the
term inside the query after preprocessing).

Cosine similarity between query vector **q** and document vector **d**:

    sim(q, d) = (q · d) / (||q|| * ||d||)

Only documents that share at least one term with the query are scored
(sparse evaluation – no need to iterate all 20 docs for every term).

Public API
----------
    SearchEngine(index_path)          – load index from JSON file
    SearchEngine.search(query, lang,  – run a query
                        top_k) -> list[SearchResult]

    SearchResult                      – named tuple with fields:
        doc_id   : str   e.g. "en_3"
        score    : float cosine similarity (0–1)
        path     : str   relative corpus path
        lang     : str   "en" | "ar"

Usage (CLI)
-----------
    python src/search_engine.py "information retrieval" --lang en
    python src/search_engine.py "استرجاع المعلومات" --lang ar --top 5
    python src/search_engine.py --index path/to/index.json "query" --lang en

Dependencies
------------
    Same virtual-environment as preprocessor.py / indexer.py (nltk).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sibling-module import – works whether run as a script or as a package member
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from indexer import load_index          # noqa: E402
from preprocessor import process_text  # noqa: E402

# ---------------------------------------------------------------------------
# Logging  (inherits root config set by indexer; safe to call twice)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_ENCODING = "utf-8"


# ===========================================================================
# Result type
# ===========================================================================

@dataclass(frozen=True, order=True)
class SearchResult:
    """
    Immutable record for a single ranked result.

    Fields are ordered so that ``sorted(..., reverse=True)`` sorts by score.
    """
    score:  float   # cosine similarity – primary sort key
    doc_id: str     # e.g. "en_3"
    path:   str     # relative corpus path stored in doc_meta
    lang:   str     # "en" | "ar"

    def __str__(self) -> str:
        return f"[{self.score:.4f}]  {self.doc_id:<8s}  ({self.lang})  {self.path}"


# ===========================================================================
# TF-IDF helpers
# ===========================================================================

def _tf_weight(raw_tf: int) -> float:
    """Logarithmic TF: ``1 + log10(tf)`` for tf > 0, else 0."""
    return (1.0 + math.log10(raw_tf)) if raw_tf > 0 else 0.0


def _idf_weight(N: int, df: int) -> float:
    """Standard IDF: ``log10(N / df)``.  df=0 is guarded against division."""
    if df <= 0:
        return 0.0
    return math.log10(N / df)


def _build_tfidf_vector(
    term_counts: dict[str, int],
    df_map:      dict[str, int],
    N:           int,
) -> dict[str, float]:
    """
    Convert a ``{term: raw_count}`` mapping into a TF-IDF weight vector.

    Parameters
    ----------
    term_counts : dict[str, int]
        Raw term frequencies (from a query or from positional postings).
    df_map : dict[str, int]
        Global document-frequency table from the index.
    N : int
        Total number of indexed documents.

    Returns
    -------
    dict[str, float]
        Sparse TF-IDF vector ``{term: weight}``.
    """
    vector: dict[str, float] = {}
    for term, tf in term_counts.items():
        idf = _idf_weight(N, df_map.get(term, 0))
        weight = _tf_weight(tf) * idf
        if weight > 0.0:
            vector[term] = weight
    return vector


def _l2_norm(vector: dict[str, float]) -> float:
    """Euclidean (L2) norm of a sparse vector."""
    return math.sqrt(sum(w * w for w in vector.values()))


def _cosine_similarity(
    vec_a: dict[str, float],
    vec_b: dict[str, float],
) -> float:
    """
    Cosine similarity between two sparse vectors.

    Only iterates over the (shorter) query vector; documents missing a term
    contribute 0 to the dot product.
    """
    norm_a = _l2_norm(vec_a)
    norm_b = _l2_norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    # Dot product – use the smaller vector as the outer loop
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a

    dot = sum(w * vec_b.get(term, 0.0) for term, w in vec_a.items())
    return dot / (norm_a * norm_b)


# ===========================================================================
# SearchEngine
# ===========================================================================

class SearchEngine:
    """
    Vector-space search engine backed by a positional inverted index.

    Parameters
    ----------
    index_path : str | Path
        Path to the ``index.json`` file produced by :func:`indexer.build_index`.
        Defaults to ``"index.json"`` relative to the project root.

    Attributes
    ----------
    N        : int                              – total indexed documents
    index    : dict[term → dict[docID → [pos]]] – positional index
    df       : dict[term → int]                 – document frequencies
    doc_meta : dict[docID → {path, lang, length}]

    Examples
    --------
    >>> engine = SearchEngine()
    >>> results = engine.search("information retrieval", lang="en", top_k=5)
    >>> for r in results:
    ...     print(r)
    """

    def __init__(self, index_path: str | Path = "index.json") -> None:
        payload        = load_index(index_path)
        self.N:        int                              = payload["N"]
        self.index:    dict[str, dict[str, list[int]]] = payload["index"]
        self.df:       dict[str, int]                  = payload["df"]
        self.doc_meta: dict[str, dict[str, Any]]       = payload["doc_meta"]
        logger.info(
            "SearchEngine ready – N=%d docs, vocab=%d terms",
            self.N, len(self.index),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_vector(self, query: str, lang: str) -> dict[str, float]:
        """
        Preprocess *query* and produce its TF-IDF weight vector.

        Raw TF for the query is simply the count of each term in the
        preprocessed token list.
        """
        terms = process_text(query, lang)
        if not terms:
            return {}

        # Count raw term frequencies in the query
        raw_tf: dict[str, int] = {}
        for t in terms:
            raw_tf[t] = raw_tf.get(t, 0) + 1

        return _build_tfidf_vector(raw_tf, self.df, self.N)

    def _candidate_docs(self, query_terms: set[str]) -> set[str]:
        """
        Return the set of document IDs that contain *at least one* query term.

        This avoids scoring every document in the collection.
        """
        candidates: set[str] = set()
        for term in query_terms:
            if term in self.index:
                candidates.update(self.index[term].keys())
        return candidates

    def _doc_vector(self, doc_id: str, query_terms: set[str]) -> dict[str, float]:
        """
        Build the TF-IDF vector for *doc_id*, restricted to *query_terms*
        (projection – only the dimensions relevant to this query matter for
        cosine similarity computation).

        TF is derived from the length of the positional postings list.
        """
        raw_tf: dict[str, int] = {}
        for term in query_terms:
            postings = self.index.get(term, {})
            if doc_id in postings:
                raw_tf[term] = len(postings[doc_id])   # tf = number of occurrences

        return _build_tfidf_vector(raw_tf, self.df, self.N)

    # ------------------------------------------------------------------
    # Public search method
    # ------------------------------------------------------------------

    def search(
        self,
        query:  str,
        lang:   str,
        top_k:  int = 10,
    ) -> list[SearchResult]:
        """
        Rank documents against *query* using TF-IDF cosine similarity.

        Parameters
        ----------
        query : str
            Raw user query (UTF-8 text in the target language).
        lang : str
            Language code – ``"en"`` or ``"ar"`` – passed to the preprocessor.
        top_k : int
            Maximum number of results to return (default: 10).

        Returns
        -------
        list[SearchResult]
            Results sorted by descending cosine similarity score.
            Empty list if the query yields no indexed terms.

        Raises
        ------
        ValueError
            If *lang* is not ``"en"`` or ``"ar"``.
        """
        lang = lang.strip().lower()
        if lang not in ("en", "ar"):
            raise ValueError(
                f"Unsupported language {lang!r}.  Use 'en' or 'ar'."
            )

        # Step 1 – Build query TF-IDF vector
        q_vec = self._query_vector(query, lang)
        if not q_vec:
            logger.warning("Query produced no indexable terms after preprocessing.")
            return []

        query_terms = set(q_vec.keys())
        logger.info(
            "Query terms (%s): %s", lang, ", ".join(sorted(query_terms))
        )

        # Step 2 – Retrieve candidate documents (sparse evaluation)
        candidates = self._candidate_docs(query_terms)
        logger.info("Candidate documents: %d", len(candidates))

        if not candidates:
            return []

        # Step 3 – Score each candidate via cosine similarity
        results: list[SearchResult] = []
        for doc_id in candidates:
            d_vec  = self._doc_vector(doc_id, query_terms)
            score  = _cosine_similarity(q_vec, d_vec)
            if score > 0.0:
                meta = self.doc_meta.get(doc_id, {})
                results.append(SearchResult(
                    score  = score,
                    doc_id = doc_id,
                    path   = meta.get("path", ""),
                    lang   = meta.get("lang", lang),
                ))

        # Step 4 – Sort descending by score, truncate to top_k
        results.sort(reverse=True)
        return results[:top_k]


# ===========================================================================
# CLI entry-point
# ===========================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="search_engine",
        description="TF-IDF cosine-similarity search over the IR corpus.",
    )
    parser.add_argument(
        "query",
        help="Search query (surround multi-word queries with quotes).",
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=["en", "ar"],
        help="Query language: 'en' (default) or 'ar'.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="K",
        help="Number of results to display (default: 5).",
    )
    parser.add_argument(
        "--index",
        default=None,
        metavar="FILE",
        help="Path to index.json (default: <project_root>/index.json).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    # Reconfigure stdout to UTF-8 so Arabic text displays on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()

    project_root = Path(__file__).resolve().parents[1]
    index_path   = Path(args.index) if args.index else project_root / "index.json"

    engine  = SearchEngine(index_path=index_path)
    results = engine.search(query=args.query, lang=args.lang, top_k=args.top)

    print(f"\n── Results for: {args.query!r}  (lang={args.lang}, top={args.top}) ──")
    if not results:
        print("  No matching documents found.")
    else:
        for rank, result in enumerate(results, start=1):
            print(f"  {rank}. {result}")
    print("─" * 60 + "\n")
