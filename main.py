"""
main.py
=======
Information Retrieval Project – interactive terminal entry point.

On startup:
    • Ensures ``index.json`` exists; builds it from ``data/`` if not.
    • Launches an interactive search loop supporting English and Arabic queries.
    • Displays the top-5 ranked results with score, doc-ID, and file path.

Run from the project root:
    python main.py
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# UTF-8 stdout – must happen before ANY Arabic text is printed
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ---------------------------------------------------------------------------
# Silence the noisy INFO logger produced by indexer / search_engine so the
# terminal output stays clean for the user.  Change to INFO to debug.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Path setup – add src/ so that sibling modules are importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR      = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from indexer       import build_index   # type: ignore[import]  # noqa: E402
from search_engine import SearchEngine  # type: ignore[import]  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_INDEX_PATH = _PROJECT_ROOT / "index.json"
_DATA_DIR   = _PROJECT_ROOT / "data"
_TOP_K      = 5

_LANG_MAP: dict[str, str] = {
    "1": "en",
    "2": "ar",
}

_LANG_LABEL: dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
}

# ===========================================================================
# Terminal UI helpers
# ===========================================================================

_W = 62  # total width of bordered boxes


def _banner() -> None:
    """Print the application welcome banner."""
    inner = "  IR Search Engine  --  TF-IDF Cosine Similarity  "
    print()
    print("+" + "=" * _W + "+")
    print("|" + inner.center(_W) + "|")
    print("|" + "  Languages: English  |  Arabic".center(_W) + "|")
    print("+" + "=" * _W + "+")
    print()


def _section(title: str) -> None:
    print("┌" + "─" * _W + "┐")
    print("│  " + title.ljust(_W - 2) + "│")
    print("└" + "─" * _W + "┘")


def _divider() -> None:
    print("─" * (_W + 2))


def _print_results(results: list, lang: str) -> None:
    """Render a numbered results table inside a box."""
    label = _LANG_LABEL.get(lang, lang)
    print()
    print("┌" + "─" * _W + "┐")
    print("│" + f"  Top {_TOP_K} results  [{label}]".ljust(_W) + "│")
    print("├" + "─" * _W + "┤")

    if not results:
        msg = "  ✗  No matching documents found for your query."
        print("│" + msg.ljust(_W) + "│")
        hint = "  Tip: try different keywords or switch language."
        print("│" + hint.ljust(_W) + "│")
    else:
        for rank, r in enumerate(results, start=1):
            # Line 1 – rank + score + doc_id
            score_line = f"  {rank}. Score: {r.score:.4f}   Doc ID: {r.doc_id}"
            print("│" + score_line.ljust(_W) + "│")
            # Line 2 – path (truncated to fit)
            path_str   = str(r.path)
            if len(path_str) > _W - 9:
                path_str = "…" + path_str[-((_W - 10)):]
            path_line  = f"     Path : {path_str}"
            print("│" + path_line.ljust(_W) + "│")
            if rank < len(results):
                print("│" + "  " + "·" * (_W - 4) + "  " + "│")

    print("└" + "─" * _W + "┘")
    print()


def _choose_language() -> str | None:
    """
    Prompt the user to pick a search language.

    Returns ``\"en\"``, ``\"ar\"``, or ``None`` if the user chose to exit.
    """
    _section("Select Search Language")
    print("    [1]  English")
    print("    [2]  Arabic")
    print("    [0]  Exit")
    print()
    choice = input("  ›  Enter choice: ").strip()

    if choice == "0":
        return None
    if choice in _LANG_MAP:
        return _LANG_MAP[choice]

    print("\n  ⚠  Invalid choice – please enter 1, 2, or 0.\n")
    return _choose_language()  # simple recursion for bad input


# ===========================================================================
# Index bootstrap
# ===========================================================================

def _ensure_index() -> bool:
    """
    Guarantee ``index.json`` exists.

    Returns ``True`` when ready, ``False`` when the data directory is absent.
    """
    if _INDEX_PATH.is_file():
        return True

    print()
    print("  ℹ  index.json not found – building index from data/ …")
    _divider()

    if not _DATA_DIR.is_dir():
        print()
        print("  ✗  ERROR: The 'data/' directory was not found.")
        print(f"     Expected location: {_DATA_DIR}")
        print()
        print("  Please place your corpus files under:")
        print("     data/English/1.txt … 10.txt")
        print("     data/Arabic/1.txt  … 10.txt")
        print()
        return False

    try:
        # Re-enable INFO just for the build step so progress is visible
        logging.getLogger().setLevel(logging.INFO)
        build_index(data_dir=_DATA_DIR, output_path=_INDEX_PATH)
        logging.getLogger().setLevel(logging.WARNING)
        print()
        print(f"  ✓  Index built and saved → {_INDEX_PATH.name}")
        _divider()
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ✗  Failed to build index: {exc}\n")
        return False

    return True


# ===========================================================================
# Main interactive loop
# ===========================================================================

def main() -> None:
    _banner()

    # ── Step 1: ensure index exists ─────────────────────────────────────────
    if not _ensure_index():
        sys.exit(1)

    # ── Step 2: load the search engine ──────────────────────────────────────
    try:
        engine = SearchEngine(index_path=_INDEX_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ✗  Could not load the search engine: {exc}\n")
        sys.exit(1)

    print(f"  ✓  Engine ready  –  {engine.N} documents indexed\n")

    # ── Step 3: interactive search loop ─────────────────────────────────────
    while True:
        # --- Language selection ---
        lang = _choose_language()
        if lang is None:
            _divider()
            print("  Goodbye!")
            _divider()
            print()
            break

        lang_label = _LANG_LABEL[lang]

        # --- Query input ---
        _section(f"Enter Query  [{lang_label}]")
        print("    (leave blank to go back to language selection)")
        print()
        query = input("  ›  Query: ").strip()

        if not query:
            print("\n  ↩  Returning to language selection …\n")
            continue

        # --- Search ---
        try:
            results = engine.search(query=query, lang=lang, top_k=_TOP_K)
        except ValueError as exc:
            print(f"\n  ✗  Search error: {exc}\n")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ✗  Unexpected error during search: {exc}\n")
            continue

        # --- Display results ---
        _print_results(results, lang)

        # --- Continue prompt ---
        again = input("  Press Enter to search again, or type 'exit' to quit: ").strip().lower()
        print()
        if again in {"exit", "quit", "q", "0"}:
            _divider()
            print("  Goodbye! / وداعاً")
            _divider()
            print()
            break


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    main()
