"""
preprocessor.py
===============
Information Retrieval – text preprocessing module.

Exposes a single public function:
    process_text(text: str, lang: str) -> list[str]

Supported languages
-------------------
    "en"  – English  : tokenize → stop-word removal (NLTK) → Porter stemming
    "ar"  – Arabic   : normalise → stop-word removal → ISRI light stemming

Dependencies
------------
    pip install nltk

On first run, the required NLTK corpora are downloaded automatically.
"""

from __future__ import annotations

import re
import unicodedata
import logging

import nltk

# ---------------------------------------------------------------------------
# Bootstrap – download NLTK data only when missing
# ---------------------------------------------------------------------------
_NLTK_RESOURCES: list[tuple[str, str]] = [
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]

for _path, _pkg in _NLTK_RESOURCES:
    try:
        nltk.data.find(_path)
    except LookupError:
        logging.info("Downloading NLTK resource: %s", _pkg)
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, ISRIStemmer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------
_porter = PorterStemmer()
_isri   = ISRIStemmer()

_EN_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Arabic constants
# ---------------------------------------------------------------------------

# Comprehensive Arabic stop-word list (function words, pronouns, particles …)
_AR_STOPWORDS: frozenset[str] = frozenset({
    "من", "إلى", "عن", "على", "في", "مع", "هو", "هي", "هم", "هن",
    "أنا", "أنت", "أنتِ", "نحن", "أنتم", "أنتن", "هذا", "هذه",
    "ذلك", "تلك", "هؤلاء", "أولئك", "التي", "الذي", "اللذين",
    "اللتين", "الذين", "اللواتي", "كان", "كانت", "يكون", "تكون",
    "ليس", "لم", "لن", "لا", "ما", "ماذا", "لماذا", "كيف", "متى",
    "أين", "أي", "كل", "بعض", "أكثر", "أقل", "قد", "قبل", "بعد",
    "حين", "عند", "ثم", "أو", "و", "ف", "ب", "ك", "ل", "بل",
    "إن", "أن", "إذا", "لو", "حتى", "رغم", "لأن", "لكن", "غير",
    "سوى", "مما", "مهما", "حيث", "إذ", "إذن", "إلا", "أيضا",
    "جدا", "فقط", "ربما", "دائما", "أحيانا", "هناك", "هنا",
    "نعم", "بلى", "لقد", "إنه", "إنها", "إنهم", "إننا", "يا",
    "ها", "ال", "ذا", "ذي", "تي", "اي",
})

# Arabic Unicode ranges – keep only Arabic letters and spaces
_AR_KEEP_PATTERN = re.compile(r"[^\u0600-\u06FF\s]")

# Diacritic (tashkeel) codepoints
_DIACRITICS = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4"
    r"\u06E7\u06E8\u06EA-\u06ED\u0640]"
)

# Alif variants → bare Alif (ا)
_ALIF_RE = re.compile(r"[إأآٱ]")

# Ya variants: dotless Ya (ى) → Ya with dots (ي)
_YA_RE = re.compile(r"ى")

# Ta Marbuta (ة) → Ha (ه)  – optional but common in IR normalisation
_TA_MARBUTA_RE = re.compile(r"ة")


# ===========================================================================
# English pipeline
# ===========================================================================

def _english_pipeline(text: str) -> list[str]:
    """Tokenise → lower-case → remove stop-words → Porter-stem."""
    tokens: list[str] = word_tokenize(text.lower())
    stems: list[str] = []
    for token in tokens:
        # Keep only alphabetic tokens
        if not token.isalpha():
            continue
        if token in _EN_STOPWORDS:
            continue
        stems.append(_porter.stem(token))
    return stems


# ===========================================================================
# Arabic pipeline
# ===========================================================================

def _normalise_arabic(text: str) -> str:
    """
    Apply a series of character-level normalisations:
      1. Remove diacritics (tashkeel).
      2. Unify Alif variants → ا
      3. Unify Dotless-Ya → ي
      4. Normalise Ta Marbuta → ه
      5. Strip non-Arabic characters.
    """
    text = _DIACRITICS.sub("", text)
    text = _ALIF_RE.sub("ا", text)
    text = _YA_RE.sub("ي", text)
    text = _TA_MARBUTA_RE.sub("ه", text)
    text = _AR_KEEP_PATTERN.sub(" ", text)
    return text


def _arabic_pipeline(text: str) -> list[str]:
    """Normalise → tokenise → remove stop-words → ISRI-stem."""
    text = _normalise_arabic(text)
    tokens: list[str] = text.split()
    stems: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in _AR_STOPWORDS:
            continue
        stems.append(_isri.stem(token))
    return stems


# ===========================================================================
# Public API
# ===========================================================================

def process_text(text: str, lang: str) -> list[str]:
    """
    Preprocess *text* according to the chosen language pipeline.

    Parameters
    ----------
    text : str
        Raw input text (UTF-8 encoded strings are handled transparently
        because Python 3 strings are Unicode-aware).
    lang : str
        Language code – ``"en"`` for English, ``"ar"`` for Arabic.
        The value is case-insensitive.

    Returns
    -------
    list[str]
        Ordered list of stems produced by the language-specific pipeline.

    Raises
    ------
    ValueError
        If *lang* is not one of the supported language codes.

    Examples
    --------
    >>> process_text("Running dogs are quickly chasing the cats", "en")
    ['run', 'dog', 'quickli', 'chase', 'cat']

    >>> process_text("الكلاب تركض بسرعة في الحديقة", "ar")
    ['كلب', 'ركض', 'سرع', 'حدق']
    """
    if not isinstance(text, str):
        raise TypeError(f"'text' must be str, got {type(text).__name__!r}")

    lang = lang.strip().lower()

    if lang == "en":
        return _english_pipeline(text)
    elif lang == "ar":
        return _arabic_pipeline(text)
    else:
        raise ValueError(
            f"Unsupported language code {lang!r}. "
            "Use 'en' for English or 'ar' for Arabic."
        )


# ===========================================================================
# Quick smoke-test  (python preprocessor.py)
# ===========================================================================

if __name__ == "__main__":
    import sys, io

    # Reconfigure stdout to UTF-8 so Arabic characters display on Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    _EN_SAMPLE = (
        "Information Retrieval systems are designed to efficiently "
        "retrieve relevant documents from large text collections."
    )
    _AR_SAMPLE = (
        "أنظمة استرجاع المعلومات تهدف إلى إيجاد الوثائق ذات الصلة "
        "بسرعة ودقة من مجموعات النصوص الكبيرة."
    )

    print("=== English pipeline ===")
    print(process_text(_EN_SAMPLE, "en"))

    print("\n=== Arabic pipeline ===")
    print(process_text(_AR_SAMPLE, "ar"))
