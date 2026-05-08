# 🔍 Positional Inverted Index — IR Search Engine

> A bilingual (English & Arabic) Information Retrieval system built from scratch,
> featuring a **positional inverted index** and **TF-IDF cosine-similarity** ranking.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Usage Example](#usage-example)
- [Scoring Formula](#scoring-formula)
- [Corpus Layout](#corpus-layout)
- [Modules](#modules)
- [Notes](#notes)

---

## Overview

This project implements a complete **vector-space Information Retrieval engine** that:

1. Reads a corpus of **20 plain-text documents** (10 English + 10 Arabic).
2. Builds a **positional inverted index** and saves it to `index.json`.
3. Accepts free-text queries in either language and ranks documents using **TF-IDF weighted cosine similarity**.
4. Presents the **Top-5 results** through a clean interactive terminal interface.

---

## Features

| Feature | Details |
|---|---|
| 🌍 Bilingual | English (Porter stemming) + Arabic (ISRI stemming) |
| ⚡ Auto-indexing | Builds `index.json` on first run automatically |
| 📐 Ranking | TF-IDF · Cosine Similarity |
| 🗂️ Positional Index | Stores token positions for potential phrase-query extension |
| 🖥️ Interactive UI | Clean terminal interface with ASCII borders |
| 🔒 Error Handling | Missing index / missing data / empty query all handled gracefully |

---

## Project Structure

```
perpindex/
│
├── data/                       # Corpus (not committed to git if large)
│   ├── English/
│   │   ├── 1.txt
│   │   ├── 2.txt
│   │   └── ... (up to 10.txt)
│   └── Arabic/
│       ├── 1.txt
│       └── ... (up to 10.txt)
│
├── src/
│   ├── preprocessor.py         # Text preprocessing (tokenize, stop-words, stem)
│   ├── indexer.py              # Positional inverted index builder & loader
│   └── search_engine.py        # TF-IDF cosine-similarity ranking engine
│
├── main.py                     # ← Entry point (interactive terminal UI)
├── index.json                  # Generated automatically on first run
├── pyrightconfig.json          # IDE/Pylance path configuration
├── requirements.txt            # Python dependencies
└── README.md
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        PIPELINE                             │
│                                                             │
│  Raw Text                                                   │
│     │                                                       │
│     ▼  preprocessor.py                                      │
│  Tokenize → Remove Stop-words → Stem (Porter / ISRI)        │
│     │                                                       │
│     ▼  indexer.py                                           │
│  Positional Inverted Index  +  DF table  →  index.json      │
│     │                                                       │
│     ▼  search_engine.py                                     │
│  Query Vector  ·  Doc Vectors  →  Cosine Similarity Score   │
│     │                                                       │
│     ▼  main.py                                              │
│  Top-K Ranked Results displayed in terminal                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Setup & Installation

### Prerequisites

- Python **3.10+** (tested on 3.13)
- `pip` or a virtual environment manager

### 1 — Clone the repository

```bash
git clone https://github.com/A7MEDRAGAB82/search-engine.git
cd search-engine
```

### 2 — Create & activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install nltk
```

> NLTK corpora (`punkt_tab`, `stopwords`) are downloaded **automatically** on the first run.

---

## Running the Application

Make sure you are in the **project root** (where `main.py` lives) and the virtual environment is active, then:

```bash
python main.py
```

> ⚠️ **Arabic display tip:** Use **Windows Terminal** (not classic PowerShell/CMD) for correct RTL rendering of Arabic text.

---

## Usage Example

```
+==============================================================+
|        IR Search Engine  --  TF-IDF Cosine Similarity        |
|                 Languages: English  |  Arabic                |
+==============================================================+

  ✓  Engine ready  –  20 documents indexed

┌──────────────────────────────────────────────────────────────┐
│  Select Search Language                                      │
└──────────────────────────────────────────────────────────────┘
    [1]  English
    [2]  Arabic
    [0]  Exit

  ›  Enter choice: 1

  ›  Query: information retrieval systems

┌──────────────────────────────────────────────────────────────┐
│  Top 5 results  [English]                                    │
├──────────────────────────────────────────────────────────────┤
│  1. Score: 0.9241   Doc ID: en_3                             │
│     Path : data\English\3.txt                                │
│  ··········································                  │
│  2. Score: 0.8105   Doc ID: en_7                             │
│     Path : data\English\7.txt                                │
│  ...                                                         │
└──────────────────────────────────────────────────────────────┘
```

---

## Scoring Formula

### TF Weight (logarithmic)
```
tf_weight(t, d) = 1 + log₁₀(tf(t, d))    if tf > 0
               = 0                          otherwise
```

### IDF Weight
```
idf_weight(t) = log₁₀(N / df(t))
```

### TF-IDF
```
tfidf(t, d) = tf_weight(t, d) × idf_weight(t)
```

### Cosine Similarity
```
sim(q, d) = (q · d) / (‖q‖ × ‖d‖)
```

Only documents sharing **at least one term** with the query are scored (sparse evaluation).

---

## Corpus Layout

| Language | Directory | Files | Doc IDs |
|---|---|---|---|
| English | `data/English/` | `1.txt` – `10.txt` | `en_1` – `en_10` |
| Arabic | `data/Arabic/` | `1.txt` – `10.txt` | `ar_1` – `ar_10` |

---

## Modules

### `preprocessor.py`
| Step | English | Arabic |
|---|---|---|
| Tokenization | NLTK `word_tokenize` | Whitespace split |
| Normalization | Lowercase | Remove diacritics, unify Alif/Ya/Ta-Marbuta |
| Stop-word removal | NLTK English list | Custom Arabic list (~80 words) |
| Stemming | Porter Stemmer | ISRI Light Stemmer |

### `indexer.py`
- Iterates over all corpus files.
- Calls `preprocessor.process_text()` on each document.
- Builds a **positional inverted index**: `{ term → { doc_id → [positions] } }`.
- Stores **document frequency (df)** and **document metadata** (path, lang, length).
- Serializes the full index to `index.json`.

### `search_engine.py`
- Loads `index.json` via `load_index()`.
- Converts the query to a TF-IDF vector using the same preprocessor.
- Retrieves candidate documents (sparse — only docs containing query terms).
- Scores each candidate via cosine similarity.
- Returns a ranked `list[SearchResult]`.

### `main.py`
- Entry point.
- Auto-builds the index if `index.json` is missing.
- Interactive loop: language selection → query input → display Top-5 results.

---

## Notes

- The index is built **once** and reused on subsequent runs.  
  To **rebuild** it (e.g. after changing corpus files), delete `index.json` and rerun `python main.py`.
- Both the query and documents go through the **same preprocessing pipeline**, ensuring consistent stemming.
- The engine supports **sparse evaluation** — it never scores documents that share zero terms with the query.
