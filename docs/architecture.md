# Hybrid Extractive-Abstractive Summarization Platform

This document captures the high-level architecture that implements the requirements laid out in the README and assignment briefs.

## System Overview

```
Client (React/Vite SPA)
        │
        ▼
REST API (FastAPI)
        │
 ┌──────┼───────────────────────────────────────────┐
 │ HybridSummarizer                                 │
 │                                                  │
 │ 1. Preprocessing (NLTK + spaCy)                  │
 │ 2. Extractive (Sentence-BERT + TextRank + MMR)   │
 │ 3. Abstractive (BART-large with beam search)     │
 │ 4. Post-processing (coverage + repetition drop)  │
 │ 5. Evaluation (ROUGE, BLEU, METEOR, BERTScore)   │
 └──────────────────────────────────────────────────┘
```

## Key Components

| Layer | Description |
| ----- | ----------- |
| `frontend/` | React UI to collect documents, trigger summarization, and visualize extractive seeds, timings, and metric evaluations. |
| `backend/app.py` | FastAPI server exposing `/summarize`, `/evaluate`, and `/health`. |
| `backend/src/preprocessing.py` | Cleans text, segments sentences, removes stopwords, and lemmatizes tokens per assignment spec. |
| `backend/src/extractive_summarizer.py` | Sentence-BERT embeddings + TextRank + handcrafted features + KL-aware MMR for sentence selection. |
| `backend/src/abstractive_summarizer.py` | BART-large encoder-decoder with configurable generation hyperparameters. |
| `backend/src/postprocessing.py` | Trigram blocking, redundancy suppression, and length control for coverage and quality. |
| `backend/src/evaluation.py` | Implements ROUGE, BLEU, METEOR, and BERTScore for quantitative analysis. |

## Data Flow

1. **Input**: Raw document ingested from UI.
2. **Preprocess**: Cleaning, segmentation, normalization.
3. **Extractive Ranking**:
   - Sentence-BERT embeddings → cosine similarity graph → TextRank.
   - Feature fusion (position, TF-IDF centroid, length, named-entity density, cue phrases).
   - Maximal Marginal Relevance with KL-inspired redundancy control to pick top-*k* sentences.
4. **Abstractive Refinement**:
   - Concatenated extracted sentences fed to `facebook/bart-large-cnn`.
   - Beam search, no-repeat trigrams, and configurable length penalties.
5. **Post-processing**: Coverage enforcement via trigram blocking, sentence deduplication using embeddings, and compression-length control.
6. **Evaluation**: Optional comparison with reference summaries through `/evaluate`.

## Deployment

1. Install Python deps: `pip install -r backend/requirements.txt`.
2. Download spaCy model: `python -m spacy download en_core_web_sm`.
3. Launch API: `python backend/main.py` or `uvicorn backend.app:app --reload`.
4. Frontend: `cd frontend && npm install && npm run dev`.


