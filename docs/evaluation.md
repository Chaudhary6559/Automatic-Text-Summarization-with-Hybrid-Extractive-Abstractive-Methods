# Evaluation & Testing Guide

This guide explains how to benchmark the hybrid summarization system quantitatively and how to run automated checks.

## 1. Automatic Metrics (`/evaluate`)

The backend exposes a dedicated endpoint for ROUGE, BLEU, METEOR, and BERTScore.

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
        "generated_summary": "Model produced summary here.",
        "reference_summary": "Human-written or gold summary here."
      }'
```

The response is a JSON payload containing the requested metrics (defaults to all four). To limit the computation to a subset:

```json
{
  "generated_summary": "...",
  "reference_summary": "...",
  "metrics": ["rouge", "bertscore"]
}
```

## 2. Manual Quality Checklist

1. **Relevance** – Does the summary keep the main facts?
2. **Coherence** – Are the sentences logically ordered?
3. **Fluency** – Does the language read naturally?
4. **Non-redundancy** – Are repeated ideas removed?
5. **Faithfulness** – Is every claim grounded in the source?

The React UI includes inputs for both the source document and an optional reference summary so analysts can quickly inspect qualitative outputs next to metric scores.

## 3. Automated Tests

```
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
pytest backend/tests
```

The current test suite validates preprocessing + feature extraction logic to ensure the extractive module receives the handcrafted features called for in the assignment.


