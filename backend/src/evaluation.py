from __future__ import annotations

from typing import Dict, Iterable, List

from bert_score import score as bert_score
from nltk.translate import bleu_score, meteor
from rouge_score import rouge_scorer


DEFAULT_METRICS = {"rouge", "bleu", "meteor", "bertscore"}


def evaluate_summary(
    generated: str,
    reference: str,
    metrics: Iterable[str] | None = None,
) -> Dict[str, float]:
    metrics = set(metrics or DEFAULT_METRICS)
    scores: Dict[str, float] = {}

    if "rouge" in metrics:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge = scorer.score(reference, generated)
        scores.update({k: round(v.fmeasure, 4) for k, v in rouge.items()})

    if "bleu" in metrics:
        smoothie = bleu_score.SmoothingFunction().method3
        bleu = bleu_score.sentence_bleu(
            [reference.split()],
            generated.split(),
            smoothing_function=smoothie,
        )
        scores["bleu"] = round(float(bleu), 4)

    if "meteor" in metrics:
        scores["meteor"] = round(
            float(meteor.single_meteor_score(reference, generated)), 4
        )

    if "bertscore" in metrics:
        precision, recall, f1 = bert_score(
            [generated], [reference], lang="en", verbose=False, rescale_with_baseline=True
        )
        scores["bertscore_f1"] = round(float(f1.mean()), 4)

    return scores


