"""Post-ranking triple selection helpers.

This module takes the output of ``rank_triples`` (from ``triple_ranker``)
and performs an additional selection pass that combines the semantic
similarity score (top-M average) with the triple confidence emitted by the
graph expansion LLM. Users can control the proportion of triples kept and
the weighting strategy.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence


RankedTriple = Dict[str, Any]


def select_triples(
    ranked: Sequence[RankedTriple],
    keep_fraction: float,
    weight_avg: float = 0.7,
    weight_confidence: float = 0.3,
    confidence_default: float = 0.0,
) -> List[RankedTriple]:
    """Select top triples by combining semantic score and confidence.

    Args:
        ranked: Output from ``rank_triples`` (must contain ``avg_score`` and
            optionally ``confidence``).
        keep_fraction: Fraction of items to keep (0 < fraction <= 1). Values
            outside the range are clamped.
        weight_avg: Weight for the semantic similarity score (avg_score).
        weight_confidence: Weight for the confidence value.
        confidence_default: Value to use when ``confidence`` is missing.

    Returns:
        List of triples sorted by the combined score, limited to the requested
        fraction. Each entry is the original dict extended with
        ``combined_score`` for downstream inspection.
    """

    if not ranked:
        return []

    fraction = max(0.0, min(1.0, keep_fraction))
    if fraction <= 0.0:
        return []

    # Normalise weights to avoid accidental over/underflow.
    total_weight = weight_avg + weight_confidence
    if total_weight == 0:
        weight_avg = 1.0
        weight_confidence = 0.0
        total_weight = 1.0
    alpha = weight_avg / total_weight
    beta = weight_confidence / total_weight

    enriched: List[RankedTriple] = []
    for item in ranked:
        avg_score = float(item.get("avg_score", 0.0))
        confidence = item.get("confidence")
        if confidence is None:
            confidence = confidence_default
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = confidence_default

        combined = alpha * avg_score + beta * confidence
        enriched.append({**item, "combined_score": combined})

    enriched.sort(key=lambda x: x["combined_score"], reverse=True)

    keep_count = max(1, math.ceil(len(enriched) * fraction))
    return enriched[:keep_count]


def filter_triples_by_threshold(
    ranked: Sequence[RankedTriple],
    min_combined_score: float,
    weight_avg: float = 0.7,
    weight_confidence: float = 0.3,
    confidence_default: float = 0.0,
) -> List[RankedTriple]:
    """Filter triples whose combined score crosses the given threshold."""

    if not ranked:
        return []

    total_weight = weight_avg + weight_confidence
    if total_weight == 0:
        weight_avg = 1.0
        weight_confidence = 0.0
        total_weight = 1.0
    alpha = weight_avg / total_weight
    beta = weight_confidence / total_weight

    filtered: List[RankedTriple] = []
    for item in ranked:
        avg_score = float(item.get("avg_score", 0.0))
        confidence = item.get("confidence")
        if confidence is None:
            confidence = confidence_default
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = confidence_default

        combined = alpha * avg_score + beta * confidence
        if combined >= min_combined_score:
            filtered.append({**item, "combined_score": combined})

    filtered.sort(key=lambda x: x["combined_score"], reverse=True)
    return filtered


def _demo() -> None:
    sample_ranked = [
        {
            "triple": ("A", "causes", "B"),
            "avg_score": 0.62,
            "confidence": 0.8,
            "top_m": 3,
        },
        {
            "triple": ("C", "leads_to", "D"),
            "avg_score": 0.55,
            "confidence": 0.6,
            "top_m": 3,
        },
    ]

    print("Top 50% by combined score:")
    for item in select_triples(sample_ranked, keep_fraction=0.5):
        print(item)


if __name__ == "__main__":  # pragma: no cover
    _demo()

