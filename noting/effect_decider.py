from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


POSITIVE_RELS = {
    "increases",
    "produces",
    "enables",
    "promotes",
    "causes",  # treated as positive toward tail
}

NEGATIVE_RELS = {
    "reduces",
    "decreases",
    "inhibits",
    "suppresses",
}

NEUTRAL_RELS = {"no_relation", "depends on"}


def decide_effect(
    question: str,
    ranked: Sequence[Dict[str, Any]],
    target: Optional[str] = None,
    weight_avg: float = 0.7,
    weight_confidence: float = 0.3,
    min_evidence: float = 0.15,
    margin: float = 1.1,
    return_details: bool = True,
    # New knobs to strengthen "no_effect" behavior
    support_threshold: float = 0.20,
    min_dir_supports: int = 1,
    prefer_neutral: bool = True,
    neutral_factor: float = 1.0,
) -> Dict[str, Any]:
    """
    Decide among {"more", "less", "no_effect"} using ranked triples.

    - Aggregates evidence for increase vs decrease using (avg_score, confidence)
      into a combined score per triple.
    - Sums positives/negatives (optionally focusing on a target variable).
    - If both sides weak and neutral evidence present (or nothing strong), returns "no_effect".

    Args:
      question: Original question text (for logging only).
      ranked: Output from rank_triples (items with keys: triple, avg_score, confidence, ...).
      target: Optional target variable to focus on (e.g., "current").
      weight_avg: Weight for semantic similarity avg_score.
      weight_confidence: Weight for LLM confidence.
      min_evidence: Minimal score to call an effect; otherwise prefer "no_effect".
      margin: How much one side should exceed the other (ratio) to win.
      return_details: Whether to include intermediate scores and supports.

    Returns: dict with keys: decision, pos_score, neg_score, no_rel_score, used_target, supports
    """

    if not ranked:
        return {
            "decision": "no_effect",
            "pos_score": 0.0,
            "neg_score": 0.0,
            "no_rel_score": 0.0,
            "used_target": target,
            "supports": [],
        }

    # normalize weights
    total_w = (weight_avg or 0.0) + (weight_confidence or 0.0)
    if total_w <= 0:
        alpha, beta = 1.0, 0.0
    else:
        alpha = weight_avg / total_w
        beta = weight_confidence / total_w

    def combined(item: Dict[str, Any]) -> float:
        a = float(item.get("avg_score", 0.0))
        c = item.get("confidence")
        try:
            c = 0.0 if c is None else float(c)
        except (TypeError, ValueError):
            c = 0.0
        return alpha * a + beta * c

    def matches_target(tr: Tuple[str, str, str]) -> bool:
        if not target:
            return True
        h, r, t = (str(tr[0]).lower(), str(tr[1]).lower(), str(tr[2]).lower())
        tgt = target.lower()
        return tgt in (h, t) or tgt in h or tgt in t

    pos_sum = 0.0
    neg_sum = 0.0
    neu_sum = 0.0
    # Count of directional triples whose combined score crosses support_threshold
    pos_strong = 0
    neg_strong = 0
    pos_items: List[Dict[str, Any]] = []
    neg_items: List[Dict[str, Any]] = []
    neu_items: List[Dict[str, Any]] = []

    for item in ranked:
        triple = item.get("triple")
        if not (isinstance(triple, (tuple, list)) and len(triple) == 3):
            continue
        head, relation, tail = map(lambda x: str(x).strip(), triple)
        relation_l = relation.lower()
        if not matches_target((head, relation, tail)):
            continue

        score = combined(item)

        if relation_l in POSITIVE_RELS:
            pos_sum += score
            if return_details:
                pos_items.append({**item, "combined_score": score})
            if score >= support_threshold:
                pos_strong += 1
        elif relation_l in NEGATIVE_RELS:
            neg_sum += score
            if return_details:
                neg_items.append({**item, "combined_score": score})
            if score >= support_threshold:
                neg_strong += 1
        elif relation_l in NEUTRAL_RELS:
            neu_sum += score
            if return_details:
                neu_items.append({**item, "combined_score": score})
        else:
            # unknown relation: treat as neutral evidence
            neu_sum += score
            if return_details:
                neu_items.append({**item, "combined_score": score})

    # Decision logic
    decision: str
    # 1) If no strong directional supports at all, prefer no_effect
    if (pos_strong + neg_strong) < max(1, int(min_dir_supports)):
        decision = "no_effect"
    # 2) Not enough directional mass overall
    elif max(pos_sum, neg_sum) < float(min_evidence):
        decision = "no_effect"
    # 3) Neutral dominance (optional): if neutral outweighs directional
    elif prefer_neutral and (neu_sum * float(neutral_factor) >= max(pos_sum, neg_sum)):
        decision = "no_effect"
    else:
        # 4) Choose the stronger side if it's sufficiently larger
        if pos_sum >= neg_sum * float(margin):
            decision = "more"
        elif neg_sum >= pos_sum * float(margin):
            decision = "less"
        else:
            decision = "no_effect"

    result: Dict[str, Any] = {
        "decision": decision,
        "pos_score": float(pos_sum),
        "neg_score": float(neg_sum),
        "no_rel_score": float(neu_sum),
        "used_target": target,
    }

    if return_details:
        # Keep top 3 supports in each category for traceability
        def topk(arr: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
            arr = sorted(arr, key=lambda x: x.get("combined_score", 0.0), reverse=True)
            # trim heavy fields
            out = []
            for x in arr[:k]:
                y = {
                    "triple": x.get("triple"),
                    "avg_score": x.get("avg_score"),
                    "confidence": x.get("confidence"),
                    "combined_score": x.get("combined_score"),
                }
                out.append(y)
            return out

        result["supports"] = {
            "positive": topk(pos_items),
            "negative": topk(neg_items),
            "neutral": topk(neu_items),
        }
        # Add lightweight stats for debugging/tuning
        result["stats"] = {
            "pos_strong": int(pos_strong),
            "neg_strong": int(neg_strong),
            "support_threshold": float(support_threshold),
            "min_dir_supports": int(min_dir_supports),
        }

    return result


__all__ = ["decide_effect"]
