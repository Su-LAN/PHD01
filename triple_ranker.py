"""Utility to rank causal triples against a question using semantic similarity.

Given a builder result dictionary, a question, and the desired number of
textual variations per triple, this module generates natural-language
descriptions for each triple, scores them with ``SemanticRanker`` against the
question, averages the scores, and returns triples ordered by relevance.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from semantic_ranker import SemanticRanker
except Exception as exc:  # pragma: no cover - semantic_ranker should exist
    raise ImportError(
        "semantic_ranker.py must be importable to rank triples."
    ) from exc


TripleEntry = Dict[str, Any]


def rank_triples(
    result: Dict[str, Any],
    question: str,
    num_variations: int,
    backend: str = "auto",
    top_m: int | None = 3,
) -> List[Dict[str, Any]]:
    """Rank triples by semantic similarity to the question.

    Args:
        result: Output dictionary from ``EgoExpansionCausalBuilder.build_causal_chain``
            (must contain an ``"edges"`` list) or a list of triple entries.
        question: Original question string.
        num_variations: Number of textual variants to generate per triple (>=1).
        backend: SemanticRanker backend (``"auto"``, ``"sbert"``, ``"tfidf"``, ``"bow"``).
        top_m: Use only the top ``m`` textual scores per triple when averaging.
            ``None`` or values <=0 fallback to using all variations.

    Returns:
        List of triples ordered by descending average similarity score. Each item
        contains the triple tuple, averaged score, per-text scores, and metadata
        (description, confidence, depth if available).
    """

    if num_variations <= 0:
        num_variations = 1

    triples_source: Sequence[Any]
    if isinstance(result, dict):
        triples_source = result.get("edges") or result.get("triples") or []
    else:
        triples_source = result  # allow direct list input

    if not triples_source:
        return []

    ranker = SemanticRanker(backend=backend)
    ranked_output: List[Dict[str, Any]] = []

    for idx, entry in enumerate(triples_source):
        norm = _normalize_triple_entry(entry)
        if norm is None:
            continue

        head, relation, tail, description, meta = norm
        texts = _generate_text_variations(head, relation, tail, description, num_variations)
        if not texts:
            continue

        scores_info = ranker.rank(question, texts, top_k=None) if question else []
        index_to_score = {item["index"]: float(item["score"]) for item in scores_info}

        text_scores: List[Dict[str, Any]] = []
        for text_index, text in enumerate(texts):
            score = index_to_score.get(text_index, 0.0)
            text_scores.append({"text": text, "score": score})

        used_m = len(text_scores)
        if text_scores:
            scores_sorted = sorted((item["score"] for item in text_scores), reverse=True)
            m = top_m if top_m and top_m > 0 else len(scores_sorted)
            m = min(m, len(scores_sorted))
            used_m = m if m > 0 else len(scores_sorted)
            avg_score = sum(scores_sorted[:m]) / m if m > 0 else 0.0
        else:
            avg_score = 0.0

        ranked_output.append(
            {
                "index": idx,
                "triple": (head, relation, tail),
                "description": description,
                "confidence": meta.get("confidence"),
                "depth": meta.get("depth"),
                "avg_score": avg_score,
                "top_m": used_m,
            }
        )

    ranked_output.sort(key=lambda item: item["avg_score"], reverse=True)
    return ranked_output


def _normalize_triple_entry(entry: Any) -> Tuple[str, str, str, str, Dict[str, Any]] | None:
    """Extract (head, relation, tail, description, meta) from a triple entry."""

    if isinstance(entry, dict):
        if "triple" in entry and isinstance(entry["triple"], (tuple, list)):
            triple_seq = entry["triple"]
            if len(triple_seq) != 3:
                return None
            head, relation, tail = map(str, triple_seq)
        elif all(k in entry for k in ("head", "relation", "tail")):
            head = str(entry.get("head", "")).strip()
            relation = str(entry.get("relation", "")).strip()
            tail = str(entry.get("tail", "")).strip()
        else:
            return None

        description = str(entry.get("description", "")).strip()
        meta = {
            key: entry.get(key)
            for key in ("confidence", "depth", "source", "raw")
            if key in entry
        }
    elif isinstance(entry, (tuple, list)) and len(entry) == 3:
        head, relation, tail = map(str, entry)
        description = ""
        meta = {}
    else:
        return None

    head = head.strip()
    relation = relation.strip()
    tail = tail.strip()

    if not head or not relation or not tail:
        return None

    return head, relation, tail, description, meta


def _generate_text_variations(
    head: str,
    relation: str,
    tail: str,
    description: str,
    num_variations: int,
) -> List[str]:
    """Create ``num_variations`` textual descriptions for a triple."""

    relation_phrase = relation.replace("_", " ") or "relates to"
    description_clean = description.strip()

    base_variations: List[str] = []
    base_variations.append(_ensure_sentence(f"{head} {relation_phrase} {tail}"))
    base_variations.append(
        _ensure_sentence(f"{head} directly {relation_phrase} {tail}")
    )
    base_variations.append(
        _ensure_sentence(f"{tail} is an outcome when {head} {relation_phrase}")
    )
    base_variations.append(
        _ensure_sentence(f"{head} {relation_phrase} {tail}, leading to {tail}")
    )

    if description_clean:
        base_variations.append(
            _ensure_sentence(
                f"{head} {relation_phrase} {tail}. {description_clean.rstrip('.')}"
            )
        )

    # Deduplicate while preserving order
    unique_variations: List[str] = []
    seen = set()
    for text in base_variations:
        normalized = text.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_variations.append(normalized)

    if not unique_variations:
        unique_variations = [f"{head} {relation_phrase} {tail}."]

    if num_variations <= len(unique_variations):
        return unique_variations[:num_variations]

    # If more variations requested than available, cycle through the existing ones
    cycled: List[str] = []
    for i in range(num_variations):
        cycled.append(unique_variations[i % len(unique_variations)])
    return cycled


def _ensure_sentence(text: str) -> str:
    """Ensure the text ends with sentence punctuation and has normalized spacing."""

    cleaned = " ".join(text.split())
    if not cleaned:
        return cleaned
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _demo() -> None:
    """Quick manual check when running the module directly."""

    sample_result = {
        "edges": [
            {
                "head": "高盐饮食",
                "relation": "导致",
                "tail": "血压升高",
                "description": "钠离子摄入增加导致体液潴留",
                "confidence": 0.82,
                "depth": 1,
            },
            {
                "head": "运动",
                "relation": "降低",
                "tail": "血压",
                "description": "通过改善血管弹性降低外周阻力",
                "confidence": 0.73,
                "depth": 1,
            },
        ]
    }

    ranked = rank_triples(sample_result, "导致血压升高的主要因素是什么？", num_variations=3)
    for item in ranked:
        print(f"score={item['avg_score']:.3f} triple={item['triple']}")


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    _demo()
