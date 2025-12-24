"""
LLM Predictors Flow (print-only key steps)
=========================================

Purpose:
- Print only the major flow for a given question (and optional gold label)
- Do NOT print causal triples or graph construction details
- Show LLM Stage-A draft output and Stage-B reflection output
- Show the final corrected result from the reflective predictors

Implementation notes:
- Uses the quiet reflective predictors to suppress internal builder/parser logs
- Keeps the interface simple: run_case(...) and run_cases(...)
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from question_parser import QuestionParser
from ego_expansion_builder import EgoExpansionCausalBuilder
from llm_predictors_quiet import (
    predict_meta_informed_llm_reflective,
    predict_combined_context_llm_reflective,
)


DEFAULT_MODEL = "gemma2:27b"


def _norm(lbl: Optional[str]) -> Optional[str]:
    if lbl is None:
        return None
    m = {
        "no effect": "no_effect",
        "no_effect": "no_effect",
        "more": "more",
        "less": "less",
    }
    return m.get(str(lbl).strip().lower())


def init_defaults(
    model: str = DEFAULT_MODEL,
    max_depth: int = 1,
    max_neighbors: int = 2,
    max_relations: int = 2,
) -> Tuple[QuestionParser, EgoExpansionCausalBuilder, str]:
    """Create default parser/builder with a light config to reduce latency/noise."""
    parser = QuestionParser(model_name=model)
    builder = EgoExpansionCausalBuilder(
        model_name=model,
        max_neighbors_per_seed=max_neighbors,
        max_expansion_depth=max_depth,
        max_relations_per_entity=max_relations,
    )
    return parser, builder, model


def run_case(
    question: str,
    gold: Optional[str] = None,
    parser: Optional[QuestionParser] = None,
    builder: Optional[EgoExpansionCausalBuilder] = None,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Run both reflective predictors and print only the key flow.

    Prints:
      - QUESTION and GOLD (if provided)
      - [Meta-Informed] Stage A & Stage B raw LLM outputs
      - [Meta-Informed] Final answer (+ corrected/source flags)
      - [Combined-Context] Stage A & Stage B raw LLM outputs
      - [Combined-Context] Final answer (+ corrected/source flags)
    Returns a compact result dict with normalized labels.
    """

    if parser is None or builder is None:
        parser, builder, model = init_defaults(model)

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)
    if gold is not None:
        print(f"GOLD: {_norm(gold)}")
    print("-" * 80)

    # Meta-informed reflective
    r1 = predict_meta_informed_llm_reflective(question, parser, builder, model)
    print("[Meta-Informed] Stage A - Draft (LLM raw):")
    print(r1.get("raw", {}).get("analysis", ""))
    print("[Meta-Informed] Stage B - Reflection (LLM raw):")
    print(r1.get("raw", {}).get("reflection", ""))
    print(
        f"[Meta-Informed] Final: {r1.get('final_answer')} | corrected={r1.get('corrected')} | source={r1.get('correction_source')}"
    )
    print("-" * 80)

    # Combined-context reflective
    r2 = predict_combined_context_llm_reflective(question, parser, builder, model)
    print("[Combined-Context] Stage A - Draft (LLM raw):")
    print(r2.get("raw", {}).get("analysis", ""))
    print("[Combined-Context] Stage B - Reflection (LLM raw):")
    print(r2.get("raw", {}).get("reflection", ""))
    print(
        f"[Combined-Context] Final: {r2.get('final_answer')} | corrected={r2.get('corrected')} | source={r2.get('correction_source')}"
    )

    return {
        "question": question,
        "gold": _norm(gold) if gold else None,
        "meta_final": _norm(r1.get("final_answer")),
        "meta_corrected": r1.get("corrected"),
        "comb_final": _norm(r2.get("final_answer")),
        "comb_corrected": r2.get("corrected"),
    }


def run_cases(
    cases: Iterable[Dict[str, Any]],
    parser: Optional[QuestionParser] = None,
    builder: Optional[EgoExpansionCausalBuilder] = None,
    model: str = DEFAULT_MODEL,
) -> List[Dict[str, Any]]:
    """Run multiple cases; each item should have keys: question, and optional ground_truth."""
    if parser is None or builder is None:
        parser, builder, model = init_defaults(model)
    results: List[Dict[str, Any]] = []
    for case in cases:
        q = case.get("question", "")
        g = case.get("ground_truth")
        results.append(run_case(q, g, parser, builder, model))
    return results


__all__ = [
    "init_defaults",
    "run_case",
    "run_cases",
]

