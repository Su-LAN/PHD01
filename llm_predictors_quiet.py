"""
Quiet versions of LLM-based predictors for WIQA-style causal questions.

These functions mirror the behavior of llm_predictors.predict_meta_informed_llm
and predict_combined_context_llm but suppress prints and return results directly.

Use these in batch evaluation to avoid noisy logs, while keeping the original
verbose functions available elsewhere for debugging.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

import ollama
from effect_decider import POSITIVE_RELS, NEGATIVE_RELS


def _normalize_label(lbl: str | None) -> str | None:
    if lbl is None:
        return None
    mapping = {
        "no effect": "no_effect",
        "no_effect": "no_effect",
        "more": "more",
        "less": "less",
    }
    return mapping.get(str(lbl).strip().lower(), None)


def _normalize_dir(x: str | None) -> str | None:
    if not x:
        return None
    t = str(x).strip().lower()
    if t in {"increase", "increases", "more", "higher", "greater", "grow", "rise", "rises"}:
        return "more"
    if t in {"decrease", "decreases", "less", "lower", "fewer", "reduced", "decline", "drop"}:
        return "less"
    if t in {"no_effect", "no effect", "none", "unchanged", "no change"}:
        return "no_effect"
    return None


def _summarize_relations(triples: list[dict]) -> Tuple[str, str]:
    pos_rels = [t for t in triples if t["triple"][1] in POSITIVE_RELS]
    neg_rels = [t for t in triples if t["triple"][1] in NEGATIVE_RELS]
    pos_context = (
        "\n".join(
            [f"  - {t['triple'][0]} → {t['triple'][2]} ({t['triple'][1]})" for t in pos_rels[:3]]
        )
        if pos_rels
        else "  (none)"
    )
    neg_context = (
        "\n".join(
            [f"  - {t['triple'][0]} → {t['triple'][2]} ({t['triple'][1]})" for t in neg_rels[:3]]
        )
        if neg_rels
        else "  (none)"
    )
    return pos_context, neg_context


# ======= Tunable guardrail parameters for stronger no_effect behavior =======
SUPPORT_THRESHOLD = 0.25       # confidence threshold to count a triple as strong
MIN_DIR_SUPPORTS = 2           # need at least this many strong directional supports
MIN_EVIDENCE_SUM = 0.25        # minimal directional mass; otherwise prefer no_effect
CONFIDENCE_GATE = 0.35         # model reflection confidence gate; under this prefer no_effect
PATH_HOPS = 2                  # require a path within this many hops between intervention and target


def _matches_name(node: str, name: str) -> bool:
    if not node or not name:
        return False
    a = str(node).lower().strip()
    b = str(name).lower().strip()
    return (a in b) or (b in a)


def _has_path_within_k(triples: list[dict], src_name: str | None, tgt_name: str | None, k: int = PATH_HOPS) -> bool:
    if not src_name or not tgt_name:
        return False
    # Build directed and undirected adjacency
    from collections import defaultdict, deque
    adj = defaultdict(list)
    undirected = defaultdict(list)
    nodes = set()
    for t in triples:
        h, r, ta = t.get("triple", (None, None, None))
        if not h or not ta:
            continue
        nodes.add(h)
        nodes.add(ta)
        adj[h].append(ta)
        undirected[h].append(ta)
        undirected[ta].append(h)

    # Candidates for source/target
    srcs = [n for n in nodes if _matches_name(n, src_name)]
    tgts = [n for n in nodes if _matches_name(n, tgt_name)]
    if not srcs or not tgts:
        # Try looser: if src_name matches any head relation token in triples
        return False

    def bfs(graph, sources) -> bool:
        for s in sources:
            dq = deque([(s, 0)])
            seen = {s}
            while dq:
                cur, d = dq.popleft()
                if d > k:
                    continue
                # Check hit
                if any(_matches_name(cur, t) for t in tgts):
                    return True
                for nb in graph.get(cur, []):
                    if nb not in seen and d + 1 <= k:
                        seen.add(nb)
                        dq.append((nb, d + 1))
        return False

    # Prefer directed path; fallback to undirected connectivity within k
    if bfs(adj, srcs):
        return True
    return bfs(undirected, srcs)


def _directional_evidence_stats(triples: list[dict], thr: float = SUPPORT_THRESHOLD) -> Tuple[float, float, int, int]:
    pos_sum = 0.0
    neg_sum = 0.0
    pos_strong = 0
    neg_strong = 0
    for t in triples:
        tr = t.get("triple")
        if not tr or len(tr) != 3:
            continue
        rel = str(tr[1]).lower()
        conf = t.get("confidence")
        try:
            c = float(conf) if conf is not None else 0.0
        except Exception:
            c = 0.0
        if rel in POSITIVE_RELS:
            pos_sum += c
            if c >= thr:
                pos_strong += 1
        elif rel in NEGATIVE_RELS:
            neg_sum += c
            if c >= thr:
                neg_strong += 1
    return pos_sum, neg_sum, pos_strong, neg_strong


class suppress_output:
    """Context manager to silence stdout/stderr during noisy calls."""

    def __enter__(self):
        self._stdout = io.StringIO()
        self._stderr = io.StringIO()
        self._enter_stdout = redirect_stdout(self._stdout)
        self._enter_stderr = redirect_stderr(self._stderr)
        self._enter_stdout.__enter__()
        self._enter_stderr.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self._enter_stderr.__exit__(exc_type, exc, tb)
        finally:
            self._enter_stdout.__exit__(exc_type, exc, tb)
        # swallow nothing; propagate exceptions if any
        return False


def predict_meta_informed_llm(
    question: str,
    parser,
    builder,
    model: str,
) -> Dict[str, Any]:
    """
    Quiet version of meta-informed LLM decision.
    Parses question structure, builds a small causal context, prompts LLM, and
    returns a structured result without printing.
    """

    # Parse question structure quietly
    with suppress_output():
        question_structure = parser.parse_question_structure(question)

    # Build causal context quietly
    with suppress_output():
        builder_result = builder.build_causal_chain(question)
    triples = builder.get_all_triples(builder_result, format="structured")

    top_triples = triples[:5] if len(triples) >= 5 else triples
    triple_context = "\n".join(
        [
            f"  - {t['triple'][0]} {t['triple'][1]} {t['triple'][2]} (confidence: {t.get('confidence', 0):.2f})"
            for t in top_triples
        ]
    )

    meta_prompt = f"""You are analyzing a causal reasoning question. Here is the parsed structure:

Original Question: {question}

Question Analysis:
- Is meta-level question: {question_structure.get('is_meta_level')}
- Intervention: {question_structure.get('intervention')}
- Target phrase: {question_structure.get('target_phrase')}
- Target direction: {question_structure.get('target_direction')}
- Target entity: {question_structure.get('target_entity')}
- Question type: {question_structure.get('question_type')}

Relevant Causal Relations:
{triple_context if triple_context else "  (No causal relations found)"}

Task: Based on the question structure and causal relations, determine the FINAL ANSWER.

Return your answer in JSON format:
{{
  "final_answer": "more|less|no_effect",
  "rationale": "Brief explanation of your reasoning (2-3 sentences)"
}}

IMPORTANT: Return ONLY valid JSON, nothing else."""

    try:
        response = ollama.generate(model=model, prompt=meta_prompt)
        llm_response = response.get("response", "").strip()

        json_match = re.search(r"\{[^}]+\}", llm_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            final_answer = _normalize_label(result.get("final_answer", ""))
            rationale = result.get("rationale", "")
        else:
            final_answer = None
            rationale = llm_response
            for label in ["more", "less", "no_effect"]:
                if label in llm_response.lower():
                    final_answer = label
                    break

        return {
            "method": "meta_informed_llm",
            "question_structure": question_structure,
            "final_answer": final_answer,
            "rationale": rationale,
            "raw_response": llm_response,
        }

    except Exception as e:
        return {
            "method": "meta_informed_llm",
            "question_structure": question_structure,
            "final_answer": "error",
            "rationale": str(e),
        }


def predict_combined_context_llm(
    question: str,
    parser,
    builder,
    model: str,
) -> Dict[str, Any]:
    """
    Quiet version of combined-context LLM decision.
    Combines original question, parsed meta info, causal relation summary and a
    reasoning guide into one prompt; returns JSON result without printing.
    """

    with suppress_output():
        question_structure = parser.parse_question_structure(question)

    with suppress_output():
        builder_result = builder.build_causal_chain(question)
    triples = builder.get_all_triples(builder_result, format="structured")

    from effect_decider import POSITIVE_RELS, NEGATIVE_RELS

    pos_rels = [t for t in triples if t["triple"][1] in POSITIVE_RELS]
    neg_rels = [t for t in triples if t["triple"][1] in NEGATIVE_RELS]

    pos_context = (
        "\n".join(
            [f"  - {t['triple'][0]} → {t['triple'][2]} ({t['triple'][1]})" for t in pos_rels[:3]]
        )
        if pos_rels
        else "  (none)"
    )
    neg_context = (
        "\n".join(
            [f"  - {t['triple'][0]} → {t['triple'][2]} ({t['triple'][1]})" for t in neg_rels[:3]]
        )
        if neg_rels
        else "  (none)"
    )

    combined_prompt = f"""You are a causal reasoning expert. Analyze the following question and determine the final answer.

ORIGINAL QUESTION:
{question}

CONTEXT INFORMATION:

1. Question Type: {"Meta-level (asking about MORE/LESS phenomenon)" if question_structure.get('is_meta_level') else "Direct causal question"}

2. Key Elements:
   - Intervention/Change: {question_structure.get('intervention', 'N/A')}
   - Target Being Asked About: {question_structure.get('target_phrase', 'N/A')}
   - Direction Mentioned: {question_structure.get('target_direction', 'N/A')}

3. Causal Relations Found:
   
   Positive/Increasing Relations (cause increase):
{pos_context}
   
   Negative/Decreasing Relations (cause decrease):
{neg_context}

REASONING GUIDE:

If this is a META-LEVEL question (asking about "LESS X" or "MORE X"):
- You are reasoning about the PHENOMENON itself, not the entity
- Example: "How will it affect LESS rabbits?"
  * You're asking about the phenomenon of "having fewer rabbits"
  * If intervention causes rabbits to decrease → "LESS rabbits" phenomenon INCREASES → answer: MORE
  * If intervention causes rabbits to increase → "LESS rabbits" phenomenon DECREASES → answer: LESS

If this is a DIRECT question:
- Simply determine if the intervention increases, decreases, or doesn't affect the target

TASK:
Provide your final answer with complete reasoning.

Return JSON format:
{{
  "final_answer": "more|less|no_effect",
  "rationale": "Complete step-by-step explanation of your reasoning (3-5 sentences)",
  "confidence": 0.0-1.0
}}

Return ONLY valid JSON."""

    try:
        response = ollama.generate(model=model, prompt=combined_prompt)
        llm_response = response.get("response", "").strip()

        json_match = re.search(r"\{[^}]+\}", llm_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            final_answer = _normalize_label(result.get("final_answer", ""))
            rationale = result.get("rationale", "")
            confidence = result.get("confidence", 0.0)
        else:
            final_answer = None
            rationale = llm_response
            confidence = 0.0
            for label in ["more", "less", "no_effect"]:
                if label in llm_response.lower():
                    final_answer = label
                    break

        return {
            "method": "combined_context_llm",
            "question_structure": question_structure,
            "final_answer": final_answer,
            "rationale": rationale,
            "confidence": confidence,
            "raw_response": llm_response,
        }

    except Exception as e:
        return {
            "method": "combined_context_llm",
            "question_structure": question_structure,
            "final_answer": "error",
            "rationale": str(e),
            "confidence": 0.0,
        }


__all__ = [
    "predict_meta_informed_llm",
    "predict_combined_context_llm",
    "predict_meta_informed_llm_reflective",
    "predict_combined_context_llm_reflective",
]


# ===================== Reflective variants (quiet) =====================

def predict_meta_informed_llm_reflective(
    question: str,
    parser,
    builder,
    model: str,
) -> Dict[str, Any]:
    with suppress_output():
        question_structure = parser.parse_question_structure(question)
    with suppress_output():
        builder_result = builder.build_causal_chain(question)
    triples = builder.get_all_triples(builder_result, format="structured")
    pos_context, neg_context = _summarize_relations(triples)

    analysis_prompt = f"""Analyze the causal effect and produce a draft.

Question: {question}
Parsed Structure:
{json.dumps(question_structure)}

Return ONLY JSON:
{{
  "entity_effect": "increase|decrease|no_effect",
  "label_guess": "more|less|no_effect",
  "rationale": "2-3 concise sentences"
}}"""

    a_resp = ollama.generate(model=model, prompt=analysis_prompt)
    a_text = a_resp.get("response", "").strip()
    a_json = {}
    try:
        m = re.search(r"\{[^}]+\}", a_text, re.DOTALL)
        if m:
            a_json = json.loads(m.group())
    except Exception:
        a_json = {}

    entity_effect_raw = a_json.get("entity_effect")
    label_guess_raw = a_json.get("label_guess")
    rationale = a_json.get("rationale", a_text)

    reflection_prompt = f"""Reflect using rules and evidence.

Question: {question}
Structure: {json.dumps(question_structure)}

Positive (increase):
{pos_context}

Negative (decrease):
{neg_context}

Draft: {json.dumps(a_json)}

Rules:
- Map entity_effect increase->more, decrease->less, no_effect->no_effect
- If question_type == "meta" and target_direction == "less": invert entity label (more<->less)
- If question_type == "meta" and target_direction == "more": keep entity label
- If direct: keep entity label

No-Effect checklist (prefer \"no_effect\" if ANY holds):
- No plausible direct or 2-hop causal path between intervention and target
- Both positive and negative evidence are weak or cancel out (difference within margin)
- Your own confidence is low (< {CONFIDENCE_GATE:.2f}) or analysis is contradictory

Return ONLY JSON:
{{
  "computed_label": "more|less|no_effect",
  "final_label": "more|less|no_effect",
  "correction_reason": "...",
  "confidence": 0.0-1.0
}}"""

    r_resp = ollama.generate(model=model, prompt=reflection_prompt)
    r_text = r_resp.get("response", "").strip()
    r_json = {}
    try:
        m = re.search(r"\{[^}]+\}", r_text, re.DOTALL)
        if m:
            r_json = json.loads(m.group())
    except Exception:
        r_json = {}

    causal_decision = _normalize_dir(entity_effect_raw) or _normalize_label(label_guess_raw)
    if not causal_decision:
        causal_decision = _normalize_label(r_json.get("computed_label")) or "no_effect"
    _, program_final = parser.should_invert_answer(question_structure, causal_decision)

    model_final = _normalize_label(r_json.get("final_label")) or _normalize_label(r_json.get("computed_label"))
    corrected = False
    correction_source = "none"

    # Additional no-effect guardrail
    triples_struct = triples
    has_path = _has_path_within_k(triples_struct, question_structure.get("intervention"), question_structure.get("target_entity"), PATH_HOPS)
    pos_sum, neg_sum, pos_strong, neg_strong = _directional_evidence_stats(triples_struct, SUPPORT_THRESHOLD)
    try:
        r_conf = float(r_json.get("confidence", 0.0))
    except Exception:
        r_conf = 0.0
    trigger_no_effect = False
    if not has_path:
        trigger_no_effect = True
    elif (pos_strong + neg_strong) < MIN_DIR_SUPPORTS and max(pos_sum, neg_sum) < MIN_EVIDENCE_SUM:
        trigger_no_effect = True
    elif r_conf < CONFIDENCE_GATE and (_normalize_label(r_json.get("computed_label")) != _normalize_label(label_guess_raw)):
        trigger_no_effect = True
    if trigger_no_effect:
        final_answer = "no_effect"
        corrected = (model_final != "no_effect")
        correction_source = "no_effect_guardrail"
    else:
        if model_final != program_final:
            corrected = True
            correction_source = "guardrail"
            final_answer = program_final
        else:
            final_answer = model_final or program_final

    return {
        "method": "meta_informed_llm_reflective",
        "question_structure": question_structure,
        "entity_effect": entity_effect_raw,
        "draft_label": _normalize_label(label_guess_raw),
        "computed_label": _normalize_label(r_json.get("computed_label")),
        "final_answer": final_answer,
        "corrected": corrected,
        "correction_source": correction_source,
        "rationale": rationale,
        "raw": {"analysis": a_text, "reflection": r_text},
    }


def predict_combined_context_llm_reflective(
    question: str,
    parser,
    builder,
    model: str,
) -> Dict[str, Any]:
    with suppress_output():
        question_structure = parser.parse_question_structure(question)
    with suppress_output():
        builder_result = builder.build_causal_chain(question)
    triples = builder.get_all_triples(builder_result, format="structured")
    pos_context, neg_context = _summarize_relations(triples)

    analysis_prompt = f"""Draft analysis for the question.

Question: {question}
Structure: {json.dumps(question_structure)}

Return ONLY JSON:
{{
  "entity_effect": "increase|decrease|no_effect",
  "label_guess": "more|less|no_effect",
  "rationale": "2-3 concise sentences"
}}"""

    a_resp = ollama.generate(model=model, prompt=analysis_prompt)
    a_text = a_resp.get("response", "").strip()
    a_json = {}
    try:
        m = re.search(r"\{[^}]+\}", a_text, re.DOTALL)
        if m:
            a_json = json.loads(m.group())
    except Exception:
        a_json = {}

    entity_effect_raw = a_json.get("entity_effect")
    label_guess_raw = a_json.get("label_guess")
    rationale = a_json.get("rationale", a_text)

    reflect_prompt = f"""Reflect and finalize with rules + evidence.

Question: {question}
Structure: {json.dumps(question_structure)}

Positive (increase):
{pos_context}

Negative (decrease):
{neg_context}

Draft: {json.dumps(a_json)}

Rules:
- Map entity_effect increase->more, decrease->less, no_effect->no_effect
- If question_type == "meta" and target_direction == "less": invert entity label (more<->less)
- If question_type == "meta" and target_direction == "more": keep entity label
 - If direct: keep entity label

 No-Effect checklist (prefer \"no_effect\" if ANY holds):
 - No plausible direct or 2-hop causal path between intervention and target
 - Both positive and negative evidence are weak or cancel out (difference within margin)
 - Your own confidence is low (< {CONFIDENCE_GATE:.2f}) or analysis is contradictory

 Return ONLY JSON:
{{
  "computed_label": "more|less|no_effect",
  "final_label": "more|less|no_effect",
  "correction_reason": "...",
  "confidence": 0.0-1.0
}}"""

    r_resp = ollama.generate(model=model, prompt=reflect_prompt)
    r_text = r_resp.get("response", "").strip()
    r_json = {}
    try:
        m = re.search(r"\{[^}]+\}", r_text, re.DOTALL)
        if m:
            r_json = json.loads(m.group())
    except Exception:
        r_json = {}

    causal_decision = _normalize_dir(entity_effect_raw) or _normalize_label(label_guess_raw)
    if not causal_decision:
        causal_decision = _normalize_label(r_json.get("computed_label")) or "no_effect"
    _, program_final = parser.should_invert_answer(question_structure, causal_decision)
    model_final = _normalize_label(r_json.get("final_label")) or _normalize_label(r_json.get("computed_label"))
    corrected = False
    correction_source = "none"
    # Additional no-effect guardrail
    triples_struct = triples
    has_path = _has_path_within_k(triples_struct, question_structure.get("intervention"), question_structure.get("target_entity"), PATH_HOPS)
    pos_sum, neg_sum, pos_strong, neg_strong = _directional_evidence_stats(triples_struct, SUPPORT_THRESHOLD)
    try:
        r_conf = float(r_json.get("confidence", 0.0))
    except Exception:
        r_conf = 0.0
    trigger_no_effect = False
    if not has_path:
        trigger_no_effect = True
    elif (pos_strong + neg_strong) < MIN_DIR_SUPPORTS and max(pos_sum, neg_sum) < MIN_EVIDENCE_SUM:
        trigger_no_effect = True
    elif r_conf < CONFIDENCE_GATE and (_normalize_label(r_json.get("computed_label")) != _normalize_label(label_guess_raw)):
        trigger_no_effect = True
    if trigger_no_effect:
        final_answer = "no_effect"
        corrected = (model_final != "no_effect")
        correction_source = "no_effect_guardrail"
    else:
        if model_final != program_final:
            corrected = True
            correction_source = "guardrail"
            final_answer = program_final
        else:
            final_answer = model_final or program_final

    return {
        "method": "combined_context_llm_reflective",
        "question_structure": question_structure,
        "entity_effect": entity_effect_raw,
        "draft_label": _normalize_label(label_guess_raw),
        "computed_label": _normalize_label(r_json.get("computed_label")),
        "final_answer": final_answer,
        "corrected": corrected,
        "correction_source": correction_source,
        "rationale": rationale,
        "raw": {"analysis": a_text, "reflection": r_text},
    }
