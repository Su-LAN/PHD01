"""
LLM-based predictors for WIQA-style causal questions.

Includes two methods ported from 01.ipynb:
- predict_meta_informed_llm: Uses parsed meta info from QuestionParser
- predict_combined_context_llm: Uses full context with reasoning guide

Both functions expect a `parser` (QuestionParser) and a `builder`
(EgoExpansionCausalBuilder) instance to be passed in.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Tuple

import ollama
from effect_decider import POSITIVE_RELS, NEGATIVE_RELS

# ======= Tunable guardrail parameters for stronger no_effect behavior =======
SUPPORT_THRESHOLD = 0.25       # confidence threshold to count a triple as strong
MIN_DIR_SUPPORTS = 2           # need at least this many strong directional supports
MIN_EVIDENCE_SUM = 0.25        # minimal directional mass; otherwise prefer no_effect
MARGIN = 1.25                  # not used directly here, retained for prompt guidance
CONFIDENCE_GATE = 0.35         # model reflection confidence gate; under this prefer no_effect
PATH_HOPS = 2                  # require a path within this many hops between intervention and target


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


def predict_meta_informed_llm(
    question: str,
    parser,
    builder,
    model: str,
) -> Dict[str, Any]:
    """
    Method 1: After parsing question structure, pass all meta information to LLM
    and ask it to directly output the final label (more/less/no_effect) with rationale.
    """

    print("=" * 60)
    print("METHOD 1: Meta-Informed LLM Decision")
    print("=" * 60)

    # Step 1: Parse question structure
    question_structure = parser.parse_question_structure(question)

    print("\nQuestion Structure:")
    print(json.dumps(question_structure, indent=2))

    # Step 2: Build causal context (get some triples for context)
    builder_result = builder.build_causal_chain(question)
    triples = builder.get_all_triples(builder_result, format="structured")

    # Extract top few triples as context
    top_triples = triples[:5] if len(triples) >= 5 else triples
    triple_context = "\n".join(
        [
            f"  - {t['triple'][0]} {t['triple'][1]} {t['triple'][2]} (confidence: {t.get('confidence', 0):.2f})"
            for t in top_triples
        ]
    )

    # Step 3: Construct meta-informed prompt
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

Key reasoning steps:
1. If this is a meta-level question (asking about "MORE X" or "LESS X"), you need to reason about the phenomenon itself:
   - "LESS rabbits" means the phenomenon of having fewer rabbits
   - If intervention causes rabbits to decrease, then "LESS rabbits" increases (answer: more)
   - If intervention causes rabbits to increase, then "LESS rabbits" decreases (answer: less)

2. If this is a direct question, just determine the direct causal effect.

3. Consider the causal relations provided as context.

Return your answer in JSON format:
{{
  "final_answer": "more|less|no_effect",
  "rationale": "Brief explanation of your reasoning (2-3 sentences)"
}}

IMPORTANT: Return ONLY valid JSON, nothing else."""

    try:
        response = ollama.generate(model=model, prompt=meta_prompt)
        llm_response = response.get("response", "").strip()

        # Parse JSON response
        json_match = re.search(r"\{[^}]+\}", llm_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            final_answer = _normalize_label(result.get("final_answer", ""))
            rationale = result.get("rationale", "")
        else:
            # Fallback parsing
            final_answer = None
            rationale = llm_response
            for label in ["more", "less", "no_effect"]:
                if label in llm_response.lower():
                    final_answer = label
                    break

        print("\n" + "=" * 60)
        print(f"Final Answer: {final_answer}")
        print(f"Rationale: {rationale}")
        print("=" * 60)

        return {
            "method": "meta_informed_llm",
            "question_structure": question_structure,
            "final_answer": final_answer,
            "rationale": rationale,
            "raw_response": llm_response,
        }

    except Exception as e:
        print(f"Error in meta-informed LLM prediction: {e}")
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
    Method 2: Merge original question with meta reasoning context,
    let LLM give answer in one shot with full explanation.
    """

    print("=" * 60)
    print("METHOD 2: Combined Context LLM Decision")
    print("=" * 60)

    # Step 1: Parse question structure
    question_structure = parser.parse_question_structure(question)

    # Step 2: Build causal context
    builder_result = builder.build_causal_chain(question)
    triples = builder.get_all_triples(builder_result, format="structured")

    # Extract causal chain summary
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

    # Construct combined prompt with full context
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

        # Parse JSON response
        json_match = re.search(r"\{[^}]+\}", llm_response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            final_answer = _normalize_label(result.get("final_answer", ""))
            rationale = result.get("rationale", "")
            confidence = result.get("confidence", 0.0)
        else:
            # Fallback parsing
            final_answer = None
            rationale = llm_response
            confidence = 0.0
            for label in ["more", "less", "no_effect"]:
                if label in llm_response.lower():
                    final_answer = label
                    break

        print("\n" + "=" * 60)
        print(f"Final Answer: {final_answer}")
        print(f"Confidence: {confidence}")
        print(f"Rationale: {rationale}")
        print("=" * 60)

        return {
            "method": "combined_context_llm",
            "question_structure": question_structure,
            "final_answer": final_answer,
            "rationale": rationale,
            "confidence": confidence,
            "raw_response": llm_response,
        }

    except Exception as e:
        print(f"Error in combined context LLM prediction: {e}")
        return {
            "method": "combined_context_llm",
            "question_structure": question_structure,
            "final_answer": "error",
            "rationale": str(e),
            "confidence": 0.0,
        }


# ===================== Reflective variants (verbose) =====================

def predict_meta_informed_llm_reflective(
    question: str,
    parser,
    builder,
    model: str,
) -> Dict[str, Any]:
    print("=" * 60)
    print("METHOD 1R: Meta-Informed LLM (Reflective)")
    print("=" * 60)

    # Stage 0: structure + context
    question_structure = parser.parse_question_structure(question)
    builder_result = builder.build_causal_chain(question)
    triples = builder.get_all_triples(builder_result, format="structured")
    pos_context, neg_context = _summarize_relations(triples)

    print("\nQuestion Structure:")
    print(json.dumps(question_structure, indent=2))

    # Stage A: analysis draft
    analysis_prompt = f"""Analyze the causal effect and produce a draft.

Question: {question}

Parsed Structure:
{json.dumps(question_structure)}

Guidance:
- entity_effect is the direct effect on the entity (increase|decrease|no_effect), do NOT apply meta inversion here.
- label_guess is your first guess at the final label (more|less|no_effect).
- rationale should be concise (2-3 sentences).

Return ONLY JSON:
{{
  "entity_effect": "increase|decrease|no_effect",
  "label_guess": "more|less|no_effect",
  "rationale": "..."
}}"""

    print("\n-- Stage A: analysis draft --")
    a_resp = ollama.generate(model=model, prompt=analysis_prompt)
    a_text = a_resp.get("response", "").strip()
    print(a_text)
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

    # Stage B: reflection
    reflection_prompt = f"""Reflect on your draft and apply meta rules.

Question: {question}

Structure: {json.dumps(question_structure)}

Causal Relations Summary:
Positive (cause increase):
{pos_context}

Negative (cause decrease):
{neg_context}

Your Draft:
{json.dumps(a_json)}

Checklist:
1) Compute computed_label from entity_effect using these rules:
   - Map entity_effect increase->more, decrease->less, no_effect->no_effect (entity-level)
   - If question_type == "meta" and target_direction == "less": invert entity-level label (more<->less)
   - If question_type == "meta" and target_direction == "more": keep entity-level label
   - If direct: keep entity-level label
2) Compare computed_label with label_guess and with rationale; if inconsistent, fix it.
3) Output final_label and a short correction_reason.

No-Effect checklist (prefer "no_effect" if ANY holds):
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

    print("\n-- Stage B: reflection --")
    r_resp = ollama.generate(model=model, prompt=reflection_prompt)
    r_text = r_resp.get("response", "").strip()
    print(r_text)
    r_json = {}
    try:
        m = re.search(r"\{[^}]+\}", r_text, re.DOTALL)
        if m:
            r_json = json.loads(m.group())
    except Exception:
        r_json = {}

    # Programmatic guardrail (meta inversion)
    causal_decision = _normalize_dir(entity_effect_raw) or _normalize_label(label_guess_raw)
    if not causal_decision:
        causal_decision = _normalize_label(r_json.get("computed_label")) or "no_effect"
    _, program_final = parser.should_invert_answer(question_structure, causal_decision)

    model_final = _normalize_label(r_json.get("final_label")) or _normalize_label(r_json.get("computed_label"))
    corrected = False
    correction_source = "none"

    # No-effect guardrail
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

    print("\n" + "=" * 60)
    print(f"Final Answer: {final_answer} (corrected={corrected}, source={correction_source})")
    print(f"Rationale: {rationale}")
    print("=" * 60)

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
    print("=" * 60)
    print("METHOD 2R: Combined Context LLM (Reflective)")
    print("=" * 60)

    # Stage 0: structure + context
    question_structure = parser.parse_question_structure(question)
    builder_result = builder.build_causal_chain(question)
    triples = builder.get_all_triples(builder_result, format="structured")
    pos_context, neg_context = _summarize_relations(triples)

    # Stage A
    analysis_prompt = f"""You are a causal reasoning expert. Produce a draft analysis.

Question: {question}
Structure: {json.dumps(question_structure)}

Return ONLY JSON:
{{
  "entity_effect": "increase|decrease|no_effect",
  "label_guess": "more|less|no_effect",
  "rationale": "2-3 concise sentences"
}}"""

    print("\n-- Stage A: analysis draft --")
    a_resp = ollama.generate(model=model, prompt=analysis_prompt)
    a_text = a_resp.get("response", "").strip()
    print(a_text)
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

    # Stage B: reflection with full context
    combined_reflect = f"""Reflect and finalize the answer using rules and evidence.

Question: {question}
Structure: {json.dumps(question_structure)}

Causal Relations (summary):
Positive (increase):
{pos_context}

Negative (decrease):
{neg_context}

Draft: {json.dumps(a_json)}

Rules (meta inversion):
- entity_effect increase->entity label more; decrease->less; no_effect->no_effect
- If question_type == "meta" and target_direction == "less": invert entity label (more<->less)
- If question_type == "meta" and target_direction == "more": keep entity label
- If direct: keep entity label

No-Effect checklist (prefer "no_effect" if ANY holds):
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

    print("\n-- Stage B: reflection --")
    r_resp = ollama.generate(model=model, prompt=combined_reflect)
    r_text = r_resp.get("response", "").strip()
    print(r_text)
    r_json = {}
    try:
        m = re.search(r"\{[^}]+\}", r_text, re.DOTALL)
        if m:
            r_json = json.loads(m.group())
    except Exception:
        r_json = {}

    # Guardrail (meta inversion first)
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

    print("\n" + "=" * 60)
    print(f"Final Answer: {final_answer} (corrected={corrected}, source={correction_source})")
    print(f"Rationale: {rationale}")
    print("=" * 60)

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


__all__ = [
    "predict_meta_informed_llm",
    "predict_combined_context_llm",
    "predict_meta_informed_llm_reflective",
    "predict_combined_context_llm_reflective",
]
