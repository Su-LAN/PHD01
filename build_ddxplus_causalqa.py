from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ollama  # type: ignore


# ---------------------------------------------------------------------------
# NOTE
# ---------------------------------------------------------------------------
# This script builds a WIQA-style 3-way QA dataset ("more/less/no effect") from
# DDXPlus. It supports:
#   - LLM-only generation (both question + label from LLM)
#   - Data-grounded labels: label is decided by observational statistics from
#     DDXPlus patient files, while the question stem is generated from a chosen
#     evidence description.


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
BASE_DIR = THIS_DIR / "Dataset" / "DDXPlus" / "22687585"

CONDITION_FILE = BASE_DIR / "release_conditions.json"
EVIDENCE_FILE = BASE_DIR / "release_evidences.json"

TRAIN_PATIENTS_FILE = BASE_DIR / "release_train_patients"
VALIDATE_PATIENTS_FILE = BASE_DIR / "release_validate_patients"
TEST_PATIENTS_FILE = BASE_DIR / "release_test_patients"

# 输出文件：WIQA 风格的 DDXPlus CausalQA 数据集
OUTPUT_FILE = THIS_DIR / "DDXPlus_CausalQA.jsonl"

# 固定选项，和 WIQA 保持一致
CHOICES_TEXT = ["more", "less", "no effect"]
CHOICES_LABEL = ["A", "B", "C"]

ANSWER_TO_CHOICE = {"more": "A", "less": "B", "no effect": "C"}


# ---------------------------------------------------------------------------
# Load DDXPlus conditions (diseases)
# ---------------------------------------------------------------------------


def load_conditions() -> Dict[str, dict]:
    """
    Load condition (disease) definitions from DDXPlus.

    Returns a dict:
        { condition_name (c['condition_name']) : c-dict }
    """
    with CONDITION_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Already keyed by condition_name (e.g. "Anemia")
        return data

    cond_map: Dict[str, dict] = {}
    for c in data:
        name = c.get("condition_name") or c.get("cond-name-eng")
        if not name:
            continue
        cond_map[name] = c
    return cond_map


CONDITION_MAP: Dict[str, dict] = load_conditions()


def get_disease_label(disease: str) -> str:
    """
    Human-friendly disease label.

    Preference: cond-name-eng > condition_name > dict key.
    """
    cond = CONDITION_MAP.get(disease, {})
    return (
        cond.get("cond-name-eng")
        or cond.get("condition_name")
        or disease
    )


# ---------------------------------------------------------------------------
# Load DDXPlus evidences (E_XXX -> question text / metadata)
# ---------------------------------------------------------------------------


def load_evidences() -> Dict[str, dict]:
    with EVIDENCE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("release_evidences.json is expected to be a dict keyed by evidence code.")
    return data


EVIDENCE_MAP: Dict[str, dict] = load_evidences()


def get_evidence_question_en(evidence_code: str) -> str:
    meta = EVIDENCE_MAP.get(evidence_code, {})
    return str(meta.get("question_en") or evidence_code).strip()


def is_boolean_evidence(evidence_code: str) -> bool:
    meta = EVIDENCE_MAP.get(evidence_code, {})
    return str(meta.get("data_type") or "").strip().upper() == "B"


def is_antecedent_evidence(evidence_code: str) -> bool:
    meta = EVIDENCE_MAP.get(evidence_code, {})
    return bool(meta.get("is_antecedent"))


# ---------------------------------------------------------------------------
# Evidence statistics from patient files (observational association, not do())
# ---------------------------------------------------------------------------


EVIDENCE_CODE_RE = re.compile(r"E_\d+")


@dataclass(frozen=True)
class EvidenceStats:
    total_patients: int
    disease_counts: Dict[str, int]
    evidence_counts: Dict[str, int]
    disease_evidence_counts: Dict[str, Dict[str, int]]


def _parse_evidence_codes(evidences_field: str) -> List[str]:
    """
    Fast extraction of base evidence codes from a row's EVIDENCES string.

    Example field:
        "['E_48', 'E_54_@_V_161', 'E_204_@_V_10']"
    We extract: ["E_48", "E_54", "E_204"] (dedup is done by caller).
    """
    if not evidences_field:
        return []
    return EVIDENCE_CODE_RE.findall(evidences_field)


def compute_evidence_stats(patient_files: Sequence[Path]) -> EvidenceStats:
    """
    One-pass counter:
      - nD:   disease_counts[D]
      - nE:   evidence_counts[E]
      - nDE:  disease_evidence_counts[D][E]

    Notes:
      - Evidence codes are deduplicated per patient row.
      - This is purely observational (association), not causal.
    """
    total = 0
    disease_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    disease_evidence_counts: Dict[str, Counter[str]] = {}

    for file_path in patient_files:
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                disease = str(row.get("PATHOLOGY") or "").strip()
                if not disease:
                    continue
                disease_counts[disease] += 1

                ev_set = set(_parse_evidence_codes(str(row.get("EVIDENCES") or "")))
                if not ev_set:
                    continue

                evidence_counts.update(ev_set)

                if disease not in disease_evidence_counts:
                    disease_evidence_counts[disease] = Counter()
                disease_evidence_counts[disease].update(ev_set)

    return EvidenceStats(
        total_patients=total,
        disease_counts=dict(disease_counts),
        evidence_counts=dict(evidence_counts),
        disease_evidence_counts={k: dict(v) for k, v in disease_evidence_counts.items()},
    )


def save_evidence_stats(stats: EvidenceStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_patients": stats.total_patients,
        "disease_counts": stats.disease_counts,
        "evidence_counts": stats.evidence_counts,
        "disease_evidence_counts": stats.disease_evidence_counts,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_evidence_stats(path: Path) -> EvidenceStats:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return EvidenceStats(
        total_patients=int(payload["total_patients"]),
        disease_counts={str(k): int(v) for k, v in payload["disease_counts"].items()},
        evidence_counts={str(k): int(v) for k, v in payload["evidence_counts"].items()},
        disease_evidence_counts={
            str(d): {str(e): int(c) for e, c in m.items()}
            for d, m in payload["disease_evidence_counts"].items()
        },
    )


def association_direction(
    stats: EvidenceStats,
    disease: str,
    evidence: str,
    tau: float = 0.005,
    min_evidence_support: int = 200,
) -> Tuple[str, float, float, float, int]:
    """
    Decide label by comparing:
        P(D|E) vs P(D|not E)

    Returns:
        (label, p_d_given_e, p_d_given_not_e, diff, nE)
    """
    N = stats.total_patients
    nD = stats.disease_counts.get(disease, 0)
    nE = stats.evidence_counts.get(evidence, 0)
    nDE = stats.disease_evidence_counts.get(disease, {}).get(evidence, 0)

    if N <= 0 or nE < min_evidence_support or N == nE:
        return ("no effect", 0.0, 0.0, 0.0, nE)

    p_d_given_e = nDE / nE if nE else 0.0
    p_d_given_not_e = (nD - nDE) / (N - nE) if (N - nE) else 0.0
    diff = p_d_given_e - p_d_given_not_e

    if diff > tau:
        label = "more"
    elif diff < -tau:
        label = "less"
    else:
        label = "no effect"

    return (label, p_d_given_e, p_d_given_not_e, diff, nE)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _call_llm(
    prompt: str,
    model_name: str = "gemma2:27b",
    *,
    temperature: float = 0.0,
    seed: int = 42,
    num_predict: int = 1024,
) -> str:
    """
    Thin wrapper around ollama.generate with deterministic-ish settings.
    """
    if ollama is None:
        raise RuntimeError(
            "ollama is not available. Install the 'ollama' Python package and ensure the Ollama "
            "server is running, or run with --no-llm-question / --mode llm disabled."
        )
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": float(temperature),
            "seed": int(seed),
            "num_predict": int(num_predict),
        },
    )
    return response["response"].strip()


def _clean_llm_json(response: str) -> str:
    """
    Strip common Markdown fences to recover raw JSON.
    """
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    return response.strip()


def _normalize_step(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*[-*\u2022]\s+", "", t)
    t = re.sub(r"^\s*(?:step\s*)?\d+\s*[\.\):]\s*", "", t, flags=re.IGNORECASE)
    t = t.strip()
    if len(t) >= 2 and (
        (t[0] == '"' and t[-1] == '"')
        or (t[0] == "'" and t[-1] == "'")
    ):
        t = t[1:-1].strip()
    if not t:
        return ""
    if t[-1] not in ".?!":
        t += "."
    return t[0].upper() + t[1:]


def _parse_steps_from_text(text: str) -> List[str]:
    if not text:
        return []

    cleaned = _clean_llm_json(text)

    # Try JSON first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
            steps = [_normalize_step(str(s)) for s in obj["steps"]]
            return [s for s in steps if s]
        if isinstance(obj, list):
            steps = [_normalize_step(str(s)) for s in obj]
            return [s for s in steps if s]
    except Exception:
        pass

    # Then try pipe-separated.
    if "|" in cleaned:
        parts = [p.strip() for p in cleaned.split("|")]
        steps = [_normalize_step(p) for p in parts]
        return [s for s in steps if s]

    # Fallback: one per line.
    parts = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    steps = [_normalize_step(p) for p in parts]
    return [s for s in steps if s]


def _build_outcome_step(disease_label: str) -> str:
    # Keep the final step neutral: do NOT reveal the direction label (more/less/no effect).
    return _normalize_step(f"This affects the probability of {disease_label}")


def generate_process_steps(
    *,
    disease_label: str,
    evidence_question_en: str,
    evidence_present: bool,
    answer_label: str,
    model_name: str,
    steps_min: int = 3,
    steps_max: int = 5,
    use_llm_steps: bool = True,
    steps_temperature: float = 0.7,
    steps_num_predict: int = 256,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Create WIQA-style process steps (3–5) for the causal question.

    Returns:
        (para_steps, cause_event)
    """
    if steps_min < 3 or steps_max < steps_min:
        raise ValueError(f"Invalid steps range: {steps_min}..{steps_max}")

    condition = _evidence_question_to_patient_condition(
        evidence_question_en=evidence_question_en,
        answer_yes=evidence_present,
    )
    if condition:
        cause_event = condition.strip()
    else:
        q = evidence_question_en.strip().rstrip("?")
        ans = "YES" if evidence_present else "NO"
        cause_event = f'the patient answers {ans} to "{q}"'

    n_steps = random.randint(steps_min, steps_max)
    first = _normalize_step(cause_event)
    last = _build_outcome_step(disease_label)

    mid_needed = max(0, n_steps - 2)
    mids: List[str] = []

    if use_llm_steps and mid_needed > 0:
        prompt = f"""
You are writing a short causal process chain in medical context.

Start condition (Step 1): {first}
Outcome: {disease_label}
Direction: {answer_label} (more=more likely, less=less likely)

Write EXACTLY {mid_needed} intermediate steps that plausibly connect Step 1 to the outcome.
Rules:
- Each step must be ONE short English sentence.
- Do NOT number the steps.
- Do NOT include the final conclusion; I will add it.
- Do NOT mention the outcome disease name ("{disease_label}").
- Do NOT explicitly state the direction for the outcome (avoid phrases like
  "more likely", "less likely", "increase risk", "decrease risk", "higher probability",
  "lower probability", "more", "less", "no effect").
- Separate steps with the '|' character.
- Output ONLY the steps (no extra text).
""".strip()
        try:
            raw = _call_llm(
                prompt,
                model_name=model_name,
                temperature=steps_temperature,
                seed=seed,
                num_predict=steps_num_predict,
            )
            mids = _parse_steps_from_text(raw)[:mid_needed]
        except Exception:
            mids = []

    # Template fallback / padding.
    template_pool = [
        "This triggers downstream physiological changes",
        "This affects the patient's immune and inflammatory responses",
        "This changes organ function and clinical stability",
        "This alters susceptibility to related complications",
        f"This shifts risk factors linked to {disease_label}",
    ]
    while len(mids) < mid_needed:
        mids.append(_normalize_step(random.choice(template_pool)))

    steps = [first] + mids[:mid_needed] + [last]
    para_steps = "|".join(steps)
    return para_steps, cause_event


# ---------------------------------------------------------------------------
# WIQA-style QA generation for a single disease
# ---------------------------------------------------------------------------


def generate_wiqa_style_qa_for_disease(
    disease_key: str,
    model_name: str = "gemma2:27b",
) -> Optional[dict]:
    """
    调用 LLM，为一个疾病生成一条 WIQA 风格 QA。

    目标结构示例：
        {
            "question_stem": "...",
            "answer_label": "more" | "less" | "no effect",
            "answer_label_as_choice": "A" | "B" | "C",
            "choices": {
                "text": ["more", "less", "no effect"],
                "label": ["A", "B", "C"]
            }
        }
    """
    disease_label = get_disease_label(disease_key)

    prompt = f"""
You are a medical QA writer.

Task:
Create ONE multiple-choice causal question about how a change in a patient's condition
affects the rate or probability of the disease: "{disease_label}".

Requirements:
- The question must be in the form of a single English sentence, such as:
  "Suppose the patient's fever increases to above 39.5°C. How will this affect the pneumonia rate?"
- The answer is about the direction of effect on the disease:
  - "more"      = increases the rate / probability of the disease
  - "less"      = decreases the rate / probability of the disease
  - "no effect" = no clear causal effect on the disease
- Use medically reasonable, general-knowledge assumptions.
- The question must mention "{disease_label}" explicitly.

Output format:
Return ONLY a JSON object with EXACTLY these fields:

{{
  "question_stem": "<the question sentence>",
  "answer_label": "<one of: more, less, no effect>",
  "answer_label_as_choice": "<A, B, or C>",
  "choices": {{
    "text": ["more", "less", "no effect"],
    "label": ["A", "B", "C"]
  }}
}}

Rules:
- Do NOT add any extra fields or explanations.
- "answer_label" must be exactly one of: "more", "less", "no effect".
- "answer_label_as_choice" must match the index in choices.text:
    index 0 -> "A", index 1 -> "B", index 2 -> "C".
- The question should be clear, grammatical English.
"""

    raw = _call_llm(prompt, model_name=model_name)
    cleaned = _clean_llm_json(raw)

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[LLM JSON parse error] disease={disease_key}")
        print(cleaned[:300])
        return None

    question = str(obj.get("question_stem") or "").strip()
    answer_label = str(obj.get("answer_label") or "").strip()
    answer_choice = str(obj.get("answer_label_as_choice") or "").strip()

    if not question:
        return None
    if answer_label not in ("more", "less", "no effect"):
        return None
    if answer_choice not in ("A", "B", "C"):
        return None

    # 强制覆盖 choices，防止 LLM 私自修改
    obj["choices"] = {
        "text": CHOICES_TEXT,
        "label": CHOICES_LABEL,
    }

    return obj


# ---------------------------------------------------------------------------
# Data-grounded QA generation (label from stats, question from evidence)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceCandidate:
    evidence_code: str
    evidence_question_en: str
    is_antecedent: bool
    answer_label: str
    p_d_given_e: float
    p_d_given_not_e: float
    diff: float
    evidence_support: int


def get_condition_evidence_codes(disease_key: str, evidence_pool: str = "both") -> List[str]:
    """
    evidence_pool:
      - "symptoms"     -> only condition['symptoms']
      - "antecedents"  -> only condition['antecedents']
      - "both"         -> union
    """
    cond = CONDITION_MAP.get(disease_key, {})
    symptoms = list((cond.get("symptoms") or {}).keys())
    antecedents = list((cond.get("antecedents") or {}).keys())

    if evidence_pool == "symptoms":
        codes = symptoms
    elif evidence_pool == "antecedents":
        codes = antecedents
    elif evidence_pool == "both":
        codes = symptoms + antecedents
    else:
        raise ValueError(f"Unknown evidence_pool: {evidence_pool}")

    # preserve order, unique
    seen: set[str] = set()
    uniq: List[str] = []
    for c in codes:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def rank_evidence_candidates_for_disease(
    stats: EvidenceStats,
    disease_key: str,
    evidence_pool: str = "both",
    tau: float = 0.005,
    min_evidence_support: int = 200,
    only_boolean: bool = True,
) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    for evidence_code in get_condition_evidence_codes(disease_key, evidence_pool=evidence_pool):
        if evidence_code not in EVIDENCE_MAP:
            continue
        if only_boolean and not is_boolean_evidence(evidence_code):
            continue
        answer_label, p1, p0, diff, nE = association_direction(
            stats,
            disease=disease_key,
            evidence=evidence_code,
            tau=tau,
            min_evidence_support=min_evidence_support,
        )
        candidates.append(
            EvidenceCandidate(
                evidence_code=evidence_code,
                evidence_question_en=get_evidence_question_en(evidence_code),
                is_antecedent=is_antecedent_evidence(evidence_code),
                answer_label=answer_label,
                p_d_given_e=p1,
                p_d_given_not_e=p0,
                diff=diff,
                evidence_support=nE,
            )
        )

    # Prefer strong |diff|; break ties by evidence support.
    candidates.sort(key=lambda c: (abs(c.diff), c.evidence_support), reverse=True)
    return candidates


def _fallback_question_stem(disease_label: str, evidence_question_en: str) -> str:
    # Keep it a single sentence; WIQA-style format.
    q = evidence_question_en.strip().rstrip("?")
    return f'If the patient answers YES to "{q}", how will this affect the probability of {disease_label}?'


def invert_direction(label: str) -> str:
    if label == "more":
        return "less"
    if label == "less":
        return "more"
    return "no effect"


def _format_outcome_probability_phrase(disease_label: str, outcome_polarity: Optional[str] = None) -> str:
    """
    WIQA-style "meta" outcome target.

    Examples:
      - None  -> "the probability of <disease>"
      - more  -> "the more probability of <disease>"
      - less  -> "the less probability of <disease>"
    """
    if not outcome_polarity:
        return f"the probability of {disease_label}"
    pol = str(outcome_polarity).strip().lower()
    if pol not in ("more", "less"):
        raise ValueError(f"Invalid outcome_polarity: {outcome_polarity!r} (expected: more/less)")
    return f"the {pol} probability of {disease_label}"


def _replace_second_person_pronouns(text: str) -> str:
    text = re.sub(r"\byour\b", "the patient's", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou\b", "the patient", text, flags=re.IGNORECASE)
    return text


def _evidence_question_to_patient_condition(evidence_question_en: str, answer_yes: bool) -> Optional[str]:
    """
    Heuristic conversion of an evidence question into a patient-condition statement.

    Example:
      Q: "Do you have a fever (either felt or measured with a thermometer)?"
      YES -> "the patient has a fever (either felt or measured with a thermometer)"
      NO  -> "the patient does not have a fever (either felt or measured with a thermometer)"

    Returns None if no safe conversion is found.
    """
    q = evidence_question_en.strip().strip('"').strip().rstrip("?").strip()
    if not q:
        return None

    patterns: List[Tuple[str, str, str]] = [
        (r"^Do you have (.+)$", "the patient has {rest}", "the patient does not have {rest}"),
        (r"^Do you feel (.+)$", "the patient feels {rest}", "the patient does not feel {rest}"),
        (r"^Do you suffer from (.+)$", "the patient suffers from {rest}", "the patient does not suffer from {rest}"),
        (r"^Do you smoke (.+)$", "the patient smokes {rest}", "the patient does not smoke {rest}"),
        (r"^Do you drink (.+)$", "the patient drinks {rest}", "the patient does not drink {rest}"),
        (r"^Do you take (.+)$", "the patient takes {rest}", "the patient does not take {rest}"),
        (r"^Do you (.+)$", "the patient does {rest}", "the patient does not {rest}"),
        (r"^Have you ever had (.+)$", "the patient has had {rest} before", "the patient has never had {rest} before"),
        (r"^Have you ever been (.+)$", "the patient has been {rest} before", "the patient has never been {rest} before"),
        (r"^Have you ever (.+)$", "the patient has {rest} before", "the patient has never {rest} before"),
        (r"^Have you (.+)$", "the patient has {rest}", "the patient has not {rest}"),
        (r"^Are you unable to (.+)$", "the patient is unable to {rest}", "the patient is able to {rest}"),
        (r"^Are you (.+)$", "the patient is {rest}", "the patient is not {rest}"),
        (r"^Are you currently (.+)$", "the patient is currently {rest}", "the patient is not currently {rest}"),
        (r"^Is your (.+)$", "the patient's {rest}", "the patient's {rest}"),
        (r"^Are your (.+)$", "the patient's {rest}", "the patient's {rest}"),
        (r"^Can you (.+)$", "the patient can {rest}", "the patient cannot {rest}"),
        (r"^Did you (.+)$", "the patient did {rest}", "the patient did not {rest}"),
    ]

    for pat, pos_tpl, neg_tpl in patterns:
        m = re.match(pat, q, flags=re.IGNORECASE)
        if not m:
            continue
        rest = m.group(1).strip()
        rest = _replace_second_person_pronouns(rest)

        # Handle "Is/Are your ..." by splitting into subject + predicate when possible.
        if pat in (r"^Is your (.+)$", r"^Are your (.+)$"):
            parts = rest.split(maxsplit=1)
            if len(parts) != 2:
                return None
            subject, predicate = parts[0], parts[1]
            verb = "is" if pat.startswith("^Is") else "are"
            if answer_yes:
                return f"the patient's {subject} {verb} {predicate}"
            return f"the patient's {subject} {verb} not {predicate}"

        tpl = pos_tpl if answer_yes else neg_tpl
        return tpl.format(rest=rest)

    return None


def _fallback_question_stem_with_polarity(
    disease_label: str,
    evidence_question_en: str,
    evidence_present: bool,
    outcome_polarity: Optional[str] = None,
) -> str:
    # Keep it a single sentence; WIQA-style format.
    outcome_phrase = _format_outcome_probability_phrase(disease_label, outcome_polarity)
    condition = _evidence_question_to_patient_condition(
        evidence_question_en=evidence_question_en,
        answer_yes=evidence_present,
    )
    if condition:
        return f"If {condition}, how will this affect {outcome_phrase}?"

    q = evidence_question_en.strip().rstrip("?")
    ans = "YES" if evidence_present else "NO"
    return f'If the patient answers {ans} to "{q}", how will this affect {outcome_phrase}?'


def generate_question_stem_from_evidence(
    disease_label: str,
    evidence_question_en: str,
    model_name: str = "gemma2:27b",
    use_llm: bool = True,
    evidence_present: bool = True,
    outcome_polarity: Optional[str] = None,
) -> str:
    """
    Generate a single-sentence English question stem that mentions the disease explicitly.
    """
    base = _fallback_question_stem_with_polarity(
        disease_label,
        evidence_question_en,
        evidence_present,
        outcome_polarity=outcome_polarity,
    )
    if not use_llm:
        return base

    try:
        keep_polarity_rule = ""
        if outcome_polarity:
            keep_polarity_rule = f'- MUST keep the word "{outcome_polarity}" to describe the outcome probability.\\n'
        prompt = f"""
You are a medical QA writer.

Rewrite the following question to be fluent, grammatical English, while preserving the exact meaning.

Rules:
- Output ONLY the question sentence.
- MUST keep "{disease_label}" explicitly.
{keep_polarity_rule}- Must be ONE sentence (no line breaks).
- Do not include answer choices.

Question:
{base}
""".strip()
        raw = _call_llm(prompt, model_name=model_name)
        question = raw.strip().splitlines()[0].strip()
    except Exception:
        return base

    if not question.endswith("?"):
        question = question.rstrip(".") + "?"
    if disease_label not in question:
        question = question.rstrip("?") + f" ({disease_label})?"
    if outcome_polarity:
        pol = str(outcome_polarity).strip().lower()
        if pol and re.search(rf"\\b{re.escape(pol)}\\b", question, flags=re.IGNORECASE) is None:
            return base
    return question


def build_data_grounded_causal_qa_dataset(
    num_items: int = 200,
    model_name: str = "gemma2:27b",
    patient_files: Sequence[Path] = (TRAIN_PATIENTS_FILE,),
    evidence_pool: str = "both",
    tau: float = 0.005,
    min_evidence_support: int = 200,
    use_llm_question: bool = True,
    stats_cache_path: Optional[Path] = None,
    recompute_stats: bool = False,
    top_k_per_disease: int = 5,
    balance_more_less: bool = False,
    target_more_ratio: float = 0.5,
    with_steps: bool = False,
    steps_min: int = 3,
    steps_max: int = 5,
    use_llm_steps: bool = True,
    steps_temperature: float = 0.7,
    steps_num_predict: int = 256,
    steps_model_name: Optional[str] = None,
    outcome_meta: bool = False,
    outcome_more_ratio: float = 0.5,
) -> List[dict]:
    """
    Data-grounded builder:
      - Choose an evidence from the disease's (symptoms/antecedents) list.
      - Decide more/less/no effect by stats: P(D|E) vs P(D|not E).
      - Generate question stem from the evidence description.

    The label is observational (association), not do()-causal.
    """
    if stats_cache_path and stats_cache_path.exists() and not recompute_stats:
        stats = load_evidence_stats(stats_cache_path)
    else:
        stats = compute_evidence_stats(patient_files)
        if stats_cache_path:
            save_evidence_stats(stats, stats_cache_path)

    diseases = list(CONDITION_MAP.keys())
    random.shuffle(diseases)

    qa_items: List[dict] = []

    ranked_by_disease: Dict[str, List[EvidenceCandidate]] = {}
    for disease_key in diseases:
        ranked = rank_evidence_candidates_for_disease(
            stats,
            disease_key=disease_key,
            evidence_pool=evidence_pool,
            tau=tau,
            min_evidence_support=min_evidence_support,
            only_boolean=True,
        )
        if ranked:
            ranked_by_disease[disease_key] = ranked

    buildable_diseases = list(ranked_by_disease.keys())
    if not buildable_diseases:
        raise RuntimeError(
            "No buildable diseases found: no boolean evidences matched the requested evidence_pool."
        )

    used_pairs: set[tuple[str, str, bool]] = set()

    def _build_one(
        disease_key: str,
        desired_base_label: Optional[str] = None,
        outcome_polarity: Optional[str] = None,
    ) -> Optional[dict]:
        ranked = ranked_by_disease.get(disease_key)
        if not ranked:
            return None

        top = ranked[: max(1, top_k_per_disease)]

        if balance_more_less:
            top = [c for c in top if c.answer_label in ("more", "less")]
        else:
            # Prefer non-"no effect" when possible.
            non_zero = [c for c in top if c.answer_label != "no effect"]
            top = non_zero or top

        if not top:
            return None

        # Try a few times to avoid duplicates and satisfy desired base label.
        for _ in range(30):
            chosen = random.choice(top)
            base_label = chosen.answer_label

            if desired_base_label is None:
                desired_base_label = base_label
                evidence_present = True
            else:
                if desired_base_label == base_label:
                    evidence_present = True
                elif desired_base_label == invert_direction(base_label):
                    evidence_present = False
                else:
                    continue

            key = (disease_key, chosen.evidence_code, evidence_present)
            if key in used_pairs:
                continue
            used_pairs.add(key)
            break
        else:
            return None

        disease_label = get_disease_label(disease_key)
        if outcome_meta and not outcome_polarity:
            outcome_polarity = "more" if random.random() < outcome_more_ratio else "less"
        question_stem = generate_question_stem_from_evidence(
            disease_label=disease_label,
            evidence_question_en=chosen.evidence_question_en,
            model_name=model_name,
            use_llm=use_llm_question,
            evidence_present=evidence_present,
            outcome_polarity=(outcome_polarity if outcome_meta else None),
        )

        base_answer_label = desired_base_label
        if outcome_meta and outcome_polarity == "less":
            answer_label = invert_direction(base_answer_label)
        else:
            answer_label = base_answer_label
        item: Dict[str, object] = {
            "question_stem": question_stem,
            "answer_label": answer_label,
            "answer_label_as_choice": ANSWER_TO_CHOICE[answer_label],
            "choices": {"text": CHOICES_TEXT, "label": CHOICES_LABEL},
        }
        if outcome_meta:
            item["outcome_polarity"] = outcome_polarity
            item["answer_label_base"] = base_answer_label

        if with_steps:
            steps_model = steps_model_name or model_name
            para_steps, cause_event = generate_process_steps(
                disease_label=disease_label,
                evidence_question_en=chosen.evidence_question_en,
                evidence_present=evidence_present,
                answer_label=base_answer_label,
                model_name=steps_model,
                steps_min=steps_min,
                steps_max=steps_max,
                use_llm_steps=use_llm_steps,
                steps_temperature=steps_temperature,
                steps_num_predict=steps_num_predict,
                seed=random.randint(0, 2**31 - 1),
            )
            item["para_steps"] = para_steps
            item["cause_event"] = cause_event
            item["outcome_base"] = f"{disease_label} probability"

        return item  # type: ignore[return-value]

    if balance_more_less:
        target_more = max(0, min(num_items, int(num_items * target_more_ratio)))
        target_less = num_items - target_more
        desired_final_labels = (["more"] * target_more) + (["less"] * target_less)
        random.shuffle(desired_final_labels)

        for desired_final in desired_final_labels:
            outcome_polarity = None
            desired_base = desired_final
            if outcome_meta:
                outcome_polarity = "more" if random.random() < outcome_more_ratio else "less"
                if outcome_polarity == "less":
                    desired_base = invert_direction(desired_final)
            # Keep sampling diseases until we build an item (or give up).
            for _ in range(200):
                disease = random.choice(buildable_diseases)
                item = _build_one(disease, desired_base_label=desired_base, outcome_polarity=outcome_polarity)
                if item is not None:
                    qa_items.append(item)
                    break
            else:
                raise RuntimeError(
                    f"Unable to build enough items while balancing; missing label: {desired_final}"
                )

        return qa_items[:num_items]

    # Default: try to cover as many different diseases as possible, then repeat.
    for disease in buildable_diseases:
        if len(qa_items) >= num_items:
            break
        item = _build_one(disease)
        if item is not None:
            qa_items.append(item)

    while len(qa_items) < num_items:
        disease = random.choice(buildable_diseases)
        item = _build_one(disease)
        if item is not None:
            qa_items.append(item)

    return qa_items[:num_items]


# ---------------------------------------------------------------------------
# Dataset builder (LLM-only, WIQA style)
# ---------------------------------------------------------------------------


def build_llm_causal_qa_dataset(
    num_items: int = 200,
    model_name: str = "gemma2:27b",
) -> List[dict]:
    """
    使用 LLM 构造 DDXPlus 的 WIQA 风格 CausalQA 数据集。

    - 尽量让每个 QA 对应不同疾病；
    - 如果疾病数量不足或生成失败，会允许重复疾病补足到 num_items。
    """
    diseases = list(CONDITION_MAP.keys())
    random.shuffle(diseases)

    qa_items: List[dict] = []

    # 第一步：每个疾病最多先生成一条，尽量保证疾病多样性
    for disease in diseases:
        if len(qa_items) >= num_items:
            break
        dp = generate_wiqa_style_qa_for_disease(disease, model_name=model_name)
        if dp is not None:
            qa_items.append(dp)

    # 如果还不够 num_items，再允许疾病重复
    if len(qa_items) < num_items:
        remaining = num_items - len(qa_items)
        print(
            f"[Info] Only got {len(qa_items)} unique-disease QAs; "
            f"sampling repeats to reach {num_items}."
        )
        while remaining > 0:
            for disease in diseases:
                if remaining <= 0:
                    break
                dp = generate_wiqa_style_qa_for_disease(disease, model_name=model_name)
                if dp is not None:
                    qa_items.append(dp)
                    remaining -= 1
            # 防止极端情况下死循环
            if len(qa_items) >= num_items:
                break

    return qa_items[:num_items]


def write_jsonl(items: List[dict], path: Path) -> None:
    """Write items to a JSONL file (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def summarize_dataset(items: List[dict]) -> None:
    """Print a short summary of the generated QA dataset."""
    total = len(items)
    print(f"Total QA items: {total}")

    if not items:
        return

    label_counts = Counter(item.get("answer_label", "") for item in items)

    print("\nLabel distribution (answer_label):")
    for label, count in label_counts.most_common():
        pct = count / total * 100.0
        print(f"  {label}: {count} ({pct:.1f}%)")

    if any("outcome_polarity" in item for item in items):
        pol_counts = Counter(item.get("outcome_polarity", "") for item in items)
        print("\nOutcome target distribution (outcome_polarity):")
        for pol, count in pol_counts.most_common():
            pct = count / total * 100.0
            print(f"  {pol}: {count} ({pct:.1f}%)")

    if any("answer_label_base" in item for item in items):
        base_counts = Counter(item.get("answer_label_base", "") for item in items)
        print("\nBase label distribution (answer_label_base):")
        for label, count in base_counts.most_common():
            pct = count / total * 100.0
            print(f"  {label}: {count} ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(num_items: int = 200, model_name: str = "gemma2:27b") -> None:
    random.seed(42)

    print("DDXPlus WIQA-style Causal QA builder")
    print(f"  Condition file: {CONDITION_FILE}")
    print(f"  Evidence file:  {EVIDENCE_FILE}")
    print(f"  Output file:    {OUTPUT_FILE}")
    print(f"  Num items:      {num_items}")
    print(f"  Model:          {model_name}")

    qa_items = build_llm_causal_qa_dataset(num_items=num_items, model_name=model_name)
    write_jsonl(qa_items, OUTPUT_FILE)

    print("\nDataset written to:", OUTPUT_FILE)
    summarize_dataset(qa_items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a WIQA-style CausalQA dataset from DDXPlus.")
    parser.add_argument("--num-items", type=int, default=200)
    parser.add_argument("--model", type=str, default="llama3:8b")
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help="Output JSONL path (default: DDXPlus_CausalQA.jsonl in this directory).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["stats", "llm"],
        default="stats",
        help="stats: label from patient statistics; llm: label from LLM",
    )

    # Stats-mode options
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        choices=["train", "validate", "test"],
        help="Which patient files to use when computing evidence statistics.",
    )
    parser.add_argument("--evidence-pool", type=str, choices=["symptoms", "antecedents", "both"], default="both")
    parser.add_argument("--tau", type=float, default=0.005, help="Effect threshold for mapping to more/less/no effect.")
    parser.add_argument("--min-evidence-support", type=int, default=200)
    parser.add_argument("--top-k-per-disease", type=int, default=5)
    parser.add_argument(
        "--balance-more-less",
        action="store_true",
        help="Force the dataset to contain only 'more' and 'less' with the target ratio (default 50/50).",
    )
    parser.add_argument("--target-more-ratio", type=float, default=0.5)
    parser.add_argument(
        "--outcome-meta",
        action="store_true",
        help=(
            "Use WIQA-style outcome meta: ask about the MORE/LESS probability of the disease, "
            "and flip the answer label accordingly (reduces shortcut from question wording)."
        ),
    )
    parser.add_argument(
        "--outcome-more-ratio",
        type=float,
        default=0.5,
        help="When --outcome-meta is on, probability of choosing 'more' as the outcome target (default 0.5).",
    )
    parser.add_argument("--no-llm-question", action="store_true", help="Do not use LLM to rewrite question stems.")
    parser.add_argument(
        "--with-steps",
        action="store_true",
        help="Add WIQA-style multi-step `para_steps` (3–5 steps) to each item.",
    )
    parser.add_argument("--steps-min", type=int, default=3)
    parser.add_argument("--steps-max", type=int, default=5)
    parser.add_argument(
        "--no-llm-steps",
        action="store_true",
        help="Do not use LLM to generate intermediate steps (use templates).",
    )
    parser.add_argument("--steps-temperature", type=float, default=0.7)
    parser.add_argument("--steps-num-predict", type=int, default=256)
    parser.add_argument(
        "--steps-model",
        type=str,
        default="",
        help="Model used to generate process steps (default: same as --model).",
    )
    parser.add_argument("--stats-cache", type=str, default=str(THIS_DIR / "ddxplus_evidence_stats.json"))
    parser.add_argument("--recompute-stats", action="store_true")

    args = parser.parse_args()

    random.seed(42)

    print("DDXPlus WIQA-style Causal QA builder")
    print(f"  Mode:           {args.mode}")
    print(f"  Condition file: {CONDITION_FILE}")
    print(f"  Evidence file:  {EVIDENCE_FILE}")
    print(f"  Output file:    {args.output}")
    print(f"  Num items:      {args.num_items}")
    print(f"  Model:          {args.model}")

    if args.mode == "llm":
        if args.balance_more_less or args.target_more_ratio != 0.5:
            print("[Warning] --balance-more-less/--target-more-ratio only apply to --mode stats (ignored in llm mode).")
        if args.outcome_meta or args.outcome_more_ratio != 0.5:
            print("[Warning] --outcome-meta/--outcome-more-ratio only apply to --mode stats (ignored in llm mode).")
        if args.with_steps:
            print("[Warning] --with-steps currently applies only to --mode stats (ignored in llm mode).")
        qa_items = build_llm_causal_qa_dataset(num_items=args.num_items, model_name=args.model)
    else:
        split_to_path = {
            "train": TRAIN_PATIENTS_FILE,
            "validate": VALIDATE_PATIENTS_FILE,
            "test": TEST_PATIENTS_FILE,
        }
        patient_files = [split_to_path[s] for s in args.splits]
        qa_items = build_data_grounded_causal_qa_dataset(
            num_items=args.num_items,
            model_name=args.model,
            patient_files=patient_files,
            evidence_pool=args.evidence_pool,
            tau=args.tau,
            min_evidence_support=args.min_evidence_support,
            use_llm_question=not args.no_llm_question,
            stats_cache_path=Path(args.stats_cache),
            recompute_stats=args.recompute_stats,
            top_k_per_disease=args.top_k_per_disease,
            balance_more_less=args.balance_more_less,
            target_more_ratio=args.target_more_ratio,
            with_steps=args.with_steps,
            steps_min=args.steps_min,
            steps_max=args.steps_max,
            use_llm_steps=not args.no_llm_steps,
            steps_temperature=args.steps_temperature,
            steps_num_predict=args.steps_num_predict,
            steps_model_name=(args.steps_model.strip() or None),
            outcome_meta=args.outcome_meta,
            outcome_more_ratio=args.outcome_more_ratio,
        )

    out_path = Path(args.output)
    write_jsonl(qa_items, out_path)
    print("\nDataset written to:", out_path)
    summarize_dataset(qa_items)
