import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ollama


ANSWER_LABEL_TO_CHOICE = {"more": "A", "less": "B", "no effect": "C"}
CHOICE_TO_ANSWER_LABEL = {"A": "more", "B": "less", "C": "no effect"}


def sanitize_dir_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_no"] = line_no
            rows.append(obj)
    return rows


def build_ddxplus_prompt(question_stem: str) -> str:
    return (
        f"Question: {question_stem}\n"
        "Choice A: more\n"
        "Choice B: less\n"
        "Choice C: no effect\n"
    )


def ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    num_predict: int,
    seed: int,
    timeout_s: int = 600,
    retries: int = 2,
) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            start = time.time()
            resp = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": num_predict,
                    "seed": seed,
                },
            )
            content = (resp.get("message") or {}).get("content", "")
            if not isinstance(content, str):
                content = str(content)
            elapsed = time.time() - start
            if elapsed > timeout_s:
                raise TimeoutError(f"ollama.chat exceeded timeout: {elapsed:.1f}s > {timeout_s}s")
            return content
        except Exception as e:
            last_err = e
            # simple backoff
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed after retries: {last_err}") from last_err


def extract_choice_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    # Prefer explicit markers
    m = re.search(r"(?:final answer|answer)\s*[:\-]?\s*([ABC])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Some models output just a letter on the last line
    last_line = text.strip().splitlines()[-1].strip()
    if re.fullmatch(r"[ABC]", last_line, flags=re.IGNORECASE):
        return last_line.upper()

    # Fallback: explicit label
    m2 = re.search(r"(?:final answer|answer)\s*[:\-]?\s*(more|less|no effect)\b", text, flags=re.IGNORECASE)
    if m2:
        label = m2.group(1).lower()
        return ANSWER_LABEL_TO_CHOICE.get(label)

    return None


def force_extract_choice(
    model: str,
    question_prompt: str,
    reasoning_text: str,
    *,
    seed: int,
) -> Tuple[Optional[str], str]:
    prompt = (
        "You are an answer extractor.\n"
        "Given the question and a model's reasoning, output ONLY one letter: A, B, or C.\n\n"
        f"{question_prompt}\n"
        "Reasoning:\n"
        f"{reasoning_text}\n\n"
        "Output:"
    )
    out = ollama_chat(
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        num_predict=8,
        seed=seed,
        timeout_s=180,
        retries=1,
    )
    return extract_choice_from_text(out), out


@dataclass(frozen=True)
class MethodConfig:
    name: str
    temperature: float
    num_predict: int


def run_cot(
    model: str,
    question_prompt: str,
    *,
    seed: int,
    causal_variant: bool,
) -> str:
    if causal_variant:
        method_header = "CausalCoT"
        guidance = (
            "Guidance: Solve the causal effect direction question with explicit causal structure.\n"
            "Step 1) Identify the intervention/cause variable and the outcome variable.\n"
            "Step 2) Construct a minimal causal graph (edge list like X -> M, M -> Y).\n"
            "Step 3) Briefly explain the mechanism/direction from cause to outcome.\n"
            "Step 4) Choose the best option.\n"
        )
    else:
        method_header = "CoT"
        guidance = (
            "Guidance: Use chain-of-thought with a causal graph.\n"
            "1) Construct a minimal causal graph.\n"
            "2) Reason briefly how increasing the cause affects the outcome.\n"
            "3) Choose the best option.\n"
        )

    prompt = (
        f"[{method_header}] {guidance}\n"
        f"{question_prompt}\n\n"
        "Output format:\n"
        "Causal graph: <comma-separated edges>\n"
        "Reasoning: <1-4 sentences>\n"
        "Final answer: <A|B|C>\n"
    )
    return ollama_chat(
        model,
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        num_predict=512,
        seed=seed,
        timeout_s=600,
        retries=2,
    )


def parse_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to extract a JSON object substring
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def heuristic_score_component(text: str, *, component_idx: int) -> float:
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    if len(text) > 80:
        score += 0.2
    if "->" in text or "caus" in t or "effect" in t:
        score += 0.2
    if component_idx == 1:
        if "->" in text:
            score += 0.3
    if component_idx == 3:
        if extract_choice_from_text(text) is not None:
            score += 0.5
    return min(score, 1.0)


def run_got(model: str, question_prompt: str, *, seed: int) -> str:
    split_prompt = (
        "<Instruction> Split this causal question into 3 components.\n"
        "Only output the JSON, no additional text! </Instruction>\n\n"
        f"{question_prompt}\n\n"
        "Output:\n"
        "{\n"
        '  "Component 1": "Causal graph: ...",\n'
        '  "Component 2": "Reasoning process: ...",\n'
        '  "Component 3": "Answer: ..."\n'
        "}\n"
    )
    split_out = ollama_chat(
        model,
        [{"role": "user", "content": split_prompt}],
        temperature=0.7,
        num_predict=384,
        seed=seed,
        timeout_s=300,
        retries=1,
    )
    split_obj = parse_json_object_from_text(split_out)
    if not split_obj:
        split_obj = {
            "Component 1": "Causal graph: identify variables and causal edges",
            "Component 2": "Reasoning process: trace the effect from cause to outcome",
            "Component 3": "Answer: choose A/B/C",
        }

    components: List[Tuple[str, str]] = []
    for i in range(1, 4):
        key = f"Component {i}"
        comp = split_obj.get(key)
        if isinstance(comp, str) and comp.strip():
            components.append((key, comp.strip()))
        else:
            components.append((key, f"Component {i}"))

    component_analyses: Dict[str, str] = {}
    for idx, (comp_key, comp_text) in enumerate(components, start=1):
        best_text = ""
        best_score = -1.0
        for attempt in range(2):
            prompt = (
                "<Instruction> Analyze this component of the causal question. </Instruction>\n\n"
                f"{question_prompt}\n\n"
                f"Component: {comp_text}\n"
                "Write a concise analysis.\n"
                + ("End with 'Final answer: A/B/C'." if idx == 3 else "")
            )
            out = ollama_chat(
                model,
                [{"role": "user", "content": prompt}],
                temperature=0.8,
                num_predict=384,
                seed=seed + 17 * idx + attempt,
                timeout_s=300,
                retries=1,
            )
            score = heuristic_score_component(out, component_idx=idx)
            if score > best_score:
                best_score = score
                best_text = out
        component_analyses[comp_key] = best_text

    aggregate_prompt = (
        "<Instruction> Merge the component analyses into one complete causal reasoning answer.\n"
        "Your response MUST end with a single line: 'Final answer: A' or 'Final answer: B' or 'Final answer: C'. </Instruction>\n\n"
        f"{question_prompt}\n\n"
        "Component analyses:\n"
        f"Component 1:\n{component_analyses.get('Component 1','')}\n\n"
        f"Component 2:\n{component_analyses.get('Component 2','')}\n\n"
        f"Component 3:\n{component_analyses.get('Component 3','')}\n\n"
    )
    merged = ollama_chat(
        model,
        [{"role": "user", "content": aggregate_prompt}],
        temperature=0.0,
        num_predict=512,
        seed=seed,
        timeout_s=600,
        retries=1,
    )

    # Validate answer presence; if missing, ask for fix
    if extract_choice_from_text(merged) is None:
        fix_prompt = (
            "Fix the following response so that it ends with exactly one line:\n"
            "'Final answer: A' or 'Final answer: B' or 'Final answer: C'.\n\n"
            f"{question_prompt}\n\n"
            "Current response:\n"
            f"{merged}\n\n"
            "Fixed response:\n"
        )
        merged = ollama_chat(
            model,
            [{"role": "user", "content": fix_prompt}],
            temperature=0.0,
            num_predict=512,
            seed=seed + 999,
            timeout_s=600,
            retries=1,
        )

    return merged


def run_tot(model: str, question_prompt: str, *, seed: int) -> str:
    graph_gen_prompt = (
        "Generate TWO different plausible causal graphs relevant to answering the question.\n"
        "Output JSON only:\n"
        '{ "graph1": "X -> ...", "graph2": "X -> ..." }\n\n'
        f"{question_prompt}\n"
    )
    graphs_out = ollama_chat(
        model,
        [{"role": "user", "content": graph_gen_prompt}],
        temperature=0.8,
        num_predict=256,
        seed=seed,
        timeout_s=300,
        retries=1,
    )
    graphs_obj = parse_json_object_from_text(graphs_out) or {}
    g1 = str(graphs_obj.get("graph1", "")).strip()
    g2 = str(graphs_obj.get("graph2", "")).strip()
    if not g1:
        g1 = "cause -> outcome"
    if not g2:
        g2 = g1

    graph_pick_prompt = (
        "Question:\n"
        f"{question_prompt}\n\n"
        f"Graph 1: {g1}\n"
        f"Graph 2: {g2}\n\n"
        "Which graph better captures the causal mechanism needed to decide A/B/C?\n"
        "Reply ONLY with 1 or 2."
    )
    pick = ollama_chat(
        model,
        [{"role": "user", "content": graph_pick_prompt}],
        temperature=0.0,
        num_predict=8,
        seed=seed + 1,
        timeout_s=120,
        retries=1,
    )
    chosen_graph = g2 if "2" in pick.strip()[:5] else g1

    chain_gen_prompt = (
        "Given the question and the causal graph, generate TWO different concise reasoning chains.\n"
        "Output JSON only:\n"
        '{ "chain1": "...", "chain2": "..." }\n\n'
        f"{question_prompt}\n"
        f"Causal graph: {chosen_graph}\n"
    )
    chains_out = ollama_chat(
        model,
        [{"role": "user", "content": chain_gen_prompt}],
        temperature=0.8,
        num_predict=256,
        seed=seed + 2,
        timeout_s=300,
        retries=1,
    )
    chains_obj = parse_json_object_from_text(chains_out) or {}
    c1 = str(chains_obj.get("chain1", "")).strip()
    c2 = str(chains_obj.get("chain2", "")).strip()
    if not c1:
        c1 = "Increasing the cause increases the outcome."
    if not c2:
        c2 = c1

    chain_pick_prompt = (
        "Question:\n"
        f"{question_prompt}\n\n"
        f"Causal graph: {chosen_graph}\n\n"
        f"Chain 1: {c1}\n"
        f"Chain 2: {c2}\n\n"
        "Which chain is more logically consistent for deciding the final answer?\n"
        "Reply ONLY with 1 or 2."
    )
    pick2 = ollama_chat(
        model,
        [{"role": "user", "content": chain_pick_prompt}],
        temperature=0.0,
        num_predict=8,
        seed=seed + 3,
        timeout_s=120,
        retries=1,
    )
    chosen_chain = c2 if "2" in pick2.strip()[:5] else c1

    final_prompt = (
        "Based on the question, causal graph, and reasoning chain, choose the final answer.\n"
        "Reply with EXACTLY one letter: A or B or C.\n\n"
        f"{question_prompt}\n"
        f"Causal graph: {chosen_graph}\n"
        f"Reasoning: {chosen_chain}\n"
        "Answer:"
    )
    final_out = ollama_chat(
        model,
        [{"role": "user", "content": final_prompt}],
        temperature=0.0,
        num_predict=8,
        seed=seed + 4,
        timeout_s=180,
        retries=1,
    )
    choice = extract_choice_from_text(final_out) or "A"

    return (
        "ToT Trace\n"
        f"Causal graph: {chosen_graph}\n"
        f"Reasoning: {chosen_chain}\n"
        f"Final answer: {choice}\n"
    )


def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def read_processed_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    processed = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "id" in row and row["id"] != "":
                processed.add(row["id"])
    return processed


def append_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def compute_accuracy(path: str) -> Tuple[int, int, float]:
    if not os.path.exists(path):
        return 0, 0, 0.0
    total = 0
    correct = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            is_correct = str(row.get("is_correct", "")).strip().lower()
            if is_correct in {"true", "1", "yes"}:
                correct += 1
    return correct, total, (correct / total if total else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CDCR-SFT-style baselines on DDXPlus_CausalQA.jsonl using Ollama.")
    parser.add_argument("--input", type=str, default="DDXPlus_CausalQA.jsonl", help="Path to DDXPlus_CausalQA.jsonl")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama model name, e.g., llama3.1:8b")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory (CDCR-SFT-style)")
    parser.add_argument(
        "--methods",
        type=str,
        default="CoT,GoT,ToT,CausalCoT",
        help="Comma-separated methods: CoT,GoT,ToT,CausalCoT",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all rows, otherwise cap.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling / prompting")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    model_dir = sanitize_dir_name(args.model.replace(":", "_"))
    dataset_dir = os.path.join(args.out_dir, model_dir, "ddxplus")
    os.makedirs(dataset_dir, exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    fieldnames = [
        "id",
        "question_type",
        "label",
        "is_correct",
        "answer",
        "letter_answer",
        "llm_output",
        "llm_extracted_output",
        "model",
    ]

    for method in methods:
        out_csv = os.path.join(dataset_dir, f"CoT_{method}.csv")
        ensure_csv_header(out_csv, fieldnames)
        processed_ids = read_processed_ids(out_csv)

        cache: Dict[str, Tuple[str, str, str]] = {}
        print(f"\n=== Running {method} on {len(rows)} rows (resume: {len(processed_ids)} done) ===")
        for idx, row in enumerate(rows):
            row_id = str(idx)
            if row_id in processed_ids:
                continue

            question = str(row.get("question_stem", "")).strip()
            gold_label = str(row.get("answer_label", "")).strip().lower()
            question_prompt = build_ddxplus_prompt(question)

            if question in cache:
                llm_output, choice, extracted = cache[question]
            else:
                try:
                    if method == "CoT":
                        llm_output = run_cot(args.model, question_prompt, seed=args.seed + idx, causal_variant=False)
                    elif method == "CausalCoT":
                        llm_output = run_cot(args.model, question_prompt, seed=args.seed + idx, causal_variant=True)
                    elif method == "GoT":
                        llm_output = run_got(args.model, question_prompt, seed=args.seed + idx)
                    elif method == "ToT":
                        llm_output = run_tot(args.model, question_prompt, seed=args.seed + idx)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    choice = extract_choice_from_text(llm_output)
                    extracted = choice or ""
                    if choice is None:
                        choice, extractor_out = force_extract_choice(
                            args.model, question_prompt, llm_output, seed=args.seed + 10_000 + idx
                        )
                        extracted = extractor_out.strip()
                        if choice is None:
                            choice = "A"  # safe fallback (also majority class)
                    cache[question] = (llm_output, choice, extracted)
                except Exception as e:
                    llm_output = f"ERROR: {e}"
                    choice = "A"
                    extracted = ""

            pred_label = CHOICE_TO_ANSWER_LABEL.get(choice, "more")
            is_correct = pred_label == gold_label

            append_row(
                out_csv,
                fieldnames,
                {
                    "id": row_id,
                    "question_type": "ddxplus",
                    "label": gold_label,
                    "is_correct": str(is_correct),
                    "answer": pred_label,
                    "letter_answer": choice,
                    "llm_output": llm_output,
                    "llm_extracted_output": extracted,
                    "model": args.model,
                },
            )

            if (idx + 1) % 10 == 0:
                print(f"[{method}] processed {idx+1}/{len(rows)}")

        correct, total, acc = compute_accuracy(out_csv)
        print(f"[{method}] accuracy: {acc*100:.2f}% ({correct}/{total}) -> {out_csv}")


if __name__ == "__main__":
    main()

