import argparse
import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ollama


ANSWER_LABEL_TO_CHOICE = {
    "more": "A",
    "less": "B",
    "no_effect": "C",
    "no effect": "C",
    "no_change": "C",
    "no change": "C",
}
CHOICE_TO_ANSWER_LABEL = {"A": "more", "B": "less", "C": "no_effect"}


def sanitize_dir_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def normalize_answer_label(label: str) -> str:
    s = str(label).strip().lower().replace(" ", "_")
    if s == "no_effect" or s == "no_change":
        return "no_effect"
    return s


def normalize_choice_text(choice_text: str) -> str:
    s = str(choice_text).strip().lower().replace("_", " ")
    if s == "no change":
        return "no effect"
    return s


def extract_choice_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    cleaned = text.strip().strip('"').strip("'").strip()
    if re.fullmatch(r"[ABC]", cleaned, flags=re.IGNORECASE):
        return cleaned.upper()

    m = re.search(r"(?:final answer|answer)\s*[:\-]?\s*([ABC])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    last_line = ""
    for line in reversed(text.splitlines()):
        if line.strip():
            last_line = line.strip()
            break
    if last_line and re.fullmatch(r"[ABC]", last_line, flags=re.IGNORECASE):
        return last_line.upper()

    m2 = re.search(
        r"(?:final answer|answer)\s*[:\-]?\s*(more|less|no[_ ]effect|no[_ ]change)\b",
        text,
        flags=re.IGNORECASE,
    )
    if m2:
        label = m2.group(1).lower().replace(" ", "_")
        return ANSWER_LABEL_TO_CHOICE.get(label)

    return None


def stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.md5(f"{base_seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float,
    num_predict: int,
    seed: int,
    timeout_s: int,
    retries: int,
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
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Ollama call failed after retries: {last_err}") from last_err


def force_extract_choice(model: str, prompt: str, llm_output: str, *, seed: int, timeout_s: int) -> Tuple[Optional[str], str]:
    extractor_prompt = (
        "Given the question and a model output, output ONLY one letter: A, B, or C.\n\n"
        f"{prompt}\n\n"
        "Model output:\n"
        f"{llm_output}\n\n"
        "Output:"
    )
    out = ollama_chat(
        model,
        [{"role": "user", "content": extractor_prompt}],
        temperature=0.0,
        num_predict=8,
        seed=seed,
        timeout_s=timeout_s,
        retries=1,
    )
    return extract_choice_from_text(out), out


def ensure_csv_header(path: Path, fieldnames: Sequence[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()


def read_processed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    processed: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id", "")).strip()
            if rid:
                processed.add(rid)
    return processed


@dataclass(frozen=True)
class Datapoint:
    dataset: str
    id: str
    question_stem: str
    improved_question: str
    answer_label: str
    answer_choice: str
    choices_text: Tuple[str, str, str]
    extra: Dict[str, Any]


def load_jsonl_datapoints(path: Path, *, dataset: str, id_prefix: str = "") -> List[Datapoint]:
    rows: List[Datapoint] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            raw_id = str(obj.get("id") or obj.get("_id") or f"{line_no}")
            rid = f"{id_prefix}{raw_id}" if id_prefix else raw_id

            q = str(obj.get("question_stem", "")).strip()
            gold_label = normalize_answer_label(obj.get("answer_label", ""))
            gold_choice = str(obj.get("answer_label_as_choice") or ANSWER_LABEL_TO_CHOICE.get(gold_label, "")).strip().upper()

            choices_obj = obj.get("choices") or {}
            texts = choices_obj.get("text") if isinstance(choices_obj, dict) else None
            if not (isinstance(texts, list) and len(texts) >= 3):
                texts = ["more", "less", "no_effect"]
            a, b, c = (normalize_choice_text(texts[0]), normalize_choice_text(texts[1]), normalize_choice_text(texts[2]))

            rows.append(
                Datapoint(
                    dataset=dataset,
                    id=rid,
                    question_stem=q,
                    improved_question=str(obj.get("improved_question", "")).strip(),
                    answer_label=gold_label,
                    answer_choice=gold_choice,
                    choices_text=(a, b, c),
                    extra={k: v for k, v in obj.items() if k not in {"question_stem", "answer_label", "answer_label_as_choice", "choices"}},
                )
            )
    return rows


def load_wiqa_csv_datapoints(path: Path, *, dataset: str, use_improved_question: bool, id_prefix: str = "") -> List[Datapoint]:
    rows: List[Datapoint] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            raw_id = str(row.get("id") or idx)
            rid = f"{id_prefix}{raw_id}" if id_prefix else raw_id

            question_stem = str(row.get("question_stem", "")).strip()
            improved = str(row.get("improved_question", "")).strip()
            question_for_prompt = improved if (use_improved_question and improved) else question_stem

            gold_label = normalize_answer_label(row.get("answer_label", ""))
            gold_choice = str(row.get("answer_label_as_choice") or ANSWER_LABEL_TO_CHOICE.get(gold_label, "")).strip().upper()

            a = normalize_choice_text(row.get("choice_A", "more"))
            b = normalize_choice_text(row.get("choice_B", "less"))
            c = normalize_choice_text(row.get("choice_C", "no effect"))

            rows.append(
                Datapoint(
                    dataset=dataset,
                    id=rid,
                    question_stem=question_for_prompt,
                    improved_question=improved,
                    answer_label=gold_label,
                    answer_choice=gold_choice,
                    choices_text=(a, b, c),
                    extra={k: v for k, v in row.items() if k not in {"id", "question_stem", "answer_label", "answer_label_as_choice", "choice_A", "choice_B", "choice_C", "improved_question"}},
                )
            )
    return rows


def build_direct_prompt(question_stem: str, choices_text: Tuple[str, str, str]) -> str:
    a, b, c = choices_text
    return (
        "Answer the question.\n"
        "Reply with ONLY one letter: A, B, or C.\n\n"
        f"Question: {question_stem}\n"
        f"Choice A: {a}\n"
        f"Choice B: {b}\n"
        f"Choice C: {c}\n"
        "Answer:"
    )


def run_direct_for_datapoint(
    dp: Datapoint,
    *,
    model: str,
    base_seed: int,
    temperature: float,
    num_predict: int,
    timeout_s: int,
    retries: int,
    use_extractor: bool,
) -> Dict[str, Any]:
    prompt = build_direct_prompt(dp.question_stem, dp.choices_text)
    seed = stable_seed(base_seed, f"{dp.dataset}:{dp.id}")

    try:
        llm_output = ollama_chat(
            model,
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            num_predict=num_predict,
            seed=seed,
            timeout_s=timeout_s,
            retries=retries,
        )
        choice = extract_choice_from_text(llm_output)
        extracted = choice or ""
        if choice is None and use_extractor:
            choice, extractor_out = force_extract_choice(
                model,
                prompt,
                llm_output,
                seed=stable_seed(base_seed + 10_000, f"{dp.dataset}:{dp.id}"),
                timeout_s=min(180, timeout_s),
            )
            extracted = extractor_out.strip()

        if choice is None:
            choice = "C"
        pred_choice = choice.upper()
        pred_label = CHOICE_TO_ANSWER_LABEL.get(pred_choice, "more")
        is_correct = pred_choice == dp.answer_choice

        return {
            "id": dp.id,
            "dataset": dp.dataset,
            "question_type": str(dp.extra.get("question_type", dp.dataset)),
            "question_stem": dp.question_stem,
            "improved_question": dp.improved_question,
            "label": dp.answer_label,
            "gold_choice": dp.answer_choice,
            "is_correct": str(bool(is_correct)),
            "answer": pred_label,
            "letter_answer": pred_choice,
            "llm_output": llm_output,
            "llm_extracted_output": extracted,
            "model": model,
            "error": "",
        }
    except Exception as e:
        return {
            "id": dp.id,
            "dataset": dp.dataset,
            "question_type": str(dp.extra.get("question_type", dp.dataset)),
            "question_stem": dp.question_stem,
            "improved_question": dp.improved_question,
            "label": dp.answer_label,
            "gold_choice": dp.answer_choice,
            "is_correct": "False",
            "answer": "ERROR",
            "letter_answer": "",
            "llm_output": "",
            "llm_extracted_output": "",
            "model": model,
            "error": str(e),
        }


def compute_accuracy(path: Path) -> Tuple[int, int, float]:
    if not path.exists():
        return 0, 0, 0.0
    total = 0
    correct = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            is_correct = str(row.get("is_correct", "")).strip().lower()
            if is_correct in {"true", "1", "yes"}:
                correct += 1
    return correct, total, (correct / total if total else 0.0)


def preflight_ollama(skip: bool) -> None:
    if skip:
        return
    try:
        _ = ollama.list()
    except Exception as e:
        raise RuntimeError(
            "Ollama is not reachable. Start Ollama (open the Ollama app or run `ollama serve`), "
            "and make sure the model is pulled (e.g., `ollama pull llama3.1:8b`). "
            "If using a remote server, set OLLAMA_HOST accordingly."
        ) from e


def dataset_name_from_path(path: Path) -> str:
    return path.name.rsplit(".", 1)[0]


def run_dataset(
    datapoints: List[Datapoint],
    *,
    model: str,
    out_csv: Path,
    max_workers: int,
    base_seed: int,
    temperature: float,
    num_predict: int,
    timeout_s: int,
    retries: int,
    use_extractor: bool,
    resume: bool,
) -> None:
    fieldnames = [
        "id",
        "dataset",
        "question_type",
        "question_stem",
        "improved_question",
        "label",
        "gold_choice",
        "is_correct",
        "answer",
        "letter_answer",
        "llm_output",
        "llm_extracted_output",
        "model",
        "error",
    ]

    ensure_csv_header(out_csv, fieldnames)
    processed = read_processed_ids(out_csv) if resume else set()
    todo = [dp for dp in datapoints if dp.id not in processed]

    if not todo:
        correct, total, acc = compute_accuracy(out_csv)
        print(f"[Direct] resume: nothing to do -> {out_csv} (acc {acc*100:.2f}% {correct}/{total})")
        return

    print(f"[Direct] running {len(todo)}/{len(datapoints)} (resume: {len(processed)} done) -> {out_csv}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(
                    run_direct_for_datapoint,
                    dp,
                    model=model,
                    base_seed=base_seed,
                    temperature=temperature,
                    num_predict=num_predict,
                    timeout_s=timeout_s,
                    retries=retries,
                    use_extractor=use_extractor,
                )
                for dp in todo
            ]
            done = 0
            for fut in as_completed(futures):
                row = fut.result()
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                done += 1
                if done % 10 == 0 or done == len(futures):
                    print(f"[Direct] progress: {done}/{len(futures)}")

    correct, total, acc = compute_accuracy(out_csv)
    print(f"[Direct] accuracy: {acc*100:.2f}% ({correct}/{total}) -> {out_csv}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Direct LLM baseline (Ollama) for multiple datasets.")
    parser.add_argument("--model", type=str, default=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
    parser.add_argument("--out-dir", type=str, default=os.environ.get("DIRECT_OUT_DIR", "results"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DIRECT_SEED", "42")))
    parser.add_argument("--max-workers", type=int, default=max(1, int(os.environ.get("DIRECT_MAX_WORKERS", "4"))))
    parser.add_argument("--max-samples", type=int, default=int(os.environ.get("DIRECT_MAX_SAMPLES", "0")))
    parser.add_argument("--timeout-s", type=int, default=int(os.environ.get("DIRECT_TIMEOUT_S", "600")))
    parser.add_argument("--retries", type=int, default=int(os.environ.get("DIRECT_RETRIES", "2")))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("DIRECT_TEMPERATURE", "0.0")))
    parser.add_argument("--num-predict", type=int, default=int(os.environ.get("DIRECT_NUM_PREDICT", "32")))
    parser.add_argument(
        "--use-extractor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the model doesn't output A/B/C, make 1 extra call to extract the letter (default: True).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing CSV outputs (default: True).",
    )
    parser.add_argument(
        "--skip-preflight",
        action=argparse.BooleanOptionalAction,
        default=bool(int(os.environ.get("DIRECT_SKIP_PREFLIGHT", "0"))),
        help="Skip checking `ollama.list()` (default: False).",
    )

    parser.add_argument("--ddxplus-jsonl", type=str, default="DDXPlus_CausalQA_multistep_meta.jsonl")
    parser.add_argument("--causenet-jsonl", type=str, default="Dataset/wiqa_causenet_1hop2hop_mcqa_mix100_meta_rigorous_meta_mixed_more_less_stratified.jsonl")
    parser.add_argument("--wiqa-csv", type=str, default="other_code/CDCR-SFT/data/wiqa_test.csv")
    parser.add_argument(
        "--wiqa-use-improved-question",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For wiqa_test.csv, use the `improved_question` column as the prompt question (default: True).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    preflight_ollama(args.skip_preflight)

    model_dir = sanitize_dir_name(args.model.replace(":", "_"))
    out_base = Path(args.out_dir) / model_dir

    datasets: List[Tuple[str, Path, str]] = [
        ("ddxplus", Path(args.ddxplus_jsonl), "ddxplus-"),
        ("causenet", Path(args.causenet_jsonl), ""),
        ("wiqa_test", Path(args.wiqa_csv), "wiqa_test-"),
    ]

    for dataset_tag, path, id_prefix in datasets:
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")

        dataset_name = dataset_name_from_path(path)
        out_csv = out_base / dataset_name / "CoT_Direct.csv"

        if path.suffix.lower() == ".csv":
            datapoints = load_wiqa_csv_datapoints(
                path,
                dataset=dataset_tag,
                use_improved_question=bool(args.wiqa_use_improved_question),
                id_prefix=id_prefix,
            )
        else:
            datapoints = load_jsonl_datapoints(path, dataset=dataset_tag, id_prefix=id_prefix)

        if args.max_samples and args.max_samples > 0:
            datapoints = datapoints[: args.max_samples]

        run_dataset(
            datapoints,
            model=args.model,
            out_csv=out_csv,
            max_workers=args.max_workers,
            base_seed=args.seed,
            temperature=args.temperature,
            num_predict=args.num_predict,
            timeout_s=args.timeout_s,
            retries=args.retries,
            use_extractor=bool(args.use_extractor),
            resume=bool(args.resume),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
