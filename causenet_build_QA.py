import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ollama


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _clean_json_text(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```json"):
        t = t[7:].strip()
    elif t.startswith("```"):
        t = t[3:].strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def _extract_json_object(text: str) -> Optional[str]:
    t = _clean_json_text(text)
    if not t:
        return None
    if t.startswith("{") and t.endswith("}"):
        return t
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    return m.group(0).strip() if m else None


def _label(concept: str) -> str:
    return (concept or "").replace("_", " ").strip()


def _build_question(head: str, tail: str) -> str:
    return f'Does "{_label(head)}" causally affect "{_label(tail)}"?'


def _extract_evidence_sentences(obj: Dict[str, Any], k: int) -> List[str]:
    if k <= 0:
        return []
    ev = obj.get("evidence") or []
    out: List[str] = []
    for e in ev:
        if not isinstance(e, dict):
            continue
        s = str(e.get("sentence", "")).strip()
        if not s:
            continue
        if len(s) > 280:
            s = s[:277].rstrip() + "..."
        out.append(s)
        if len(out) >= k:
            break
    return out


def _ollama_qa_one(
    *,
    head: str,
    tail: str,
    evidence_sentences: List[str],
    model: str,
    seed: int,
    num_predict: int,
    temperature: float,
    retries: int,
) -> Dict[str, str]:
    head_label = _label(head)
    tail_label = _label(tail)
    question_fallback = _build_question(head, tail)
    evidence_block = ""
    if evidence_sentences:
        evidence_block = "Evidence sentences:\n" + "\n".join(f"- {s}" for s in evidence_sentences) + "\n\n"

    system = "Output English only. Return only the JSON object."
    prompt = (
        "You are a strict causal relation validator.\n"
        "Task:\n"
        "1) Write ONE short WIQA-style yes/no question in English about whether the cause concept causally affects the effect concept.\n"
        "2) Then answer it with 'yes' or 'no'.\n\n"
        "Rules for the question:\n"
        '- Must include both concepts verbatim in double quotes, e.g. "mind" and "health".\n'
        "- Must ask about causality (not correlation/association).\n"
        "- Avoid diagnosis/screening/detection framing (e.g., tests used to detect a disease).\n\n"
        "Rules for the answer:\n"
        "- Answer 'yes' only for genuine real-world causality (either increasing or decreasing effects).\n"
        "- If unsure, answer 'no'.\n\n"
        f'Cause concept: "{head_label}"\n'
        f'Effect concept: "{tail_label}"\n\n'
        f"{evidence_block}"
        'Return ONLY a JSON object like: {"question": "...", "answer": "yes"}.\n'
    )

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = ollama.generate(
                model=model,
                system=system,
                prompt=prompt,
                think=False,
                format={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answer": {"type": "string", "enum": ["yes", "no"]},
                    },
                    "required": ["question", "answer"],
                },
                options={
                    "temperature": float(temperature),
                    "seed": int(seed),
                    "num_predict": int(num_predict),
                },
            )
            raw = (resp.get("response") or "").strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            json_text = _extract_json_object(raw)
            if not json_text:
                raise ValueError(f"Could not extract JSON from response: {raw[:200]}")
            obj = json.loads(json_text)
            q = str(obj.get("question", "")).strip()
            a = str(obj.get("answer", "")).strip().lower()
            if a not in {"yes", "no"}:
                raise ValueError(f"Invalid answer (expected yes/no): {json_text}")
            if not q:
                q = question_fallback
            q = q.replace(f"'{head_label}'", f"\"{head_label}\"")
            q = q.replace(f"'{tail_label}'", f"\"{tail_label}\"")
            if head_label not in q or tail_label not in q or re.search(r"[\u4e00-\u9fff]", q):
                q = question_fallback
            return {"question": q, "answer": a}
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))

    raise RuntimeError(f"Ollama QA generation failed: {last_err}") from last_err


def _ollama_build_yes_question(
    *,
    head: str,
    tail: str,
    evidence_sentences: List[str],
    model: str,
    seed: int,
    num_predict: int,
    temperature: float,
    retries: int,
    question_lang: str,
) -> str:
    head_label = _label(head)
    tail_label = _label(tail)
    evidence_block = ""
    if evidence_sentences:
        evidence_block = "Evidence sentences:\n" + "\n".join(f"- {s}" for s in evidence_sentences) + "\n\n"

    if question_lang == "zh":
        system = "你只输出中文。不要输出答案或选项。"
        prompt = (
            "你是一个因果推理数据集（类似 WIQA）的题目编写助手。\n"
            "请基于给定概念与证据，写出一个自然、简短的中文问题，询问因果关系是否成立。\n"
            "该问题的正确答案应当是“yes”。\n"
            "要求：\n"
            "- 只输出一个问题句子，不要输出答案或选项。\n"
            '- 在问题中用英文双引号原样包含这两个概念（例如 "mind"、"health"），不要用单引号。\n'
            "- 不要使用 Markdown。\n\n"
            f'Cause concept: "{head_label}"\n'
            f'Effect concept: "{tail_label}"\n\n'
            "你可以优先使用下面这种句式风格（任选其一，保持自然）：\n"
            f'1) 假设发生了 "{head_label}"，这是否会导致 "{tail_label}"？\n'
            f'2) 如果 "{head_label}" 增加或出现，"{tail_label}" 是否会发生或加剧？\n'
            f'3) 假设 "{head_label}" 出现问题，"{tail_label}" 是否会受到影响？\n\n'
            f"{evidence_block}"
            'Return ONLY a JSON object like: {"question": "..."}.\n'
        )
    else:
        system = "Output English only. Do not output the answer or options."
        prompt = (
            "You write causal yes/no questions (WIQA-style).\n"
            "Using the given concepts and evidence, write ONE short, natural question.\n"
            "The correct answer to the question should be 'yes'.\n"
            "Requirements:\n"
            "- Output only one question sentence; do NOT output the answer or options.\n"
            '- Include both concepts verbatim in double quotes (e.g., "mind", "health").\n'
            "- No Markdown.\n\n"
            f'Cause concept: "{head_label}"\n'
            f'Effect concept: "{tail_label}"\n\n'
            f"{evidence_block}"
            'Return ONLY a JSON object like: {"question": "..."}.\n'
        )

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = ollama.generate(
                model=model,
                system=system,
                prompt=prompt,
                think=False,
                format={
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                },
                options={
                    "temperature": float(temperature),
                    "seed": int(seed),
                    "num_predict": int(num_predict),
                },
            )
            raw = (resp.get("response") or "").strip()
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            json_text = _extract_json_object(raw)
            if not json_text:
                raise ValueError(f"Could not extract JSON from response: {raw[:200]}")
            obj = json.loads(json_text)
            q = str(obj.get("question", "")).strip()
            if not q:
                raise ValueError(f"Invalid question JSON: {json_text}")
            q = q.replace(f"'{head_label}'", f"\"{head_label}\"")
            q = q.replace(f"'{tail_label}'", f"\"{tail_label}\"")
            if head_label not in q or tail_label not in q:
                if question_lang == "zh":
                    q = f'假设发生了 "{head_label}"，这是否会导致 "{tail_label}"？'
                else:
                    q = f'Does "{head_label}" causally affect "{tail_label}"?'
            if question_lang == "zh" and not re.search(r"[\u4e00-\u9fff]", q):
                q = f'假设发生了 "{head_label}"，这是否会导致 "{tail_label}"？'
            return q
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))

    raise RuntimeError(f"Ollama question generation failed: {last_err}") from last_err


def _default_out_path(in_path: str) -> str:
    base = os.path.basename(in_path)
    root, ext = os.path.splitext(base)
    if not ext:
        ext = ".jsonl"
    return os.path.join(os.path.dirname(in_path), f"{root}_QA{ext}")


def _load_done_pairs(out_path: str) -> set[Tuple[str, str]]:
    done: set[Tuple[str, str]] = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                head = str(obj.get("head", "")).strip()
                tail = str(obj.get("tail", "")).strip()
                if head and tail:
                    done.add((head, tail))
            except Exception:
                continue
    return done


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use Ollama to (1) judge causal pairs (yes/no) or (2) build WIQA-style yes/no questions."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        default="causenet_head_tail_pairs_min1_max1_120.jsonl",
        help="Input jsonl containing 'head' and 'tail' fields.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Output jsonl path (default: <input>_QA.jsonl).",
    )
    parser.add_argument(
        "--mode",
        choices=["judge", "build"],
        default="judge",
        help="judge: label pairs yes/no; build: for rows with answer==yes, generate a WIQA-style question (answer fixed to yes).",
    )
    parser.add_argument("--model", type=str, default="deepseek-r1:32b", help="Ollama model name.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel requests (usually 1 is fastest).")
    parser.add_argument("--limit", type=int, default=0, help="Max examples to process (0 = no limit).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-predict", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--evidence-k",
        type=int,
        default=2,
        help="Include up to K evidence sentences in the LLM prompt (0 disables).",
    )
    parser.add_argument(
        "--question-lang",
        choices=["en", "zh"],
        default="zh",
        help="Language for generated questions in --mode build.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip pairs already present in --out.")
    args = parser.parse_args()

    out_path = args.out_path or _default_out_path(args.in_path)

    done_pairs: set[Tuple[str, str]] = set()
    if args.resume:
        done_pairs = _load_done_pairs(out_path)

    rows: List[Dict[str, Any]] = []
    for obj in _iter_jsonl(args.in_path):
        head = str(obj.get("head", "")).strip()
        tail = str(obj.get("tail", "")).strip()
        if not head or not tail:
            continue
        if args.mode == "build":
            ans = str(obj.get("answer", "")).strip().lower()
            if ans != "yes":
                continue
        if done_pairs and (head, tail) in done_pairs:
            continue
        rows.append(obj)
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        return 0

    with open(out_path, "a", encoding="utf-8") as out_f:
        if args.workers <= 1:
            for obj in rows:
                evidence_sentences = _extract_evidence_sentences(obj, int(args.evidence_k))
                out_obj = dict(obj)
                if args.mode == "judge":
                    qa = _ollama_qa_one(
                        head=str(obj["head"]),
                        tail=str(obj["tail"]),
                        evidence_sentences=evidence_sentences,
                        model=args.model,
                        seed=args.seed,
                        num_predict=args.num_predict,
                        temperature=args.temperature,
                        retries=args.retries,
                    )
                    out_obj.update(qa)
                else:
                    judge_question = str(out_obj.get("question", "")).strip()
                    judge_answer = str(out_obj.get("answer", "")).strip().lower()
                    q = _ollama_build_yes_question(
                        head=str(obj["head"]),
                        tail=str(obj["tail"]),
                        evidence_sentences=evidence_sentences,
                        model=args.model,
                        seed=args.seed,
                        num_predict=args.num_predict,
                        temperature=args.temperature,
                        retries=args.retries,
                        question_lang=str(args.question_lang),
                    )
                    out_obj["judge_question"] = judge_question
                    out_obj["judge_answer"] = judge_answer
                    out_obj["question"] = q
                    out_obj["choices"] = ["yes", "no"]
                    out_obj["answer"] = "yes"
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_f.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                if args.mode == "judge":
                    futures = [
                        (
                            obj,
                            ex.submit(
                                _ollama_qa_one,
                                head=str(obj["head"]),
                                tail=str(obj["tail"]),
                                evidence_sentences=_extract_evidence_sentences(obj, int(args.evidence_k)),
                                model=args.model,
                                seed=args.seed,
                                num_predict=args.num_predict,
                                temperature=args.temperature,
                                retries=args.retries,
                            ),
                        )
                        for obj in rows
                    ]
                    for obj, fut in futures:
                        qa = fut.result()
                        out_obj = dict(obj)
                        out_obj.update(qa)
                        out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                else:
                    futures = [
                        (
                            obj,
                            ex.submit(
                                _ollama_build_yes_question,
                                head=str(obj["head"]),
                                tail=str(obj["tail"]),
                                evidence_sentences=_extract_evidence_sentences(obj, int(args.evidence_k)),
                                model=args.model,
                                seed=args.seed,
                                num_predict=args.num_predict,
                                temperature=args.temperature,
                                retries=args.retries,
                                question_lang=str(args.question_lang),
                            ),
                        )
                        for obj in rows
                    ]
                    for obj, fut in futures:
                        q = fut.result()
                        out_obj = dict(obj)
                        out_obj["judge_question"] = str(out_obj.get("question", "")).strip()
                        out_obj["judge_answer"] = str(out_obj.get("answer", "")).strip().lower()
                        out_obj["question"] = q
                        out_obj["choices"] = ["yes", "no"]
                        out_obj["answer"] = "yes"
                        out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                out_f.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
