#!/usr/bin/env python3
"""
Build a WIQA-style CauseNet MCQA "meta" dataset with balanced more/less labels.

This script modifies the *outcome target* in each question by inserting an explicit
marker ("MORE"/"LESS") before the outcome phrase, then flips the answer label when
the target is "LESS" (matching the WIQA-style "meta" idea used in this repo).

It is designed to keep downstream compatibility:
  - preserves the existing JSONL schema (no extra fields)
  - enforces the standard 3-way choices: ["more","less","no_effect"] / ["A","B","C"]

Typical use (keep the same ID subset as an existing meta file, but rebalance 1:1):
  python scripts/build_causenet_meta_mcqa.py ^
    --in Dataset/wiqa_causenet_1hop_mcqa_no_no_effect.jsonl ^
    --ids-from Dataset/wiqa_causenet_1hop_mcqa_no_no_effect_meta.jsonl ^
    --out Dataset/wiqa_causenet_1hop_mcqa_no_no_effect_meta.jsonl ^
    --target-more-ratio 0.5
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


CHOICES_TEXT = ["more", "less", "no_effect"]
CHOICES_LABEL = ["A", "B", "C"]
ANSWER_TO_CHOICE = {"more": "A", "less": "B", "no_effect": "C"}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from e
            if not isinstance(obj, dict):
                continue
            yield obj


def _normalize_label(label: str) -> str:
    l = (label or "").strip().lower()
    l = re.sub(r"[\s\-]+", "_", l)
    if l in {"noeffect", "nochange", "no_change", "no_effect"}:
        return "no_effect"
    return l


def _invert_label(label: str) -> str:
    if label == "more":
        return "less"
    if label == "less":
        return "more"
    return label


_OUTCOME_ANCHORS: Sequence[re.Pattern[str]] = (
    # Prefer longer phrases first.
    re.compile(r"\bwhat\s+happens\s+to\s+", flags=re.IGNORECASE),
    re.compile(r"\bhappens\s+to\s+", flags=re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+the\s+effect\s+on\s+", flags=re.IGNORECASE),
    re.compile(r"\beffect\s+on\s+", flags=re.IGNORECASE),
    re.compile(r"\bhow\s+does\s+", flags=re.IGNORECASE),
)


def _set_outcome_marker(question_stem: str, marker: str) -> str:
    """
    Insert/replace an outcome marker ("MORE"/"LESS") right before the outcome phrase.

    The function looks for common CauseNet/WIQA-style anchors:
      - "... effect on <OUTCOME>?"
      - "... happens to <OUTCOME>?"
      - "How does <OUTCOME> change?"
    """
    q = (question_stem or "").strip()
    if not q:
        return q

    marker = marker.strip().upper()
    if marker not in {"MORE", "LESS"}:
        raise ValueError(f"Invalid marker: {marker!r} (expected MORE/LESS)")

    insert_pos: Optional[int] = None
    for pat in _OUTCOME_ANCHORS:
        for m in pat.finditer(q):
            insert_pos = m.end()

    if insert_pos is None:
        return q

    after = q[insert_pos:]
    m2 = re.match(r"(?i)^(MORE|LESS)\b\s*", after)
    if m2:
        rest = after[m2.end() :].lstrip()
        return q[:insert_pos] + f"{marker} " + rest

    return q[:insert_pos] + f"{marker} " + after


def _load_id_set(path: Path) -> Set[str]:
    ids: Set[str] = set()
    for obj in _iter_jsonl(path):
        row_id = str(obj.get("id", "")).strip()
        if row_id:
            ids.add(row_id)
    return ids


def _choose_desired_labels(
    n: int,
    *,
    target_more_ratio: float,
    seed: int,
) -> List[str]:
    if n < 0:
        raise ValueError("n must be >= 0")
    if not (0.0 <= target_more_ratio <= 1.0):
        raise ValueError("--target-more-ratio must be in [0, 1]")
    target_more = int(round(n * float(target_more_ratio)))
    target_more = max(0, min(n, target_more))
    target_less = n - target_more

    desired = (["more"] * target_more) + (["less"] * target_less)
    rng = random.Random(int(seed))
    rng.shuffle(desired)
    return desired


def build_meta_dataset(
    items: Sequence[Dict[str, Any]],
    *,
    target_more_ratio: float,
    seed: int,
    drop_no_effect: bool,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    base_labels: List[str] = []

    for obj in items:
        q = str(obj.get("question_stem", "")).strip()
        base = _normalize_label(str(obj.get("answer_label", "")))
        if not q:
            continue
        if base not in {"more", "less", "no_effect"}:
            continue
        if drop_no_effect and base == "no_effect":
            continue
        filtered.append(obj)
        base_labels.append(base)

    desired_final = _choose_desired_labels(
        len(filtered),
        target_more_ratio=target_more_ratio,
        seed=seed,
    )

    out: List[Dict[str, Any]] = []
    for obj, base, desired in zip(filtered, base_labels, desired_final):
        if desired not in {"more", "less"}:
            raise ValueError(f"Internal error: invalid desired label: {desired}")

        # Choose marker to get the desired final label.
        marker = "MORE" if desired == base else "LESS"

        q = _set_outcome_marker(str(obj.get("question_stem", "")).strip(), marker=marker)

        out_obj = dict(obj)
        out_obj["question_stem"] = q
        out_obj["answer_label"] = desired
        out_obj["answer_label_as_choice"] = ANSWER_TO_CHOICE[desired]
        out_obj["choices"] = {"text": CHOICES_TEXT, "label": CHOICES_LABEL}
        out.append(out_obj)

    return out


def _write_jsonl(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _summarize(items: Sequence[Dict[str, Any]]) -> None:
    from collections import Counter

    labels = [_normalize_label(str(it.get("answer_label", ""))) for it in items]
    c = Counter(labels)
    total = sum(c.values())
    print(f"Total: {total}")
    for k in ["more", "less", "no_effect"]:
        if k in c:
            print(f"  {k}: {c[k]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a balanced MORE/LESS outcome-meta MCQA dataset for CauseNet.")
    parser.add_argument(
        "--in",
        dest="in_path",
        type=str,
        default="Dataset/wiqa_causenet_1hop_mcqa_no_no_effect.jsonl",
        help="Input MCQA jsonl (base, non-meta).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=str,
        default="Dataset/wiqa_causenet_1hop_mcqa_no_no_effect_meta.jsonl",
        help="Output meta jsonl.",
    )
    parser.add_argument(
        "--ids-from",
        dest="ids_from",
        type=str,
        default="",
        help="Optional jsonl whose `id`s define a subset to keep (preserves an existing split).",
    )
    parser.add_argument("--target-more-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap after filtering (0 = no limit).")
    parser.add_argument(
        "--keep-no-effect",
        action="store_true",
        help="Keep no_effect rows (they are not included in the more/less balancing target).",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    keep_ids: Set[str] = set()
    if str(args.ids_from).strip():
        keep_ids = _load_id_set(Path(args.ids_from))

    items: List[Dict[str, Any]] = []
    for obj in _iter_jsonl(in_path):
        if keep_ids:
            row_id = str(obj.get("id", "")).strip()
            if row_id not in keep_ids:
                continue
        items.append(obj)
        if int(args.limit) and len(items) >= int(args.limit):
            break

    meta = build_meta_dataset(
        items,
        target_more_ratio=float(args.target_more_ratio),
        seed=int(args.seed),
        drop_no_effect=not bool(args.keep_no_effect),
    )
    _write_jsonl(out_path, meta)

    print(f"Wrote: {out_path}")
    _summarize(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

