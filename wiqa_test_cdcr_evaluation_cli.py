# -*- coding: utf-8 -*-
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from WIQACausalBuilder import WIQACausalBuilder


class _NullWriter:
    def write(self, s):
        return len(s)

    def flush(self):
        pass


_NULL = _NullWriter()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _resolve_under_base(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve(strict=False)


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)


def _try_load_details_df(run_dir: Path):
    details_csv = run_dir / "details.csv"
    if details_csv.exists():
        try:
            return pd.read_csv(details_csv)
        except Exception:
            return None

    details_json = run_dir / "details.json"
    if details_json.exists():
        try:
            data = json.loads(details_json.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
        except Exception:
            return None

    return None


def _summarize(details_df: pd.DataFrame, run_params: dict, config_id: str, config_index: int, elapsed_sec=None) -> dict:
    total = int(len(details_df))
    correct = int(details_df["is_correct"].sum()) if total and ("is_correct" in details_df.columns) else 0
    errors = int(details_df["error"].notna().sum()) if total and ("error" in details_df.columns) else 0
    acc = (correct / total) if total else 0.0

    acc_by_type = {}
    if total and ("question_type" in details_df.columns) and ("is_correct" in details_df.columns):
        acc_by_type = details_df.groupby("question_type")["is_correct"].mean().to_dict()

    summary = {
        "config_index": int(config_index),
        "config_id": str(config_id),
        **(run_params or {}),
        "num_samples": total,
        "num_correct": correct,
        "num_wrong": total - correct - errors,
        "num_errors": errors,
        "accuracy": float(acc),
        "elapsed_sec": float(elapsed_sec) if elapsed_sec is not None else None,
    }
    for k, v in acc_by_type.items():
        summary[f"accuracy_{k}"] = float(v)

    return summary


def _persist_grid_summary(out_dir: Path, summaries_by_id: dict) -> pd.DataFrame:
    summaries = list(summaries_by_id.values())
    _atomic_write_json(out_dir / "grid_summary.json", summaries)

    df_all = pd.DataFrame(summaries)
    if not df_all.empty and ("accuracy" in df_all.columns):
        df_sorted = df_all.sort_values(by="accuracy", ascending=False).reset_index(drop=True)
    else:
        df_sorted = df_all

    _atomic_write_csv(out_dir / "grid_summary.csv", df_sorted)
    return df_sorted


def _preflight_ollama(skip: bool) -> None:
    if skip:
        return
    try:
        import ollama

        _ = ollama.list()
    except Exception as e:
        raise RuntimeError(
            "Ollama is not reachable. Start Ollama (open the Ollama app or run `ollama serve`), "
            "and make sure the model is pulled (e.g., `ollama pull llama3.1:8b`). "
            "If using a remote server, set OLLAMA_HOST accordingly."
        ) from e


def _iter_configs(space: dict):
    keys = list(space.keys())
    for values in product(*(space[k] for k in keys)):
        yield dict(zip(keys, values))


def _config_id(cfg: dict) -> str:
    raw = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:10]


def _process_record(record, run_params, config_id: str, config_index: int, model_name: str):
    try:
        datapoint = {
            "question_stem": record["question_stem"],
            "answer_label": record["answer_label"],
            "answer_label_as_choice": record["answer_label_as_choice"],
            "choices": {"text": ["more", "less", "no_effect"], "label": ["A", "B", "C"]},
        }

        wiqa = WIQACausalBuilder(datapoint, model_name=model_name)
        is_correct = wiqa.run_wiqa_pipeline(**run_params)

        return {
            "config_index": config_index,
            "config_id": config_id,
            "csv_id": record.get("id", ""),
            "question": record.get("question_stem", ""),
            "question_type": record.get("question_type", ""),
            "improved_question": record.get("improved_question", ""),
            "gold_answer": record.get("answer_label", ""),
            "gold_choice": record.get("answer_label_as_choice", ""),
            "is_correct": bool(is_correct),
            "cause_event": getattr(wiqa, "cause_event", ""),
            "outcome_base": getattr(wiqa, "outcome_base", ""),
        }
    except Exception as e:
        return {
            "config_index": config_index,
            "config_id": config_id,
            "csv_id": record.get("id", ""),
            "question": record.get("question_stem", ""),
            "question_type": record.get("question_type", ""),
            "improved_question": record.get("improved_question", ""),
            "gold_answer": record.get("answer_label", ""),
            "gold_choice": record.get("answer_label_as_choice", ""),
            "predicted_answer": "ERROR",
            "predicted_choice": "",
            "is_correct": False,
            "cause_event": "",
            "outcome_base": "",
            "error": str(e),
        }


DEFAULT_SEARCH_SPACE = {
    "bfs_max_depth": [4],
    "bfs_max_relations_per_node": [3],
    "bfs_beam_width": [5],
    "bridge_max_bridge_nodes": [3],
    "seed_max_parents": [6],
    "chain_max_path_length": [6],
    "bfs_max_nodes": [50],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WIQA CDCR grid-search evaluation (CLI version of the notebook).")
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("WIQA_BASE_DIR", ""),
        help="Resolve relative paths under this dir (default: this script's folder).",
    )
    parser.add_argument(
        "--csv",
        default=os.environ.get("WIQA_CSV_PATH", "other_code/CDCR-SFT/data/wiqa_test.csv"),
        help="Path to wiqa_test.csv (default: other_code/CDCR-SFT/data/wiqa_test.csv).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("WIQA_OUT_DIR", "grid_search_cdcr"),
        help="Output folder (default: grid_search_cdcr).",
    )
    parser.add_argument(
        "--search-space-json",
        default=os.environ.get("WIQA_SEARCH_SPACE_JSON", ""),
        help="Optional JSON file for search space (dict of param -> list of values).",
    )
    parser.add_argument("--model-name", default=os.environ.get("WIQA_MODEL_NAME", "llama3.1:8b"))
    parser.add_argument("--random-seed", type=int, default=_int_env("WIQA_RANDOM_SEED", 42))
    parser.add_argument("--max-workers", type=int, default=max(1, _int_env("WIQA_MAX_WORKERS", 8)))
    parser.add_argument("--max-samples", type=int, default=max(0, _int_env("WIQA_MAX_SAMPLES", 0)))
    parser.add_argument("--max-configs", type=int, default=max(0, _int_env("WIQA_MAX_CONFIGS", 0)))
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=bool(_int_env("WIQA_RESUME", 1)),
        help="Skip configs with complete outputs (default: True).",
    )
    parser.add_argument(
        "--save-after-each-config",
        action=argparse.BooleanOptionalAction,
        default=bool(_int_env("WIQA_SAVE_AFTER_EACH_CONFIG", 1)),
        help="Persist grid_summary after each config (default: True).",
    )
    parser.add_argument(
        "--suppress-pipeline-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress WIQACausalBuilder prints (default: True).",
    )
    parser.add_argument(
        "--skip-ollama-preflight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not check Ollama availability before running.",
    )
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir if not args.base_dir else Path(args.base_dir).expanduser().resolve(strict=False)
    csv_path = _resolve_under_base(args.csv, base_dir)
    out_dir = _resolve_under_base(args.out_dir, base_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.search_space_json:
        space_path = _resolve_under_base(args.search_space_json, base_dir)
        search_space = json.loads(space_path.read_text(encoding="utf-8"))
        if not isinstance(search_space, dict):
            raise ValueError(f"Invalid search space JSON (expected dict): {space_path}")
    else:
        search_space = DEFAULT_SEARCH_SPACE

    df = pd.read_csv(csv_path)
    print(f"CSV: {csv_path} | rows: {len(df)}")
    if "question_type" in df.columns:
        print("\nQuestion types distribution:")
        print(df["question_type"].value_counts())

    configs = list(_iter_configs(search_space))
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    out_dir.mkdir(parents=True, exist_ok=True)

    df_eval = df
    if args.max_samples > 0:
        df_eval = df.sample(n=min(args.max_samples, len(df)), random_state=args.random_seed).reset_index(drop=True)

    records = df_eval.to_dict("records")
    expected_rows = len(records)
    print(
        f"\nGrid configs: {len(configs)} | samples: {expected_rows} | "
        f"workers: {args.max_workers} | out_dir: {out_dir}"
    )

    _preflight_ollama(skip=args.skip_ollama_preflight)

    # -----------------------------
    # Resume: load completed configs
    # -----------------------------
    summaries_by_id: dict[str, dict] = {}
    if args.resume:
        for config_index, run_params in enumerate(configs, start=1):
            config_id = _config_id(run_params)
            run_dir = out_dir / f"{config_index:03d}_{config_id}"
            details_df = _try_load_details_df(run_dir)
            if details_df is None or len(details_df) != expected_rows:
                continue

            summary_path = run_dir / "summary.json"
            summary = None
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    summary = None

            if not isinstance(summary, dict):
                summary = _summarize(details_df, run_params, config_id, config_index, elapsed_sec=None)

            summary["details_dir"] = str(run_dir)
            summaries_by_id[config_id] = summary

        if summaries_by_id:
            print(f"Resume enabled: found {len(summaries_by_id)} completed configs; will skip them.")

    # -----------------------------
    # Evaluation
    # -----------------------------
    suppress_ctx = contextlib.redirect_stdout(_NULL) if args.suppress_pipeline_output else contextlib.nullcontext()

    for config_index, run_params in enumerate(configs, start=1):
        config_id = _config_id(run_params)

        if args.resume and (config_id in summaries_by_id):
            continue

        run_dir = out_dir / f"{config_index:03d}_{config_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        _atomic_write_json(run_dir / "config.json", run_params)

        t0 = time.time()
        results_cfg = [None] * len(records)
        with suppress_ctx:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_record, r, run_params, config_id, config_index, args.model_name
                    ): i
                    for i, r in enumerate(records)
                }
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"[{config_index}/{len(configs)}] cfg={config_id}",
                    file=sys.stderr,
                ):
                    i = futures[fut]
                    results_cfg[i] = fut.result()

        results_cfg = [r for r in results_cfg if r is not None]
        elapsed = time.time() - t0

        _atomic_write_json(run_dir / "details.json", results_cfg)
        details_df = pd.DataFrame(results_cfg)
        _atomic_write_csv(run_dir / "details.csv", details_df)

        summary = _summarize(details_df, run_params, config_id, config_index, elapsed_sec=elapsed)
        summary["details_dir"] = str(run_dir)
        _atomic_write_json(run_dir / "summary.json", summary)
        try:
            (run_dir / "_DONE").write_text("ok", encoding="utf-8")
        except Exception:
            pass

        summaries_by_id[config_id] = summary

        if args.save_after_each_config:
            _ = _persist_grid_summary(out_dir, summaries_by_id)

    grid_summary_df = _persist_grid_summary(out_dir, summaries_by_id)
    print(f"\nSaved grid summary to: {out_dir / 'grid_summary.csv'}")

    best_config = grid_summary_df.iloc[0].to_dict() if not grid_summary_df.empty else None
    results = []
    if best_config and best_config.get("details_dir"):
        best_dir = Path(best_config["details_dir"])
        best_details_df = _try_load_details_df(best_dir)
        if best_details_df is not None:
            results = best_details_df.to_dict("records")

    if best_config is not None:
        _atomic_write_json(out_dir / "best_details.json", results)
        _atomic_write_csv(out_dir / "best_details.csv", pd.DataFrame(results))
        _atomic_write_json(out_dir / "best_config.json", best_config)
        print(f"Best config: {best_config.get('config_id', '')} | accuracy={best_config.get('accuracy', None)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

