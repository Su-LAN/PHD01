# -*- coding: utf-8 -*-
"""
Converted from `wiqa_test_cdcr_evaluation.ipynb`.

Runs a small grid search for WIQACausalBuilder hyperparameters on `wiqa_test.csv`,
saves per-config outputs under `grid_search_cdcr/` (or `WIQA_OUT_DIR`), then reports
overall / per-type accuracy for the best config.

Inputs / controls (env vars):
  - WIQA_CSV_PATH: path to wiqa_test.csv (default: other_code/CDCR-SFT/data/wiqa_test.csv)
  - WIQA_OUT_DIR: output folder (default: grid_search_cdcr)
  - WIQA_MAX_WORKERS: thread workers (default: 4)
  - WIQA_MAX_SAMPLES: subsample rows (0 = all)
  - WIQA_MAX_CONFIGS: limit configs (0 = all)
  - WIQA_RESUME: resume/skip completed configs (default: 1)
  - WIQA_SAVE_AFTER_EACH_CONFIG: persist grid_summary after each config (default: 1)
  - WIQA_SKIP_OLLAMA_PREFLIGHT: skip ollama connectivity check (default: 0)
  - WIQA_SHOW_PLOT: show matplotlib window (default: 0)
"""

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


def _iter_configs(space: dict):
    keys = list(space.keys())
    for values in product(*(space[k] for k in keys)):
        yield dict(zip(keys, values))


def _config_id(cfg) -> str:
    raw = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:10]


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


def _process_record(record, run_params, config_id: str, config_index: int):
    try:
        datapoint = {
            "question_stem": record["question_stem"],
            "answer_label": record["answer_label"],
            "answer_label_as_choice": record["answer_label_as_choice"],
            "choices": {"text": ["more", "less", "no_effect"], "label": ["A", "B", "C"]},
        }

        wiqa = WIQACausalBuilder(datapoint)
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


def main() -> int:
    os.environ.setdefault("WIQA_RESUME", "1")
    os.environ.setdefault("WIQA_SAVE_AFTER_EACH_CONFIG", "1")

    script_dir = Path(__file__).resolve().parent
    csv_path = _resolve_under_base(
        os.environ.get("WIQA_CSV_PATH", "other_code/CDCR-SFT/data/wiqa_test.csv"),
        script_dir,
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"CSV: {csv_path}")
    print(f"Total datapoints in CSV: {len(df)}")
    if "question_type" in df.columns:
        print("\nQuestion types distribution:")
        print(df["question_type"].value_counts())
    print("\nColumn names:")
    print(list(df.columns))
    print("\nFirst few rows:")
    try:
        print(df.head(3).to_string(index=False))
    except Exception:
        print(df.head(3))

    # -----------------------------
    # Controls
    # -----------------------------
    RANDOM_SEED = 42
    MAX_WORKERS = max(1, _int_env("WIQA_MAX_WORKERS", 4))
    MAX_SAMPLES = max(0, _int_env("WIQA_MAX_SAMPLES", 0))  # 0 = all
    MAX_CONFIGS = max(0, _int_env("WIQA_MAX_CONFIGS", 0))  # 0 = all

    RESUME = bool(_int_env("WIQA_RESUME", 1))
    SAVE_AFTER_EACH_CONFIG = bool(_int_env("WIQA_SAVE_AFTER_EACH_CONFIG", 1))
    OUT_DIR = os.environ.get("WIQA_OUT_DIR", "grid_search_cdcr")
    SUPPRESS_PIPELINE_OUTPUT = True

    SKIP_OLLAMA_PREFLIGHT = bool(_int_env("WIQA_SKIP_OLLAMA_PREFLIGHT", 0))
    SHOW_PLOT = bool(_int_env("WIQA_SHOW_PLOT", 0))

    # -----------------------------
    # Parameter grid (edit these)
    # -----------------------------
    SEARCH_SPACE = {
        "bfs_max_depth": [2, 4, 6],
        "bfs_max_relations_per_node": [3, 5],
        "bfs_beam_width": [5],
        "bridge_max_bridge_nodes": [3],
        "seed_max_parents": [6],
        "chain_max_path_length": [4, 6],
        "bfs_max_nodes": [50],
    }

    configs = list(_iter_configs(SEARCH_SPACE))
    if MAX_CONFIGS > 0:
        configs = configs[:MAX_CONFIGS]

    try:
        import WIQACausalBuilder as _wiqa_mod

        _BASE_DIR = Path(_wiqa_mod.__file__).resolve().parent
    except Exception:
        _BASE_DIR = script_dir

    out_dir = _resolve_under_base(OUT_DIR, _BASE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optionally subsample for faster experimentation (deterministic for resume)
    df_eval = df
    if MAX_SAMPLES > 0:
        df_eval = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    records = df_eval.to_dict("records")
    expected_rows = len(records)
    print(f"\nGrid search configs: {len(configs)} | samples: {expected_rows} | workers: {MAX_WORKERS} | out_dir: {out_dir}")

    _preflight_ollama(skip=SKIP_OLLAMA_PREFLIGHT)

    # -----------------------------
    # Resume: load completed configs
    # -----------------------------
    summaries_by_id = {}
    if RESUME:
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
    for config_index, run_params in enumerate(configs, start=1):
        config_id = _config_id(run_params)

        # Skip completed
        if RESUME and (config_id in summaries_by_id):
            continue

        run_dir = out_dir / f"{config_index:03d}_{config_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Persist config for reproducibility
        _atomic_write_json(run_dir / "config.json", run_params)

        t0 = time.time()
        results_cfg = [None] * len(records)
        suppress_ctx = contextlib.redirect_stdout(_NULL) if SUPPRESS_PIPELINE_OUTPUT else contextlib.nullcontext()
        with suppress_ctx:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(_process_record, r, run_params, config_id, config_index): i
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

        if SAVE_AFTER_EACH_CONFIG:
            _ = _persist_grid_summary(out_dir, summaries_by_id)

    # Final grid summary
    grid_summary_df = _persist_grid_summary(out_dir, summaries_by_id)
    print(f"\nSaved grid summary to: {out_dir / 'grid_summary.csv'}")

    # Best config + load its details
    best_config = grid_summary_df.iloc[0].to_dict() if not grid_summary_df.empty else None
    results: list[dict] = []
    if best_config and best_config.get("details_dir"):
        best_dir = Path(best_config["details_dir"])
        best_details_df = _try_load_details_df(best_dir)
        if best_details_df is not None:
            results = best_details_df.to_dict("records")

    results_df = pd.DataFrame(results)

    # Comparison table (sorted by accuracy desc)
    comparison_cols = [
        "config_index",
        "config_id",
        "accuracy",
        "accuracy_EXOGENOUS_EFFECT",
        "accuracy_INPARA_EFFECT",
        "num_samples",
        "num_errors",
        "elapsed_sec",
        "bfs_max_depth",
        "bfs_max_relations_per_node",
        "bfs_beam_width",
        "bfs_max_nodes",
        "bridge_max_bridge_nodes",
        "seed_max_parents",
        "chain_max_path_length",
    ]
    comparison_cols = [c for c in comparison_cols if (not grid_summary_df.empty) and (c in grid_summary_df.columns)]
    comparison_table = grid_summary_df[comparison_cols].reset_index(drop=True) if comparison_cols else grid_summary_df
    if not comparison_table.empty:
        print("\nTop configs (best-first):")
        try:
            print(comparison_table.head(10).to_string(index=False))
        except Exception:
            print(comparison_table.head(10))

    # -----------------------------
    # Save best config/details (notebook cell 3)
    # -----------------------------
    best_details_json = out_dir / "best_details.json"
    _atomic_write_json(best_details_json, results)

    best_details_csv = out_dir / "best_details.csv"
    pd.DataFrame(results).to_csv(best_details_csv, index=False, encoding="utf-8")

    best_config_json = out_dir / "best_config.json"
    _atomic_write_json(best_config_json, best_config)

    print(f"\nBest config saved to: {best_config_json}")
    print(f"Best details saved to: {best_details_json} / {best_details_csv}")

    # -----------------------------
    # Overall statistics
    # -----------------------------
    total_count = len(results)
    correct_count = sum(1 for r in results if r.get("is_correct"))
    error_count = sum(1 for r in results if r.get("predicted_answer") == "ERROR")
    accuracy = correct_count / total_count if total_count > 0 else 0

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total processed: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Wrong: {total_count - correct_count - error_count}")
    print(f"Errors: {error_count}")
    print(f"Accuracy: {accuracy:.2%}")

    # -----------------------------
    # Statistics by question type (EXOGENOUS vs INPARA)
    # -----------------------------
    print("\n" + "=" * 80)
    print("STATISTICS BY QUESTION TYPE")
    print("=" * 80)

    for qtype in ["EXOGENOUS_EFFECT", "INPARA_EFFECT"]:
        type_results = [r for r in results if r.get("question_type") == qtype]
        if not type_results:
            continue
        type_total = len(type_results)
        type_correct = sum(1 for r in type_results if r.get("is_correct"))
        type_errors = sum(1 for r in type_results if r.get("predicted_answer") == "ERROR")
        type_accuracy = type_correct / type_total if type_total > 0 else 0

        print(f"\n{qtype}:")
        print(f"  Total: {type_total}")
        print(f"  Correct: {type_correct}")
        print(f"  Wrong: {type_total - type_correct - type_errors}")
        print(f"  Errors: {type_errors}")
        print(f"  Accuracy: {type_accuracy:.2%}")

    # -----------------------------
    # Visualize Results
    # -----------------------------
    if results:
        try:
            import matplotlib

            # Default to a non-interactive backend to avoid Tk/Tcl dependency issues on some setups.
            if not SHOW_PLOT:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"\n[WARN] Skip plotting (matplotlib/backend error): {e}")
            plt = None

        stats_by_type = {}
        for qtype in ["EXOGENOUS_EFFECT", "INPARA_EFFECT"]:
            type_results = [r for r in results if r.get("question_type") == qtype]
            if not type_results:
                continue
            type_total = len(type_results)
            type_correct = sum(1 for r in type_results if r.get("is_correct"))
            type_accuracy = type_correct / type_total if type_total > 0 else 0
            stats_by_type[qtype] = {"total": type_total, "correct": type_correct, "accuracy": type_accuracy}

        if stats_by_type and plt is not None:
            fig = None
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                types = list(stats_by_type.keys())
                accuracies = [stats_by_type[t]["accuracy"] * 100 for t in types]
                colors = ["#FF6B6B", "#4ECDC4"]

                bars1 = ax1.bar(types, accuracies, color=colors, alpha=0.7, edgecolor="black")
                ax1.set_ylabel("Accuracy (%)", fontsize=12)
                ax1.set_title("Accuracy by Question Type", fontsize=14, fontweight="bold")
                ax1.set_ylim(0, 100)
                ax1.grid(axis="y", alpha=0.3)

                for bar in bars1:
                    height = bar.get_height()
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                totals = [stats_by_type[t]["total"] for t in types]
                corrects = [stats_by_type[t]["correct"] for t in types]
                wrongs = [totals[i] - corrects[i] for i in range(len(types))]

                x = range(len(types))
                width = 0.35

                bars2 = ax2.bar(
                    [i - width / 2 for i in x],
                    corrects,
                    width,
                    label="Correct",
                    color="#2ECC71",
                    alpha=0.7,
                    edgecolor="black",
                )
                bars3 = ax2.bar(
                    [i + width / 2 for i in x],
                    wrongs,
                    width,
                    label="Wrong",
                    color="#E74C3C",
                    alpha=0.7,
                    edgecolor="black",
                )

                ax2.set_ylabel("Count", fontsize=12)
                ax2.set_title("Correct vs Wrong by Question Type", fontsize=14, fontweight="bold")
                ax2.set_xticks(list(x))
                ax2.set_xticklabels(types)
                ax2.legend()
                ax2.grid(axis="y", alpha=0.3)

                for bar in list(bars2) + list(bars3):
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

                plt.tight_layout()
                plot_path = out_dir / "wiqa_test_accuracy_by_type.png"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                if SHOW_PLOT:
                    plt.show()
                print(f"\nVisualization saved to: {plot_path}")
            except Exception as e:
                print(f"\n[WARN] Plotting failed: {e}")
            finally:
                if fig is not None:
                    try:
                        plt.close(fig)
                    except Exception:
                        pass

    # -----------------------------
    # Detailed Results Table
    # -----------------------------
    if not results_df.empty:
        display_df = results_df[["csv_id", "question_type", "gold_answer", "is_correct"]]
        print("\nDetailed results (first 20 rows):")
        try:
            print(display_df.head(20).to_string(index=False))
        except Exception:
            print(display_df.head(20))

    # -----------------------------
    # Error Analysis
    # -----------------------------
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    for qtype in ["EXOGENOUS_EFFECT", "INPARA_EFFECT"]:
        wrong_results = [
            r
            for r in results
            if r.get("question_type") == qtype and (not r.get("is_correct")) and r.get("predicted_answer") != "ERROR"
        ]

        print(f"\n{qtype} - Wrong Predictions: {len(wrong_results)}")
        print("-" * 80)

        for r in wrong_results[:5]:
            print(f"ID {r.get('csv_id', '')}: Gold={r.get('gold_answer', '')}")
            print(f"  Question: {r.get('question', '')}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
