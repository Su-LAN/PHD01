# Auto-exported from baseline_e-care.ipynb

# %% [cell 2]
import os
import csv
import json
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
try:
    import ollama  # optional (fallback uses Ollama OpenAI endpoint)
except Exception:
    ollama = None

random.seed(42)

# %% [cell 4]
import os
INPUT_JSONL = os.path.join('result', 'e-care-more.jsonl')
OLLAMA_MODEL = 'llama3.1:8b'  # change if needed
OUT_DIR = 'results'
DATASET_NAME = os.path.splitext(os.path.basename(INPUT_JSONL))[0]  # avoid mixing different datasets

# Run mode
RUN_MODE = 'batch'  # 'single' or 'batch'
SINGLE_ROW_INDEX = 0  # 0-based index into `rows`
SINGLE_ROW_LINE_NO = 0  # set to an int to select by JSONL line number
SINGLE_ROW_QUERY = ''  # substring match in question_stem (first hit)

# Rerun policy
RERUN_ERRORS = True  # rerun if an existing CSV row has llm_output starting with 'ERROR'
FORCE_RERUN_SINGLE = False  # in single mode, rerun even if already processed

CDCR_SFT_CODE_DIR = r'e:/PHD/01/other_code/CDCR-SFT/code'
OLLAMA_OPENAI_BASE_URL = 'http://localhost:11434/v1'
CDCR_DATASET_TYPE = 'wiqa'  # closest CDCR-SFT wrapper for more/less/no effect

METHODS = ['CoT', 'GoT', 'ToT', 'CausalCoT']

SEED = 42
MAX_SAMPLES = 0  # 0 = all rows

# This dataset has many duplicated questions; caching speeds up runs.
# If you want each row to be an independent trial, set to False.
CACHE_BY_QUESTION = True

# %% [cell 6]
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj['_line_no'] = line_no
            rows.append(obj)
    return rows


def normalize_moreless_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common schemas into {question_stem, answer_label} expected by this notebook."""

    # question
    q = row.get('question_stem')
    if not (isinstance(q, str) and q.strip()):
        q = row.get('question')
    row['question_stem'] = str(q or '').strip()

    # label
    label = row.get('answer_label')
    if not (isinstance(label, str) and label.strip()):
        label = row.get('label_text')

    if label is None:
        label_id = row.get('label_id')
        choices = row.get('choices')
        try:
            if label_id is not None and isinstance(choices, list):
                label = choices[int(label_id)]
        except Exception:
            label = None

    lab = str(label or '').strip().lower()
    if lab in {'noeffect', 'no-effect', 'no change', 'nochange', 'no_effect'}:
        lab = 'no effect'
    elif lab.startswith('more'):
        lab = 'more'
    elif lab.startswith('less'):
        lab = 'less'
    row['answer_label'] = lab

    if 'answer_label_as_choice' not in row:
        row['answer_label_as_choice'] = {'more': 'a', 'less': 'b', 'no effect': 'c'}.get(lab, '')

    return row


rows = [normalize_moreless_row(r) for r in load_jsonl(INPUT_JSONL)]
if MAX_SAMPLES and MAX_SAMPLES > 0:
    rows = rows[:MAX_SAMPLES]

print('rows:', len(rows))
print('keys:', sorted(set().union(*[r.keys() for r in rows[:50]])))

labels = [r.get('answer_label', '') for r in rows]
print(pd.Series(labels).value_counts())

questions = [r.get('question_stem', '') for r in rows]
unique_questions = len(set(questions))
print('unique_questions:', unique_questions)
print('duplicates:', len(rows) - unique_questions)

label_counts = pd.Series(labels).value_counts()
majority_label = label_counts.index[0]
majority_acc = float(label_counts.iloc[0]) / len(rows)
print('majority label:', majority_label)
print('majority baseline acc:', majority_acc)

q2label: Dict[str, str] = {}
conflicts: List[Tuple[str, str, str]] = []
for r in rows:
    q = str(r.get('question_stem', '') or '')
    lab = str(r.get('answer_label', '') or '')
    if q in q2label and q2label[q] != lab:
        conflicts.append((q, q2label[q], lab))
    q2label[q] = lab
print('label conflicts:', len(conflicts))

# %% [cell 8]
ANSWER_LABEL_TO_CHOICE = {'more': 'a', 'less': 'b', 'no effect': 'c'}
CHOICE_TO_ANSWER_LABEL = {'a': 'more', 'b': 'less', 'c': 'no effect'}


def sanitize_dir_name(name: str) -> str:
    return ''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in name)


def build_moreless_prompt(question_stem: str) -> str:
    return f'''answer the Question: {question_stem}
Choice A: more
Choice B: less
Choice C: no effect'''


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
            if ollama is None:
                cdcr = ensure_cdcr_sft_ready()
                resp = cdcr['client'].chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=num_predict,
                    timeout=timeout_s,
                )
                try:
                    content = resp.choices[0].message.content
                except Exception:
                    content = resp['choices'][0]['message']['content']
            else:
                resp = ollama.chat(
                    model=model,
                    messages=messages,
                    options={
                        'temperature': temperature,
                        'num_predict': num_predict,
                        'seed': seed,
                    },
                )
                content = (resp.get('message') or {}).get('content', '')
            if not isinstance(content, str):
                content = str(content)
            elapsed = time.time() - start
            if elapsed > timeout_s:
                raise TimeoutError(f'ollama.chat exceeded timeout: {elapsed:.1f}s > {timeout_s}s')
            return content
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f'Ollama call failed after retries: {last_err}') from last_err


def extract_choice_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last = lines[-1].rstrip('.')
        if len(last) == 1 and last.upper() in {'A', 'B', 'C'}:
            return last.lower()

    lower = text.lower()
    for key in ['final answer', 'answer']:
        idx = lower.rfind(key)
        if idx == -1:
            continue
        frag = text[idx:]
        frag = frag.split(':', 1)[1].strip() if ':' in frag else frag[len(key):].strip()
        token = frag.split()[0].strip().rstrip('.') if frag else ''
        if len(token) == 1 and token.upper() in {'A', 'B', 'C'}:
            return token.lower()
        frag_lower = frag.lower()
        if frag_lower.startswith('no effect') or frag_lower.startswith('no change'):
            return 'c'
        if frag_lower.startswith('more'):
            return 'a'
        if frag_lower.startswith('less'):
            return 'b'

    return None


def force_extract_choice(
    model: str,
    question_prompt: str,
    reasoning_text: str,
    *,
    seed: int,
) -> Tuple[Optional[str], str]:
    prompt = f'''You are an answer extractor.
Given the question and a model's reasoning, output ONLY one letter: A, B, or C.

{question_prompt}

Reasoning:
{reasoning_text}

Output:''' 
    out = ollama_chat(
        model,
        [{'role': 'user', 'content': prompt}],
        temperature=0.0,
        num_predict=8,
        seed=seed,
        timeout_s=180,
        retries=1,
    )
    return extract_choice_from_text(out), out


def parse_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# --- CDCR-SFT integration (GoT/ToT) ---
_CDCR_SFT_STATE: Dict[str, Any] = {}


def ensure_cdcr_sft_ready() -> Dict[str, Any]:
    global _CDCR_SFT_STATE
    if _CDCR_SFT_STATE.get('ready'):
        return _CDCR_SFT_STATE

    if not os.path.isdir(CDCR_SFT_CODE_DIR):
        raise FileNotFoundError(f'CDCR_SFT_CODE_DIR not found: {CDCR_SFT_CODE_DIR}')

    import sys
    if CDCR_SFT_CODE_DIR not in sys.path:
        sys.path.insert(0, CDCR_SFT_CODE_DIR)

    try:
        import openai  # CDCR-SFT requirement: openai==0.27.7
    except Exception as e:
        raise RuntimeError(
            'Missing openai. Install: pip install openai==0.27.7 tree-of-thoughts-llm==0.1.0 graph_of_thoughts==0.0.2 backoff'
        ) from e

    # Shim: make openai==0.27.7 look like OpenAI() client (v1 style)
    if not hasattr(openai, 'OpenAI'):
        class _ChatCompletions:
            @staticmethod
            def create(*, timeout: Optional[int] = None, **kwargs):
                if timeout is not None and 'request_timeout' not in kwargs:
                    kwargs['request_timeout'] = timeout
                return openai.ChatCompletion.create(**kwargs)

        class _Chat:
            def __init__(self):
                self.completions = _ChatCompletions()

        class OpenAIShim:
            def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **_):
                if api_key is not None:
                    openai.api_key = api_key
                if base_url is not None:
                    openai.api_base = base_url
                self.api_key = api_key
                self.base_url = base_url
                self.chat = _Chat()

        openai.OpenAI = OpenAIShim

    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY', 'ollama'), base_url=OLLAMA_OPENAI_BASE_URL)

    # Patch graph_of_thoughts==0.0.2 for CDCR-SFT GoT wrappers.
    try:
        import importlib
        import importlib.util
        spec = importlib.util.find_spec('graph_of_thoughts.controller')
        if spec and spec.origin:
            init_path = spec.origin
            txt = open(init_path, 'r', encoding='utf-8', errors='ignore').read()
            if 'from .controller import Controller' in txt and '__getattr__' not in txt:
                patched_lines = [
                    'from .abstract_language_model import AbstractLanguageModel',
                    '',
                    '',
                    'def __getattr__(name: str):',
                    "    if name == 'Controller':",
                    '        from .controller import Controller',
                    '        return Controller',
                    "    if name == 'ChatGPT':",
                    '        from .chatgpt import ChatGPT',
                    '        return ChatGPT',
                    "    if name == 'Llama2HF':",
                    '        from .llamachat_hf import Llama2HF',
                    '        return Llama2HF',
                    '    raise AttributeError(name)',
                    '',
                    '',
                    "__all__ = ['AbstractLanguageModel', 'Controller', 'ChatGPT', 'Llama2HF']",
                    '',
                ]
                open(init_path, 'w', encoding='utf-8').write('\n'.join(patched_lines))
                importlib.invalidate_caches()
                sys.modules.pop('graph_of_thoughts.controller', None)

        try:
            import graph_of_thoughts.language_models  # type: ignore
        except Exception:
            from types import ModuleType
            from graph_of_thoughts.controller.abstract_language_model import AbstractLanguageModel
            m = ModuleType('graph_of_thoughts.language_models')
            m.AbstractLanguageModel = AbstractLanguageModel
            sys.modules['graph_of_thoughts.language_models'] = m
    except Exception as e:
        raise RuntimeError('graph_of_thoughts compat failed. Install: pip install graph_of_thoughts==0.0.2') from e

    got_wrapper = importlib.import_module('got_wrapper')
    tot_wrapper = importlib.import_module('tot_wrapper')
    cdcr_utils = importlib.import_module('utils')

    _CDCR_SFT_STATE = {
        'ready': True,
        'client': client,
        'got_wrapper': got_wrapper,
        'tot_wrapper': tot_wrapper,
        'utils': cdcr_utils,
    }
    return _CDCR_SFT_STATE

# %% [cell 10]
def run_cot(
    model: str,
    question_prompt: str,
    *,
    seed: int,
    causal_variant: bool,
) -> str:
    if causal_variant:
        method_header = 'CausalCoT'
        guidance = '''Guidance: Solve the causal effect direction question with explicit causal structure.
Step 1) Identify the intervention/cause variable and the outcome variable.
Step 2) Construct a minimal causal graph (edge list like X -> M, M -> Y).
Step 3) Briefly explain the mechanism/direction from cause to outcome.
Step 4) Choose the best option.'''
    else:
        method_header = 'CoT'
        guidance = '''Guidance: Use chain-of-thought.
1) Choose the best option.'''

    prompt = f'''[{method_header}]
{guidance}

{question_prompt}

Output format:
Final answer: <A|B|C>
'''
    return ollama_chat(
        model,
        [{'role': 'user', 'content': prompt}],
        temperature=0.0,
        num_predict=512,
        seed=seed,
        timeout_s=600,
        retries=2,
    )


def heuristic_score_component(text: str, *, component_idx: int) -> float:
    if not text:
        return 0.0
    t = text.lower()
    score = 0.0
    if len(text) > 80:
        score += 0.2
    if '->' in text or 'caus' in t or 'effect' in t:
        score += 0.2
    if component_idx == 1 and '->' in text:
        score += 0.3
    if component_idx == 3 and extract_choice_from_text(text) is not None:
        score += 0.5
    return min(score, 1.0)


def run_got(model: str, question_prompt: str, *, seed: int) -> str:
    """CDCR-SFT GoT wrapper (graph_of_thoughts)."""
    _ = seed  # CDCR-SFT wrapper does not expose seeding
    cdcr = ensure_cdcr_sft_ready()
    res = cdcr['got_wrapper'].run_got_reasoning(
        cdcr['client'],
        model,
        question_prompt,
        dataset_type=CDCR_DATASET_TYPE,
    )
    if isinstance(res, dict):
        return str(res.get('reasoning', '') or res.get('answer', '') or res)
    return str(res)


def run_tot(model: str, question_prompt: str, *, seed: int) -> str:
    """CDCR-SFT ToT wrapper (tree-of-thoughts-llm)."""
    _ = seed  # CDCR-SFT wrapper does not expose seeding
    cdcr = ensure_cdcr_sft_ready()
    tot_prompt = f'Question: \n{question_prompt}\n\n'
    answer, metadata = cdcr['tot_wrapper'].run_tree_of_thoughts(
        tot_prompt,
        dataset_type=CDCR_DATASET_TYPE,
        model_name=model,
        temperature=0.0,
        client=cdcr['client'],
        model_endpoint=getattr(cdcr['client'], 'base_url', OLLAMA_OPENAI_BASE_URL),
    )
    if isinstance(metadata, dict) and metadata.get('full_output'):
        return str(metadata['full_output'])
    return str(answer)

# %% [cell 12]
CSV_FIELDS = [
    'id',
    'question_type',
    'label',
    'is_correct',
    'answer',
    'letter_answer',
    'llm_output',
    'llm_extracted_output',
    'model',
]


def ensure_csv_header(path: str, fieldnames: List[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def read_processed_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    processed = set()
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'id' in row and row['id'] != '':
                processed.add(row['id'])
    return processed


def read_csv_row_by_id(path: str, row_id: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get('id', '')) == str(row_id):
                return row
    return None


def upsert_row_by_id(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    row_id = str(row.get('id', ''))
    existing: List[Dict[str, Any]] = []

    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing.append(r)

    replaced = False
    out_rows: List[Dict[str, Any]] = []
    for r in existing:
        if str(r.get('id', '')) == row_id and row_id != '':
            out_rows.append({k: row.get(k, '') for k in fieldnames})
            replaced = True
        else:
            out_rows.append({k: r.get(k, '') for k in fieldnames})

    if not replaced:
        out_rows.append({k: row.get(k, '') for k in fieldnames})

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)


def append_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({k: row.get(k, '') for k in fieldnames})


def compute_accuracy(path: str) -> Tuple[int, int, float]:
    if not os.path.exists(path):
        return 0, 0, 0.0
    total = 0
    correct = 0
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            is_correct = str(row.get('is_correct', '')).strip().lower()
            if is_correct in {'true', '1', 'yes'}:
                correct += 1
    return correct, total, (correct / total if total else 0.0)


def get_output_csv_path(method: str) -> str:
    model_dir = sanitize_dir_name(OLLAMA_MODEL.replace(':', '_'))
    dataset_dir = os.path.join(OUT_DIR, model_dir, DATASET_NAME)
    os.makedirs(dataset_dir, exist_ok=True)
    return os.path.join(dataset_dir, f'CoT_{method}.csv')


def run_method(method: str, rows: List[Dict[str, Any]], *, row_indices: Optional[List[int]] = None) -> str:
    out_csv = get_output_csv_path(method)
    ensure_csv_header(out_csv, CSV_FIELDS)
    processed_ids = read_processed_ids(out_csv)

    cache: Dict[str, Tuple[str, str, str]] = {}
    if row_indices is None:
        indices = list(range(len(rows)))
    else:
        indices = list(row_indices)
        for i in indices:
            if i < 0 or i >= len(rows):
                raise IndexError(f'row index out of range: {i} (rows={len(rows)})')

    print(f'\n=== Running {method} on {len(indices)} rows (resume: {len(processed_ids)} done) ===')

    for local_i, idx in enumerate(indices):
        row = rows[idx]
        row_id = str(row.get('idx') or row.get('_line_no', idx))
        if row_id in processed_ids:
            prev = read_csv_row_by_id(out_csv, row_id)
            prev_out = str((prev or {}).get('llm_output', '') or '').strip()
            should_rerun = False
            if FORCE_RERUN_SINGLE and len(indices) <= 3:
                should_rerun = True
            elif RERUN_ERRORS and (prev is None or prev_out.startswith('ERROR')):
                should_rerun = True

            if not should_rerun:
                if len(indices) <= 3 and method in {'GoT', 'ToT'} and prev:
                    meta = ''
                    if row.get('outcome_polarity') is not None:
                        meta = f" outcome_polarity={row.get('outcome_polarity')} base={row.get('answer_label_base')}"
                    print(
                        f"[{method}] row_id={row_id}{meta} (from CSV) gold={prev.get('label')} "
                        f"pred={prev.get('answer')} choice={prev.get('letter_answer')}"
                    )
                    print(f"[{method}] LLM response:\n{prev.get('llm_output', '')}")
                    ex = str(prev.get('llm_extracted_output', '') or '').strip()
                    if ex:
                        print(f'[{method}] extracted: {ex}')
                    print('---')
                continue

        question = str(row.get('question_stem', '')).strip()
        gold_label = str(row.get('answer_label', '')).strip().lower()
        question_prompt = build_moreless_prompt(question)

        if CACHE_BY_QUESTION and question in cache:
            llm_output, choice, extracted = cache[question]
        else:
            try:
                if method == 'CoT':
                    llm_output = run_cot(OLLAMA_MODEL, question_prompt, seed=SEED + idx, causal_variant=False)
                elif method == 'CausalCoT':
                    llm_output = run_cot(OLLAMA_MODEL, question_prompt, seed=SEED + idx, causal_variant=True)
                elif method == 'GoT':
                    llm_output = run_got(OLLAMA_MODEL, question_prompt, seed=SEED + idx)
                elif method == 'ToT':
                    llm_output = run_tot(OLLAMA_MODEL, question_prompt, seed=SEED + idx)
                else:
                    raise ValueError(f'Unknown method: {method}')

                choice: Optional[str] = None
                extracted = ''

                if method in {'GoT', 'ToT'}:
                    try:
                        cdcr = ensure_cdcr_sft_ready()
                        extracted_output = cdcr['utils'].extract_abc_client_api(
                            cdcr['client'],
                            OLLAMA_MODEL,
                            question_prompt,
                            llm_output,
                        )
                        extracted = str(extracted_output).strip()
                        cand = cdcr['utils'].check_if_abc(extracted_output)
                        choice = cand if cand in {'a', 'b', 'c'} else None
                    except Exception as e:
                        extracted = f'ERROR_EXTRACT: {e}'
                        choice = None
                else:
                    choice = extract_choice_from_text(llm_output)
                    extracted = choice or ''

                if choice is None:
                    choice, extractor_out = force_extract_choice(
                        OLLAMA_MODEL, question_prompt, llm_output, seed=SEED + 10_000 + idx
                    )
                    extracted = extractor_out.strip()
                    if choice is None:
                        choice = 'a'  # fallback

                if CACHE_BY_QUESTION:
                    cache[question] = (llm_output, choice, extracted)
            except Exception as e:
                llm_output = f'ERROR: {e}'
                choice = 'a'
                extracted = ''

        pred_label = CHOICE_TO_ANSWER_LABEL.get(choice, 'more')
        is_correct = pred_label == gold_label

        upsert_row_by_id(
            out_csv,
            CSV_FIELDS,
            {
                'id': row_id,
                'question_type': str(row.get('format') or DATASET_NAME),
                'label': gold_label,
                'is_correct': str(is_correct),
                'answer': pred_label,
                'letter_answer': choice,
                'llm_output': llm_output,
                'llm_extracted_output': extracted,
                'model': OLLAMA_MODEL,
            },
        )

        if len(indices) <= 3:
            meta = ''
            if row.get('outcome_polarity') is not None:
                meta = f" outcome_polarity={row.get('outcome_polarity')} base={row.get('answer_label_base')}"
            print(f'[{method}] row_id={row_id}{meta} gold={gold_label} pred={pred_label} choice={choice}')
            if method in {'GoT', 'ToT'}:
                print(f'[{method}] LLM response:\n{llm_output}')
                if extracted:
                    print(f'[{method}] extracted: {extracted}')
                print('---')

        if (local_i + 1) % 10 == 0:
            print(f'[{method}] processed {local_i+1}/{len(indices)}')

    correct, total, acc = compute_accuracy(out_csv)
    print(f'[{method}] accuracy: {acc*100:.2f}% ({correct}/{total})')
    print('saved to:', out_csv)
    return out_csv

# %% [cell 14]
def select_single_index(rows):
    if SINGLE_ROW_LINE_NO is not None:
        target = int(SINGLE_ROW_LINE_NO)
        for i, r in enumerate(rows):
            if int(r.get('_line_no', -1)) == target:
                return i
        raise ValueError(f'No row found with _line_no={target}')

    if SINGLE_ROW_QUERY:
        q = str(SINGLE_ROW_QUERY).lower()
        for i, r in enumerate(rows):
            if q in str(r.get('question_stem', '')).lower():
                return i
        raise ValueError(f'No row found matching SINGLE_ROW_QUERY={SINGLE_ROW_QUERY!r}')

    return int(SINGLE_ROW_INDEX)


results = []
mode = str(RUN_MODE).strip().lower()
if mode.startswith('single'):
    idx = select_single_index(rows)
    r = rows[idx]
    print('Selected idx:', idx, 'line_no:', r.get('_line_no'))
    print('question_stem:', r.get('question_stem'))
    if 'outcome_polarity' in r or 'answer_label_base' in r:
        print('outcome_polarity:', r.get('outcome_polarity'), 'answer_label_base:', r.get('answer_label_base'))
    print('gold answer_label:', r.get('answer_label'))

    for method in METHODS:
        out_csv = run_method(method, rows, row_indices=[idx])
        correct, total, acc = compute_accuracy(out_csv)
        results.append({'method': method, 'accuracy': acc, 'correct': correct, 'total': total, 'csv': out_csv})
else:
    for method in METHODS:
        out_csv = run_method(method, rows)
        correct, total, acc = compute_accuracy(out_csv)
        results.append({'method': method, 'accuracy': acc, 'correct': correct, 'total': total, 'csv': out_csv})

pd.DataFrame(results).sort_values('accuracy', ascending=False)

# %% [cell 16]
inspect_method = 'GoT'  # change if needed
inspect_csv = get_output_csv_path(inspect_method)
df = pd.read_csv(inspect_csv)

wrong = df[df['is_correct'].astype(str).str.lower() == 'false']
wrong.head(10)
