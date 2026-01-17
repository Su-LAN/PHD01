"""
Run baselines on wiqa_causenet_1hop_mcqa.jsonl using Ollama.

Optional framework backends (for `--got_tot_backend framework`):
- Graph-of-Thoughts (`graph_of_thoughts`): `pip install graph-of-thoughts==0.0.2`
- Tree-of-Thoughts-LLM (`tot`): `pip install tree-of-thoughts-llm==0.1.0`
"""

import argparse
import csv
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ollama


ANSWER_LABEL_TO_CHOICE = {
    "more": "A",
    "less": "B",
    "no_effect": "C",
    "no effect": "C",
    "no_change": "C",
}
CHOICE_TO_ANSWER_LABEL = {"A": "more", "B": "less", "C": "no_effect"}


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
    m2 = re.search(r"(?:final answer|answer)\s*[:\-]?\s*(more|less|no[_ ]effect|no[_ ]change)\b", text, flags=re.IGNORECASE)
    if m2:
        label = m2.group(1).lower().replace(" ", "_")
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
        guidance = (
            "Guidance: Solve the causal effect direction question with explicit causal structure.\n"
            "Step 1) Identify the intervention and the outcome.\n"
            "Step 2) Construct a causal graph.\n"
            "Step 3) Choose the best option.\n"
        )
    else:
        guidance = (
            "Use chain-of-thought"
        )

    prompt = (
        f" {guidance}\n"
        f"{question_prompt}\n\n"
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
        "<Instruction> Split this question into 2 components.\n"
        f"{question_prompt}\n\n"
        "Output:\n"
        "{\n"
        '  "Component 1": "Causal graph: ...",\n'
        '  "Component 2": "Answer: ..."\n'
        "}\n"
    )
    split_out = ollama_chat(
        model,
        [{"role": "user", "content": split_prompt}],
        temperature=0,
        num_predict=384,
        seed=seed,
        timeout_s=300,
        retries=1,
    )
    split_obj = parse_json_object_from_text(split_out)
    if not split_obj:
        split_obj = {
            "Component 1": "Causal graph: identify variables and causal edges",
            "Component 2": "Answer: choose A/B/C",
        }

    components: List[Tuple[str, str]] = []
    for i in range(1, 3):
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
                "<Instruction> Analyze this component of the question. </Instruction>\n\n"
                f"{question_prompt}\n\n"
                f"Component: {comp_text}\n"
                + ("End with 'Final answer: A/B/C'." if idx == 2 else "")
            )
            out = ollama_chat(
                model,
                [{"role": "user", "content": prompt}],
                temperature=0,
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
        "<Instruction> Merge the component analyses into one answer.\n"
        "Your response MUST end with a single line: 'Final answer: A' or 'Final answer: B' or 'Final answer: C'. </Instruction>\n\n"
        f"{question_prompt}\n\n"
        "Component analyses:\n"
        f"Component 1:\n{component_analyses.get('Component 1','')}\n\n"
        f"Component 2:\n{component_analyses.get('Component 2','')}\n\n"
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


def run_got_graph_of_thoughts(model: str, question_prompt: str, *, seed: int) -> str:
    """
    GoT baseline implemented via the Graph-of-Thoughts framework (external dependency).

    Requires: `graph-of-thoughts` (import name: `graph_of_thoughts`).
    """
    try:
        from graph_of_thoughts import controller, operations, prompter as got_prompter, parser as got_parser
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Graph-of-Thoughts is not installed. Install it (e.g. `pip install graph-of-thoughts==0.0.2`) "
            "or add the project to PYTHONPATH."
        ) from e

    try:  # Compatibility: some versions don't expose `language_models.AbstractLanguageModel`
        from graph_of_thoughts.language_models import AbstractLanguageModel  # type: ignore
    except Exception:  # pragma: no cover
        AbstractLanguageModel = object  # type: ignore

    class OllamaGoTAdapter(AbstractLanguageModel):  # type: ignore[misc]
        def __init__(self, *, model_name: str, base_seed: int, cache: bool = False):
            self.model_name = model_name
            self.base_seed = base_seed
            self.cache = cache
            self._cache: Dict[Tuple[str, int], Any] = {}

        def query(self, query: str, num_responses: int = 1) -> Any:
            key = (query, num_responses)
            if self.cache and key in self._cache:
                return self._cache[key]

            if num_responses <= 1:
                result: Any = ollama_chat(
                    self.model_name,
                    [{"role": "user", "content": query}],
                    temperature=1.0,
                    num_predict=1024,
                    seed=self.base_seed,
                    timeout_s=600,
                    retries=2,
                )
            else:
                outs: List[str] = []
                for i in range(num_responses):
                    outs.append(
                        ollama_chat(
                            self.model_name,
                            [{"role": "user", "content": query}],
                            temperature=0,
                            num_predict=1024,
                            seed=self.base_seed + 31 * i,
                            timeout_s=600,
                            retries=2,
                        )
                    )
                result = outs

            if self.cache:
                self._cache[key] = result
            return result

        def get_response_texts(self, query_responses: Any) -> List[str]:
            if isinstance(query_responses, list):
                return [str(x) for x in query_responses]
            return [str(query_responses)]

    def causal_scoring_function(states: Any) -> List[float]:
        if isinstance(states, dict):
            states = [states]
        scores: List[float] = []
        for state in states:
            current = state.get("current", "") if isinstance(state, dict) else str(state)
            current_lower = current.lower()
            score = 0.5
            if "->" in current or "caus" in current_lower or "effect" in current_lower:
                score += 0.15
            if "because" in current_lower or "therefore" in current_lower:
                score += 0.1
            if extract_choice_from_text(current) is not None or (isinstance(state, dict) and state.get("answer")):
                score += 0.1
            if len(current) > 200:
                score += 0.1
            scores.append(min(score, 1.0))
        return scores

    def validation_function(state: Dict[str, Any]) -> bool:
        current = str(state.get("current", "") or "")
        if len(current) < 80:
            return False
        if extract_choice_from_text(current) is None:
            return False
        has_causal = any(term in current.lower() for term in ["cause", "caus", "effect", "->", "therefore", "because"])
        return has_causal

    def got_graph_structure() -> operations.GraphOfOperations:
        operations_graph = operations.GraphOfOperations()

        split = operations.Generate(1, 1)
        operations_graph.append_operation(split)

        for i in range(1, 4):
            component_id = f"Component {i}"

            selector = operations.Selector(
                lambda thoughts, cid=component_id: [t for t in thoughts if t.state.get("component") == cid]
            )
            selector.add_predecessor(split)
            operations_graph.add_operation(selector)

            analyze = operations.Generate(1, 3)
            analyze.add_predecessor(selector)
            operations_graph.add_operation(analyze)

            score = operations.Score(1, False, lambda st: causal_scoring_function([st])[0])
            score.add_predecessor(analyze)
            operations_graph.add_operation(score)

            keep_best = operations.KeepBestN(1, False)
            keep_best.add_predecessor(score)
            operations_graph.add_operation(keep_best)

        aggregate = operations.Aggregate(3)
        operations_graph.append_operation(aggregate)
        operations_graph.append_operation(operations.Score(1, False, lambda st: causal_scoring_function([st])[0]))
        operations_graph.append_operation(operations.KeepBestN(1, False))

        validate_improve_cls = getattr(operations, "ValidateAndImprove", None)
        if validate_improve_cls is not None:
            validate_improve = validate_improve_cls(
                num_samples=1,
                improve=True,
                num_tries=3,
                validate_function=validation_function,
            )
            operations_graph.append_operation(validate_improve)

        return operations_graph

    class CausenetGoTPrompter(got_prompter.Prompter):
        def __init__(self):
            super().__init__()
            self.split_prompt = (
                "<Instruction> Split this causal question into 2 components for analysis.\n"
                "Only output the JSON, no additional text! </Instruction>\n\n"
                "Question:\n{original}\n\n"
                "Output:\n"
                "{{\n"
                '  "Component 1": "Causal graph: ...",\n'
                '  "Component 2": "Final answer: A/B/C"\n'
                "}}\n"
            )
            self.component_prompt = (
                "<Instruction> Analyze this component of the question. </Instruction>\n\n"
                "Question:\n{original}\n\n"
                "Component: {component}\n"
                "Current: {current}\n\n"
                "Write a analysis.\n"
            )
            self.aggregate_prompt = (
                "<Instruction> Merge the component analyses into one complete answer.\n"
                "Your response MUST end with exactly one line: 'Final answer: A' or 'Final answer: B' or 'Final answer: C'.\n"
                "</Instruction>\n\n"
                "Question:\n{original}\n\n"
                "Component analyses:\n{component_analyses}\n"
            )
            self.improve_prompt_template = (
                "Improve and ends with exactly one line:\n"
                "'Final answer: A' or 'Final answer: B' or 'Final answer: C'.\n\n"
                "Question:\n{original}\n\n"
                "Current response:\n{current}\n\n"
                "Improved response:\n"
            )
            self.validate_prompt_template = (
                "Check if this response ends with exactly one valid final answer line.\n\n"
                "Question:\n{original}\n\n"
                "Response:\n{current}\n\n"
                "Reply only YES or NO:"
            )

        def generate_prompt(self, num_branches: int, **kwargs) -> str:
            original = kwargs.get("original", "")
            current = kwargs.get("current", "")
            component = kwargs.get("component", "")
            split_phase = kwargs.get("split_phase", False)
            if split_phase:
                return self.split_prompt.format(original=original)
            if component:
                return self.component_prompt.format(original=original, component=component, current=current)
            return f"Question:\n{original}\n\nAnswer with 'Final answer: A/B/C'."

        def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
            question = state_dicts[0].get("original", "") if state_dicts else ""
            analyses = ""
            for state in state_dicts:
                component = state.get("component", "")
                analysis = state.get("current", "")
                analyses += f"{component}:\n{analysis}\n\n"
            return self.aggregate_prompt.format(original=question, component_analyses=analyses.strip())

        def improve_prompt(self, **kwargs) -> str:
            return self.improve_prompt_template.format(original=kwargs.get("original", ""), current=kwargs.get("current", ""))

        def validation_prompt(self, **kwargs) -> str:
            return self.validate_prompt_template.format(original=kwargs.get("original", ""), current=kwargs.get("current", ""))

        def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
            return ""

    class CausenetGoTParser(got_parser.Parser):
        def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
            if not texts:
                return [state]

            new_states: List[Dict[str, Any]] = []
            if state.get("split_phase", False):
                for text in texts:
                    try:
                        components = parse_json_object_from_text(text)
                        if not components:
                            raise ValueError("No JSON found in split output")
                        for key, value in components.items():
                            new_state = state.copy()
                            new_state["component"] = key
                            new_state["current"] = str(value)
                            new_state["split_phase"] = False
                            new_states.append(new_state)
                    except Exception:
                        for i in range(1, 4):
                            new_state = state.copy()
                            new_state["component"] = f"Component {i}"
                            new_state["current"] = f"Component {i}"
                            new_state["split_phase"] = False
                            new_states.append(new_state)
                return new_states

            for text in texts:
                new_state = state.copy()
                new_state["current"] = text
                choice = extract_choice_from_text(text)
                if choice is not None:
                    new_state["answer"] = choice
                new_states.append(new_state)
            return new_states

        def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> Any:
            if not texts:
                return states[0] if states else {}

            new_states: List[Dict[str, Any]] = []
            for text in texts:
                merged_state = states[0].copy() if states else {}
                merged_state["current"] = text
                merged_state["component_analyses"] = [s.get("current", "") for s in states]
                choice = extract_choice_from_text(text)
                if choice is not None:
                    merged_state["answer"] = choice
                new_states.append(merged_state)
            return new_states

        def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
            if not texts:
                return {}
            improved_text = texts[0]
            choice = extract_choice_from_text(improved_text)
            return {"current": improved_text, "answer": choice or state.get("answer", "")}

        def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
            if not texts:
                return False
            response = texts[0].strip().upper()
            return "YES" in response[:20]

        def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
            return [0.5] * len(states)

    lm_adapter = OllamaGoTAdapter(model_name=model, base_seed=seed, cache=False)
    got_prompter_impl = CausenetGoTPrompter()
    got_parser_impl = CausenetGoTParser()
    operations_graph = got_graph_structure()

    initial_state: Dict[str, Any] = {
        "original": question_prompt,
        "current": "",
        "method": "got",
        "dataset_type": "causenet",
        "split_phase": True,
    }

    ctrl = controller.Controller(lm_adapter, operations_graph, got_prompter_impl, got_parser_impl, initial_state)
    ctrl.run()

    final_thoughts = ctrl.get_final_thoughts()
    if final_thoughts and isinstance(final_thoughts, list) and final_thoughts and isinstance(final_thoughts[0], list):
        final_thoughts = final_thoughts[0]
    if not final_thoughts:
        return "ERROR: GoT produced no final thoughts"

    final_thought = final_thoughts[0]
    final_state = getattr(final_thought, "state", {}) or {}
    return str(final_state.get("current", "") or "")


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
        "Given the question and the causal graph, generate TWO different reasoning chains.\n"
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


def run_tot_tree_of_thoughts(model: str, question_prompt: str, *, seed: int) -> str:
    """
    ToT baseline implemented via the Tree-of-Thoughts-LLM framework (external dependency).

    Requires: `tree-of-thoughts-llm` (import name: `tot`).
    """
    try:
        from tot.methods.bfs import solve
        from tot.tasks.base import Task
        import tot.models as tot_models
    except Exception as e:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Tree-of-Thoughts-LLM is not installed. Install it (e.g. `pip install tree-of-thoughts-llm==0.1.0`) "
            "or add the project to PYTHONPATH."
        ) from e

    call_counter = {"n": 0}
    ollama_model = model

    def _next_seed(offset: int) -> int:
        call_counter["n"] += 1
        return seed + call_counter["n"] * 97 + offset

    def gpt(
        prompt: str,
        model: str = model,
        temperature: float = 0.6,
        max_tokens: int = 1000,
        n: int = 1,
        stop: Optional[str] = None,
    ) -> List[str]:
        _ = (stop, model)  # unused, keep signature compatible
        outputs: List[str] = []
        for i in range(n):
            outputs.append(
                ollama_chat(
                    ollama_model,
                    [{"role": "user", "content": prompt}],
                    temperature=float(temperature),
                    num_predict=int(max_tokens),
                    seed=_next_seed(i),
                    timeout_s=600,
                    retries=2,
                )
            )
        return outputs

    def chatgpt(
        messages: List[Dict[str, str]],
        model: str = "",
        temperature: float = 0.6,
        max_tokens: int = 1000,
        n: int = 1,
        stop: Optional[str] = None,
    ) -> List[str]:
        _ = (stop, model)  # unused, keep signature compatible
        outputs: List[str] = []
        for i in range(n):
            outputs.append(
                ollama_chat(
                    ollama_model,
                    messages,
                    temperature=float(temperature),
                    num_predict=int(max_tokens),
                    seed=_next_seed(1000 + i),
                    timeout_s=600,
                    retries=2,
                )
            )
        return outputs

    tot_models.gpt = gpt
    tot_models.chatgpt = chatgpt

    class CausalReasoningTask(Task):
        def __init__(self, questions: List[str]):
            super().__init__()
            self.questions = questions
            self.steps = 3
            self.stops = [None, None, None]
            self.value_cache = {}
            self.current_step = 0

        def __len__(self) -> int:
            return len(self.questions)

        def get_input(self, idx: int) -> str:
            return self.questions[idx]

        def test_output(self, idx: int, output: str) -> Dict[str, Any]:
            _ = idx
            choice = extract_choice_from_text(output)
            if choice is None:
                return {"r": 0}
            return {"r": 1, "answer": CHOICE_TO_ANSWER_LABEL.get(choice, "")}

        def standard_prompt_wrap(self, x: str, y: str = "") -> str:
            _ = y
            return (
                f"{x}\n\n"
                "Construct a minimal graph, and give the final answer.\n"
                "Answer:"
            )

        def cot_prompt_wrap(self, x: str, y: str = "") -> str:
            if self.current_step == 0:
                if not y:
                    return (
                        x.strip()
                        + "\n\nGiven the question, make a graph to analyze. "
                        + "You do not need to provide the answer yet.\n"
                    )
                return x + "\n\n" + y
            if self.current_step == 1:
                return (
                    x
                    + "\n\n"
                    + y
                    + "\n\nBased on the graph, provide a process for the question. "
                    + "You do not need to provide the answer yet."
                )
            if self.current_step == 2:
                return (
                    x
                    + "\n\n"
                    + y
                    + "\n\nGive the answer to the question based on graph and the process."
                )
            return x + "\n\n" + y

        def propose_prompt_wrap(self, x: str, y: str = "") -> str:
            prompt = f"{x}\n\n"
            if y:
                prompt += f"Current reasoning:\n{y}\n\n"
            prompt += "Based on graph analysis, propose 2 different ways to continue the reasoning:\n"
            return prompt

        def value_prompt_wrap(self, x: str, y: str) -> str:
            prompt = "Evaluate the following causal reasoning:\n\n"
            prompt += f"{x}\n\n"
            prompt += f"Reasoning so far:\n{y}\n\n"
            prompt += "Will it lead to the right answer?\n"
            prompt += "Evaluation: "
            return prompt

        def value_outputs_unwrap(self, x: str, y: str, value_outputs: List[str]) -> float:
            _ = (x, y)
            values: List[float] = []
            for output in value_outputs:
                output_lower = output.lower()
                if "sure" in output_lower:
                    values.append(1.0)
                elif "likely" in output_lower:
                    values.append(0.5)
                elif "impossible" in output_lower:
                    values.append(0.001)
                else:
                    values.append(0.5)
            return sum(values) / len(values) if values else 0.5

    class Args:
        def __init__(self, model_name: str, temperature: float):
            self.backend = "ollama"
            self.model = model_name
            self.temperature = temperature
            self.method_generate = "sample"
            self.method_evaluate = "value"
            self.method_select = "greedy"
            self.n_generate_sample = 2
            self.n_evaluate_sample = 2
            self.n_select_sample = 2
            self.prompt_sample = "cot"

    task = CausalReasoningTask([question_prompt])
    args = Args(model, temperature=0.7)
    ys, _info = solve(args, task, 0, to_print=False)
    if ys and len(ys) > 0:
        return str(ys[0])
    return "ERROR: ToT produced no outputs"


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
    parser = argparse.ArgumentParser(description="Run baselines on wiqa_causenet_1hop_mcqa.jsonl using Ollama.")
    parser.add_argument("--input", type=str, default="Dataset\\wiqa_causenet_1hop2hop_mcqa_mix100_meta50.jsonl", help="Path to wiqa_causenet_1hop_mcqa.jsonl")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Ollama model name, e.g., llama3.1:8b")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory (CDCR-SFT-style)")
    parser.add_argument(
        "--methods",
        type=str,
        default="CoT,GoT,ToT,CausalCoT",
        help="Comma-separated methods: CoT,GoT,ToT,CausalCoT",
    )
    parser.add_argument(
        "--got_tot_backend",
        type=str,
        default="prompt",
        choices=["prompt", "framework"],
        help="GoT/ToT implementation: prompt (this file) or framework (Graph-of-Thoughts / Tree-of-Thoughts-LLM).",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all rows, otherwise cap.")
    parser.add_argument("--num_workers", type=int, default=1, help="Worker threads (1 = sequential).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling / prompting")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    model_dir = sanitize_dir_name(args.model.replace(":", "_"))
    dataset_dir = os.path.join(args.out_dir, model_dir, "causenet")
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
        method_tag = method
        if method in {"GoT", "ToT"}:
            method_tag = f"{method}_{args.got_tot_backend}"
        out_csv = os.path.join(dataset_dir, f"CoT_{method_tag}.csv")
        ensure_csv_header(out_csv, fieldnames)
        processed_ids = read_processed_ids(out_csv)

        print(f"\n=== Running {method_tag} on {len(rows)} rows (resume: {len(processed_ids)} done) ===")
        todo_items: List[Tuple[int, Dict[str, Any]]] = []
        first_idx_by_question: Dict[str, int] = {}
        for idx, row in enumerate(rows):
            row_id = str(idx)
            if row_id in processed_ids:
                continue
            todo_items.append((idx, row))
            question = str(row.get("question_stem", "")).strip()
            first_idx_by_question.setdefault(question, idx)

        cache: Dict[str, Tuple[str, str, str]] = {}
        inflight: Dict[str, threading.Event] = {}
        cache_lock = threading.Lock()
        tot_framework_lock = threading.Lock()

        def _process_item(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
            idx, row = item
            row_id = str(idx)

            question = str(row.get("question_stem", "")).strip()
            gold_label = str(row.get("answer_label", "")).strip().lower().replace(" ", "_")
            question_prompt = build_ddxplus_prompt(question)

            seed_idx = first_idx_by_question.get(question, idx)
            base_seed = int(args.seed) + int(seed_idx)
            extractor_seed = int(args.seed) + 10_000 + int(seed_idx)

            had_error = False
            while True:
                with cache_lock:
                    cached = cache.get(question)
                    if cached is not None:
                        llm_output, choice, extracted = cached
                        break
                    event = inflight.get(question)
                    if event is None:
                        event = threading.Event()
                        inflight[question] = event
                        leader = True
                    else:
                        leader = False

                if not leader:
                    event.wait()
                    continue

                try:
                    if method == "CoT":
                        llm_output = run_cot(args.model, question_prompt, seed=base_seed, causal_variant=False)
                    elif method == "CausalCoT":
                        llm_output = run_cot(args.model, question_prompt, seed=base_seed, causal_variant=True)
                    elif method == "GoT":
                        if args.got_tot_backend == "framework":
                            llm_output = run_got_graph_of_thoughts(args.model, question_prompt, seed=base_seed)
                        else:
                            llm_output = run_got(args.model, question_prompt, seed=base_seed)
                    elif method == "ToT":
                        if args.got_tot_backend == "framework":
                            with tot_framework_lock:
                                llm_output = run_tot_tree_of_thoughts(args.model, question_prompt, seed=base_seed)
                        else:
                            llm_output = run_tot(args.model, question_prompt, seed=base_seed)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    choice = extract_choice_from_text(llm_output)
                    extracted = choice or ""
                    if choice is None:
                        choice, extractor_out = force_extract_choice(
                            args.model, question_prompt, llm_output, seed=extractor_seed
                        )
                        extracted = extractor_out.strip()
                        if choice is None:
                            choice = "A"  # safe fallback (also majority class)

                    with cache_lock:
                        cache[question] = (llm_output, choice, extracted)
                except Exception as e:
                    msg = str(e)
                    if args.got_tot_backend == "framework" and method in {"GoT", "ToT"} and (
                        msg.startswith("Graph-of-Thoughts is not installed")
                        or msg.startswith("Tree-of-Thoughts-LLM is not installed")
                    ):
                        raise
                    had_error = True
                    llm_output = f"ERROR: {e}"
                    choice = ""
                    extracted = ""
                finally:
                    with cache_lock:
                        inflight.pop(question, None)
                    event.set()
                break

            pred_label = CHOICE_TO_ANSWER_LABEL.get(choice, "")
            is_correct = bool(pred_label) and pred_label == gold_label and not had_error
            return idx, {
                "id": row_id,
                "question_type": "ddxplus",
                "label": gold_label,
                "is_correct": str(is_correct),
                "answer": pred_label,
                "letter_answer": choice,
                "llm_output": llm_output,
                "llm_extracted_output": extracted,
                "model": args.model,
            }

        num_workers = int(getattr(args, "num_workers", 1) or 1)
        if num_workers < 1:
            num_workers = 1

        if num_workers == 1:
            for idx, out_row in map(_process_item, todo_items):
                append_row(out_csv, fieldnames, out_row)
                if (idx + 1) % 10 == 0:
                    print(f"[{method}] processed {idx+1}/{len(rows)}")
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for idx, out_row in executor.map(_process_item, todo_items):
                    append_row(out_csv, fieldnames, out_row)
                    if (idx + 1) % 10 == 0:
                        print(f"[{method}] processed {idx+1}/{len(rows)}")

        correct, total, acc = compute_accuracy(out_csv)
        print(f"[{method}] accuracy: {acc*100:.2f}% ({correct}/{total}) -> {out_csv}")


if __name__ == "__main__":
    main()
