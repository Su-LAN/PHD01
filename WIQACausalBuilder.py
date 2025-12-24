# -*- coding: utf-8 -*-
"""
WIQA Reasoning Module
Process WIQA questions for reasoning tasks
"""

import re
import json
import ast
from typing import Dict, Optional, Tuple, List, Any, Set
from dataclasses import dataclass
import ollama


@dataclass
class VariableRelation:
    relation_type: str
    reasoning: str = ""


class WIQACausalBuilder:
    """
    A class to build causal graphs from WIQA dataset entries.
    (Z, CAUSES, X)
    (X, CAUSES, Y)
    (W, CAUSES, D)
    (Y, CAUSES, A)  # if Y_affects_outcome = "more"
    (Y, CAUSES, D)  # if Y_affects_outcome = "less"
    """

    def __init__(
        self, 
        datapoint: Dict[str, Any], 
        model_name: str = "llama3.1:8b"
        ):
        self.datapoint = datapoint
        self.model_name = model_name
        self.question = datapoint.get('question_stem', '')
        self.answers = datapoint.get('answer_label', [])
        # 各类节点集合（后续因果图构建时使用）
        self.X = []  # 扰动相关节点（包括 X_q）
        self.Y = []
        self.Z = []
        self.A = []
        self.D = []
        self.W = []
        self.U = []
        self.V = []
        # 由 LLM 生成的机制链（X -> ... -> outcome_base）
        self.mechanism_chain: List[Dict[str, str]] = []
        # 记录由 close_hits 桥接产生的边，用于后续路径过滤
        self.bridge_edges: Set[Tuple[str, str, str]] = set()
        
    def _call_llm(self, prompt: str) -> str:
        """调用LLM获取响应，强制 temperature=0 以保证确定性"""
        # options 参数取决于 ollama 的 python 库版本，通常如下：
        response = ollama.generate(
            model=self.model_name, 
            prompt=prompt,
            options={
                "temperature": 0.0, # 消除随机性
                "seed": 42,         # 固定种子
                "num_predict": 1024  # 限制输出长度，防止废话
            }
        )
        return response['response'].strip()
    def _clean_response(self, response: str) -> str:
        # 清理可能的 markdown 代码块标记
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]  # 移除 ```json
        elif response.startswith('```'):
            response = response[3:]  # 移除 ```
        if response.endswith('```'):
            response = response[:-3]  # 移除末尾的 ```
        response = response.strip()
        
        return response
    
    
    def get_question(self) -> str:
        """
        Retrieve the question stem from the datapoint.
        """
        return self.question
    
    def get_answers(self) -> List[str]:
        """
        Retrieve the answer labels from the datapoint.
        """
        return self.answers
    
    def extract_start_entity(self) -> Optional[Dict[str, str]]:
        """
        [Modified for Llama 3.1 8B]
        Robust Extraction Pipeline with "Truth Check" and "Refinement".
        
        1. Raw Extraction: Get text segments.
        2. Truth Check: Verify if direction words actually exist in text.
        3. Refinement: Convert sentence to concise scientific noun phrase.
        """
        
        # =======================================================
        # Stage 1: Raw Extraction (获取原始片段和方向)
        # =======================================================
        prompt_1 = f"""
You are a STRICT string-matching text extractor.
Your job is ONLY to copy substrings and detect certain exact words.
You MUST NOT infer meanings or guess intentions.

Task: Extract the Cause and Outcome segments from the question.

Question: "{self.question}"

Definitions:
- CAUSE_EVENT: the text after the word "suppose", up to the word "affect".
- OUTCOME_TEXT_RAW: the text after the word "affect" (trim leading spaces and punctuation, but keep all words).
- DIRECTION: ONLY based on exact tokens in OUTCOME_TEXT_RAW:
    * If OUTCOME_TEXT_RAW contains the WHOLE WORD "more" or "greater" -> outcome_direction = "MORE"
    * If OUTCOME_TEXT_RAW contains the WHOLE WORD "less" or "fewer"   -> outcome_direction = "LESS"
    * Otherwise -> outcome_direction = "NONE"
- NEGATION: if OUTCOME_TEXT_RAW contains "not", "no", or "never" as whole words -> is_negated = "true", else "false".

VERY IMPORTANT HARD RULES:
- You MUST NOT use synonyms or meaning to decide the direction.
- Comparative adjectives like "smaller", "bigger", "higher", "lower", "reduced", "increased", etc. DO NOT count as direction words.
- If OUTCOME_TEXT_RAW contains only words like "smaller", "bigger", etc., you MUST output outcome_direction = "NONE".
- You MUST ignore the cause when deciding the direction; ONLY look at OUTCOME_TEXT_RAW.

EXAMPLES (follow them exactly):

Example 1:
Q: "suppose people eat less fish happens, how will it affect more fish."
OUTCOME_TEXT_RAW = "more fish"
Contains "more" -> outcome_direction = "MORE"

Example 2:
Q: "suppose people drink more water happens, how will it affect less headaches."
OUTCOME_TEXT_RAW = "less headaches"
Contains "less" -> outcome_direction = "LESS"

Example 3:
Q: "suppose tectonic plates are dormant happens, how will it affect smaller mountains."
OUTCOME_TEXT_RAW = "smaller mountains"
It does NOT contain the exact whole words "more", "less", "greater", or "fewer".
Therefore: outcome_direction = "NONE"

Now perform the extraction for the current question.

Output ONLY valid JSON (no extra text):
{{
  "cause_event": "<copied text after 'suppose' and before 'affect'>",
  "outcome_text_raw": "<copied text after 'affect'>",
  "outcome_direction": "MORE" or "LESS" or "NONE",
  "is_negated": "true" or "false"
}}
""".strip()

        # 调用 Stage 1
        response_1 = self._call_llm(prompt_1)
        
        # 解析 Stage 1 结果 (使用 Findall + Reversed 应对多块 JSON)
        import re
        data_1 = {}
        matches = re.findall(r'\{[\s\S]*?\}', response_1)
        for match in reversed(matches):
            try:
                temp = json.loads(match)
                if "outcome_text_raw" in temp:
                    data_1 = temp
                    break
            except: continue
            
        raw_cause = (data_1.get("cause_event") or "").strip()
        raw_outcome = (data_1.get("outcome_text_raw") or "").strip()
        # LLM 偶发会把 "suppose" 也抄进 cause_event/outcome_text_raw；这里做确定性清洗。
        raw_cause = re.sub(r"^\s*suppose\s+", "", raw_cause, flags=re.IGNORECASE).strip()
        raw_outcome = re.sub(r"^\s*suppose\s+", "", raw_outcome, flags=re.IGNORECASE).strip()
        raw_dir_str = (data_1.get("outcome_direction") or "NONE").strip().upper()
        is_negated = str(data_1.get("is_negated", "false")).strip().lower() == "true"

        # =======================================================
        # [CRITICAL FIX] Direction Validation (方向验真)
        # =======================================================
        # 目的：防止模型幻觉出原句不存在的 LESS/MORE
        # 逻辑：LLM 说是 X，我们就去原句里找 X。找不到就当它是瞎说的。
        
        valid_more_words = ["MORE", "GREATER", "INCREASE", "HIGHER", "LARGER"]
        valid_less_words = ["LESS", "FEWER", "DECREASE", "LOWER", "SMALLER", "REDUCE"]
        
        outcome_upper = raw_outcome.upper() 
        final_direction = "NONE" # 默认为 NONE
        
        # 只有当原句里真的包含对应词汇时，才采纳 LLM 的建议
        if any(v in raw_dir_str for v in valid_more_words):
            if any(w in outcome_upper for w in valid_more_words):
                final_direction = "MORE"
        elif any(v in raw_dir_str for v in valid_less_words):
            if any(w in outcome_upper for w in valid_less_words):
                final_direction = "LESS"
        
        # 如果代码运行到这里 final_direction 依然是 NONE，说明原句无方向词，幻觉已被修正。

        # =======================================================
        # Stage 2: Variable Refinement (把句子变短语)
        # =======================================================
        
        refined_base = raw_outcome 
        
        if raw_outcome:
            # 专门的 Prompt，负责把句子变成名词短语 (Noun Phrase)
            prompt_2 = f"""
You are a Scientific Editor.
Task: Convert a descriptive sentence into a concise Scientific Noun Phrase.

Input Sentence: "{raw_outcome}"
Current Direction: "{final_direction}"

Rules:
1. **Remove Direction**: Delete words like '{final_direction}', 'more', 'less' from the phrase.
2. **Noun Phrase Conversion**: Convert actions/verbs into nouns.
   - "lava is pushed out" -> "lava ejection"
   - "tadpoles develop legs" -> "tadpole metamorphosis"
   - "water gets hot" -> "water temperature"
   - "cracks occur" -> "cracks"
3. **Format**: The result must be a short phrase (2-4 words), NOT a sentence.

Output ONLY JSON:
{{
  "scientific_noun_phrase": "..."
}}
"""
            # 调用 Stage 2
            response_2 = self._call_llm(prompt_2)
            
            # 解析 Stage 2 结果
            try:
                matches_2 = re.findall(r'\{[\s\S]*?\}', response_2)
                for match in reversed(matches_2):
                    try:
                        data_2 = json.loads(match)
                        # 尝试获取 scientific_noun_phrase，兼容旧 key 以防万一
                        candidate = data_2.get("scientific_noun_phrase", data_2.get("scientific_variable"))
                        if candidate:
                            refined_base = candidate.strip()
                            break
                    except: continue
            except:
                print(f"[Refine] JSON parse failed, keeping raw: {raw_outcome}")

        # =======================================================
        # Python Post-Processing (最后的兜底清洗)
        # =======================================================
        
        # 确保 Refiner 没有把 "more" 抄进去
        lowered = refined_base.lower()
        prefixes_to_strip = [
            "more ", "less ", "fewer ", 
            "a greater amount of ", "a smaller amount of ", 
            "no ", "not ", "never ", "without ", "lack of ", "fail to "
        ]
        for prefix in prefixes_to_strip:
            if lowered.startswith(prefix):
                refined_base = refined_base[len(prefix):].strip()
                break

        # =======================================================
        # 最终赋值
        # =======================================================
        self.cause_event = raw_cause
        self.X.append(raw_cause)
        
        self.outcome_event = raw_outcome
        self.outcome_base = refined_base # 精加工后的短语
        self.Y = refined_base
        
        self.outcome_direction_in_question = final_direction # 验真后的方向
        
        # 否定逻辑兜底 (如果原句没有 not/no，强制设为 False)
        explicit_negation_words = ["not ", "no ", "never ", "n't ", "without ", "fail "]
        if final_direction == "LESS" and is_negated and not any(w in raw_outcome.lower() for w in explicit_negation_words):
             is_negated = False
        self.outcome_is_negated = is_negated

        if refined_base:
            self.A.append(f"MORE {refined_base}")
            self.D.append(f"LESS {refined_base}")

        return {
            "cause_event": raw_cause,
            "outcome_event": raw_outcome,
            "outcome_base": refined_base,
            "outcome_direction_in_question": final_direction,
            "outcome_is_negated": is_negated,
        }
    
    def _rank_frontier_nodes(self, nodes: List[str], target: str, top_k: int = 5) -> List[str]:
        """
        Beam Search Scorer: Select the top_k nodes most likely to lead to the target.
        """
        # 如果节点数量少于等于 top_k，直接全部保留，不需要筛选
        if len(nodes) <= top_k:
            return nodes
            
        nodes_str = ", ".join([f"'{n}'" for n in nodes])
        
        prompt = f"""
You are a strategic pathfinder in a causal graph.
Start Node: "{self.cause_event}"
Target Node: "{target}"

Current Candidates: [{nodes_str}]

Task: Select the top {top_k} candidates that are semantically closest or most likely to be an intermediate step towards the Target Node.
Ignore generic or irrelevant concepts (e.g., "Size", "Stability" unless relevant).

Output ONLY JSON:
{{
  "best_candidates": ["node_A", "node_B", ...]
}}
"""
        response = self._call_llm(prompt)
        
        # 解析
        import re
        try:
            match = re.search(r'\{[\s\S]*?\}', self._clean_response(response))
            if match:
                data = json.loads(match.group(0))
                best = data.get("best_candidates", [])
                # 过滤掉不存在的节点（防止幻觉）
                valid_best = [n for n in best if n in nodes]
                if valid_best:
                    print(f"[Beam Search] Pruned frontier from {len(nodes)} to {len(valid_best)} nodes.")
                    return valid_best
        except:
            pass
            
        # 兜底：如果 LLM 失败，默认返回前 k 个
        return nodes[:top_k]
    
    def map_effect_on_base_to_wiqa_label(
        self,
        effect_on_base: str,
    ) -> str:
        """
        Map the effect on the base outcome variable to the WIQA answer label,
        taking into account:
          - outcome_direction_in_question (MORE / LESS / NONE / ...)
          - outcome_is_negated (whether the outcome phrase is a NOT event)

        Parameters
        ----------
        effect_on_base : str
            One of "more", "less", "no_effect".
            This describes how CAUSE_EVENT affects the base variable
            self.outcome_base (without MORE/LESS/NOT).

        Returns
        -------
        str
            One of "more", "less", "no_effect",
            matching WIQA's answer_label semantics.

        核心语义：
        - LLM / 图只需要判断基础变量 outcome_base 本身的增减（effect_on_base）。
        - 题目里的 outcome_event 可能包含 MORE/LESS 或显式否定（NOT/NO/...）。
        - 这里统一把「基础变量的增减」 + 「题目的方向词」 + 「是否否定事件」
          映射到最终 WIQA 标签，避免 LLM 在 "LESS X" 这类短语上直接做含糊判断。
        """
        e = (effect_on_base or "").strip().lower().replace(" ", "_")
        d = (getattr(self, "outcome_direction_in_question", "NONE") or "NONE").strip().upper()
        is_neg = bool(getattr(self, "outcome_is_negated", False))

        # 如果基础变量不确定或无效，直接视为 no_effect
        # （否则 no_effect 会被误当成 less，因为下面 delta 的 else 分支会给 -1）
        if e in {"no_effect", "noeffect", "none", "unknown", ""}:
            return "no_effect"
        if e not in {"more", "less"}:
            return "no_effect"


        # Encode base sign as +1 / -1
        delta = 1 if e == "more" else -1

        # Helper for direction sets
        def _is_more_dir(dir_str: str) -> bool:
            return dir_str in {"MORE", "GREATER", "INCREASE", "HELPING", "NONE", ""}

        def _is_less_dir(dir_str: str) -> bool:
            return dir_str in {"LESS", "SMALLER", "DECREASE", "HURTING"}

        # === Case 1: outcome phrase is NOT negated ===
        #   - Outcome event ~ "MORE X", "LESS X", or just "X"
        if not is_neg:
            # 问的是“基础变量更多 / 没有方向”的情况
            if _is_more_dir(d):
                # 基础变量变多 -> more；变少 -> less
                return "more" if delta > 0 else "less"

            # 问的是“LESS base”（题目本身已经是“更少”的方向）
            if _is_less_dir(d):
                # 如果基础变量变少：
                #   → “LESS <base>” 这个事件会更常发生 → 选项应该是 more
                if delta < 0:
                    return "more"
                # 如果基础变量变多：
                #   → "LESS <base>" 事件发生得更少 → 选项应该是 less
                if delta > 0:
                    return "less"
                return "no_effect"

            # 方向未知时的兜底：按基础变量增减直接映射
            return "more" if delta > 0 else "less"

        # === Case 2: outcome phrase is a NOT event, e.g. "seeds will not be transported" ===
        # 设基础变量为 B；题目关心的是 NOT(B) 事件的多少。
        # 如果 B 增加 (delta = +1)  → NOT(B) 减少
        # 如果 B 减少 (delta = -1)  → NOT(B) 增加
        if is_neg:
            # Sign for the NOT(B) event
            delta_event = -delta

            if _is_more_dir(d):
                # 问的是 "MORE NOT(B) ?"
                if delta_event > 0:
                    return "more"
                if delta_event < 0:
                    return "less"
                return "no_effect"

            if _is_less_dir(d):
                # 问的是 "LESS NOT(B) ?"
                if delta_event > 0:
                    # NOT(B) 实际上增加 → LESS NOT(B) 事件更少 → 选项 less
                    return "less"
                if delta_event < 0:
                    # NOT(B) 实际上减少 → LESS NOT(B) 事件更常发生 → 选项 more
                    return "more"
                return "no_effect"

            # 方向未知时的兜底：按 NOT(B) 的增减映射
            if delta_event > 0:
                return "more"
            if delta_event < 0:
                return "less"
            return "no_effect"
  
    def find_causal_relations(
        self,
        X: str,
        Y: Optional[str] = None,
        avoid_nodes: Optional[List[str]] = None, # <--- 新增参数
        max_relations: int = 3,
    ) -> Dict[str, Any]:
        """
        [Modified]
        Includes 'avoid_nodes' in Prompt to enforce DAG property (No Cycles).
        """
        X = (X or "").strip()
        Y = (Y or "").strip() if Y is not None else None
        result: Dict[str, Any] = {"triples": [], "new_entities": set()}
        if not X:
            return result

        target_hint = Y if Y else "NONE"
        
        # 构建禁止名单字符串
        avoid_str = "NONE"
        if avoid_nodes and len(avoid_nodes) > 0:
            # 为了节省 Token，如果列表太长，只取最近的 20 个（通常循环发生在局部）
            truncated_avoid = avoid_nodes[-20:] 
            avoid_str = ", ".join([f'"{n}"' for n in truncated_avoid])
        
        prompt = f"""
You are a causal edge finder.

Input:
- CAUSE_NODE (X): "{X}"
- TARGET_HINT (Y): "{target_hint}"

Context (Existing Nodes):
- The following events have already happened or are upstream.
- **FORBIDDEN LIST**: [{avoid_str}]

Task:
- Propose up to {max_relations} SINGLE-HOP causal effects starting from X.
- **TARGETED EXPANSION (IMPORTANT)**:
  * If TARGET_HINT (Y) is NOT "NONE", you MUST expand *toward Y*.
  * Prefer effect nodes that are plausible intermediate steps on a path from X to Y.
  * Choose effect nodes that are semantically closer to Y than random associations.
  * Avoid off-topic branches. If you cannot find any reasonable effect that helps reach Y, output an EMPTY list.
- **DAG CONSTRAINT (NO LOOPS)**: Do NOT generate any node that is semantically similar to the FORBIDDEN LIST. The graph must be Acyclic.
- **CRITICAL RULE**: The output node MUST be a NEUTRAL NOUN or NOUN PHRASE.
  Do NOT output full sentences. Do NOT include "more/less" in the node text.

Signs:
- Use "INCREASES" when X causes the effect variable to increase/rise.
- Use "DECREASES" when X causes the effect variable to decrease/fall.

Output ONLY JSON:
{{
  "triples": [
    ["{X}", "INCREASES" | "DECREASES", "<neutral noun phrase>"],
    ...
  ]
}}
""".strip()

        response = self._call_llm(prompt)
        response = self._clean_response(response)

        try:
            # 优先正则提取
            matches = re.findall(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', response)
            if matches:
                 data = {"triples": [list(m) for m in matches]}
            else:
                 data = json.loads(response)
        except Exception:
            data = {"triples": []}

        triples = []
        new_entities = set()
        seen = set()

        raw_triples = data.get("triples", [])
        for t in raw_triples:
            if len(t) != 3: continue
            h, r, tail = t
            if str(h).strip() != X: continue
            
            # Relation Normalization
            rel = str(r).strip().upper()
            if "INCREASE" in rel: rel = "INCREASES"
            elif "DECREASE" in rel: rel = "DECREASES"
            elif rel == "RESULTS_IN": rel = "INCREASES"
            elif rel == "NOT_RESULTS_IN": rel = "DECREASES"
            
            if rel not in ("INCREASES", "DECREASES"): continue
            
            tail_clean = str(tail).strip()
            if not tail_clean: continue
            
            # Python 层双重检查：防止 LLM 忽略指令生成了 Forbidden 里的词
            if avoid_nodes and tail_clean in avoid_nodes:
                continue

            key = (X, rel, tail_clean.lower())
            if key in seen: continue
            seen.add(key)
            
            triples.append((X, rel, tail_clean))
            new_entities.add(tail_clean)
            if len(triples) >= max_relations: break
        
        # 强制排序
        triples.sort(key=lambda x: (x[0], x[1], x[2]))

        result["triples"] = triples
        result["new_entities"] = new_entities
        return result

    def find_target_parents(
        self,
        target_Y: str,
        max_parents: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Target-seeding: propose a small set of DIRECT parent causes of Y.

        Why:
        - Forward-only expansion often fails to ever "touch" the outcome domain (e.g., cities -> tadpoles).
        - Seeding a few plausible parents of Y (e.g., water pollution -> tadpole development)
          makes the search much more robust without heavy prompt grounding.

        Returns:
        - A list of edge dicts: {head, relation, tail, source="target_seed", cf_* ...}
        """
        Y = (target_Y or "").strip()
        if not Y:
            return []

        prompt = f"""
You are a causal edge finder.

TARGET NODE (Y): "{Y}"

Task:
- Propose up to {max_parents} DIRECT causes X of Y.
- Each X must be a NEUTRAL NOUN or NOUN PHRASE (no "more/less", no full sentences).
- Use "INCREASES" when increasing X tends to increase Y.
- Use "DECREASES" when increasing X tends to decrease Y.
- Prefer mechanistic/scientific variables. Avoid generic placeholders like "factor", "change", "process".

Output ONLY JSON:
{{
  "triples": [
    ["<neutral noun phrase>", "INCREASES" | "DECREASES", "{Y}"],
    ...
  ]
}}
""".strip()

        raw = self._call_llm(prompt)
        cleaned = self._clean_response(raw or "")

        # Parse triples robustly (regex-first).
        triples: List[Tuple[str, str, str]] = []
        try:
            matches = re.findall(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', cleaned)
            if matches:
                triples = [(a.strip(), b.strip(), c.strip()) for a, b, c in matches]
            else:
                data = json.loads(cleaned)
                for t in data.get("triples", []) or []:
                    if isinstance(t, (list, tuple)) and len(t) == 3:
                        triples.append((str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()))
        except Exception:
            triples = []

        edges: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for h, r, t in triples:
            if not h or not r or not t:
                continue
            if t.strip().lower() != Y.lower():
                continue
            rel = str(r).strip().upper()
            if "INCREASE" in rel:
                rel = "INCREASES"
            elif "DECREASE" in rel:
                rel = "DECREASES"
            if rel not in {"INCREASES", "DECREASES"}:
                continue
            key = (h, rel, Y)
            if key in seen:
                continue
            seen.add(key)

            # Soft counterfactual verification (metadata only).
            cf_info = {"is_valid": True, "confidence": 0.5, "reasoning": ""}
            try:
                cf_info = self._verify_edge_counterfactual(h, rel, Y) or cf_info
            except Exception:
                cf_info = cf_info
            try:
                cf_conf = float(cf_info.get("confidence", 0.5))
            except Exception:
                cf_conf = 0.5
            cf_conf = max(0.0, min(1.0, cf_conf))

            edges.append({
                "head": h,
                "relation": rel,
                "tail": Y,
                "is_substitute": False,
                "substitute_confidence": 0.0,
                "cf_is_valid": bool(cf_info.get("is_valid", True)),
                "cf_confidence": cf_conf,
                "cf_reasoning": str(cf_info.get("reasoning", "")).strip(),
                "source": "target_seed",
            })
            if len(edges) >= max_parents:
                break

        # Deterministic ordering for reproducibility.
        edges.sort(key=lambda e: (e.get("head", ""), e.get("relation", ""), e.get("tail", "")))
        return edges
    
    def expand_toward_target(
        self,
        start_X: str,
        target_Y: Optional[str] = None,
        max_depth: int = 3,
        max_relations_per_node: int = 3,
        max_nodes: int = 50,
        beam_width: int = 5
    ) -> Dict[str, Any]:
        """
        [Modified] 
        1. Passes 'visited' to prevent cycles (DAG).
        2. Uses Beam Search to prune frontier.
        3. [Updated] Counterfactual / substitution checks are NOT performed during BFS expansion.
           They are deferred to the Top-K path verification stage for robustness.
        """
        start = (start_X or "").strip()
        target = (target_Y or "").strip() if target_Y else None

        if not start:
            return {"triples": [], "visited": set(), "found_target": False, "depth_reached": 0, "close_hits": []}

        visited: Set[str] = set([start])
        triples_acc: List[Tuple[str, str, str]] = []
        seen_triples: Set[Tuple[str, str, str]] = set()
        # Cache each node's relation to target (lowercased key) for downstream drift filtering.
        node_rel_to_target: Dict[str, str] = {}
        
        frontier = [start]
        found = False
        depth = 0
        close_hits: List[Dict[str, Any]] = []

        while frontier and depth < max_depth and len(visited) < max_nodes:
            next_frontier: List[str] = []
            
            # 将 visited 转换为 list 传给 LLM 作为禁止名单
            current_avoid_list = list(visited)
            
            for node in frontier:
                # 1. 发现关系 (带禁止名单)
                rels = self.find_causal_relations(
                    node, 
                    target, 
                    avoid_nodes=current_avoid_list, 
                    max_relations=max_relations_per_node
                )
                
                for h, r, tail in rels.get("triples", []):
                    relation_to_target: Optional[Dict[str, Any]] = None
                    bfs_eq = ""
                    hit_weight = 0.0
                    if target:
                        relation_to_target = self.is_same_variable(tail, target, self.question)
                        bfs_eq = str(relation_to_target.get("bfs_equivalence", "")).strip().lower()
                        try:
                            node_rel_to_target[str(tail).strip().lower()] = bfs_eq
                        except Exception:
                            pass

                        # Base weights for sign aggregation / description focus.
                        if bfs_eq in {"exact_target", "close_hit"}:
                            hit_weight = 1.0
                        elif bfs_eq == "bridge_candidate":
                            hit_weight = 0.5
                        # Light adjustments from other dimensions.
                        if str(relation_to_target.get("core_entity_relation", "")).strip().lower() == "overlapping_entities":
                            hit_weight *= 0.8
                        if str(relation_to_target.get("causal_or_structural_relation", "")).strip().lower() == "correlated_or_confounding":
                            hit_weight *= 0.8
                        hit_weight = max(0.0, min(1.0, hit_weight))
                    # NOTE:
                    # - Counterfactual verification and substitution detection are deferred to
                    #   the Top-K path verification stage. Doing it here (edge-level) is brittle
                    #   and can wipe out the entire frontier, causing 0-path failures.
                    sub_is = False
                    sub_conf = 0.0
                    triple_meta = {
                        "head": h,
                        "relation": r,
                        "tail": tail,
                        "is_substitute": False,
                        "substitute_confidence": sub_conf,
                        # Filled later during Top-K path verification:
                        "cf_is_valid": True,
                        "cf_confidence": 0.5,
                        "cf_reasoning": "",
                    }
                    # ===========================

                    triple_key = (h, r, tail)
                    if triple_key not in seen_triples:
                        triples_acc.append(triple_meta)
                        seen_triples.add(triple_key)

                    if tail not in visited:
                        visited.add(tail)
                        next_frontier.append(tail)
                        current_avoid_list.append(tail) 
                    
                    if target:
                        if tail.lower() == target.lower() or bfs_eq == "exact_target":
                            found = True
                    
                    if target and relation_to_target:
                        if bfs_eq in {"exact_target", "close_hit", "bridge_candidate"}:
                            if not any(item.get("node") == tail for item in close_hits):
                                close_hits.append({
                                    "node": tail,
                                    "depth": depth + 1,
                                    "bfs_equivalence": bfs_eq,
                                    "weight": hit_weight,
                                    "core_entity_relation": relation_to_target.get("core_entity_relation", ""),
                                    "quantity_relation": relation_to_target.get("quantity_relation", ""),
                                    "causal_or_structural_relation": relation_to_target.get("causal_or_structural_relation", ""),
                                    "explanation": relation_to_target.get("explanation", ""),
                                })
                
                if found: break
            
            if found:
                depth += 1
                break
            
            # Beam Search Pruning (保留最相关的节点)
            if target and next_frontier:
                frontier = self._rank_frontier_nodes(next_frontier, target, top_k=beam_width)
            else:
                frontier = next_frontier
            
            depth += 1

        return {
            "triples": triples_acc,
            "visited": visited,
            "found_target": found,
            "depth_reached": depth,
            "close_hits": close_hits,
            "node_rel_to_target": node_rel_to_target,
        }
        

    def _check_causal_relevance(self, cause: str, effect: str, context: str) -> bool:
        """
        Universal Causal Validator (context-aware).

        Goal:
            Distinguish between real physical/logical derivation
            and fake semantic association, based on commonsense knowledge
            and the question context.

        Core logic:
            Accept if ANY of the following holds:
            - State Exclusion (strong negative)
            - Mechanism
            - Indirect dependency / necessary resource
        """
        if not cause or not effect:
            # If either side is empty, we treat it as non-usable edge.
            return False

        prompt = f"""
You are a Scientific Logic Judge.

You MUST judge causality based on commonsense scientific knowledge and the question context.

Question Context:
{context}

Cause: "{cause}"
Effect: "{effect}"

Task:
Does knowing that the Cause happened help us predict the state (presence / absence / increase / decrease) of the Effect?

Criteria for ACCEPTANCE (is_valid_link = true) – pass if ANY is true:

1. State Exclusion (Strong Negative):
   - If the Cause is present, does it make the Effect impossible or strongly suppressed?
   - Examples:
     - "Strong sunlight on the ground" -> "Rain falling now" (sun implies no rain clouds → no rain).
     - "Damaged eardrum" -> "Sound converted into nerve signals" (damage prevents conversion).

2. Mechanism:
   - Is there a plausible physical/logical process connecting them based on scientific knowledge?
   - e.g. "More water in soil" -> "More plant growth" (resource).

3. Indirect Dependency / Necessary Resource:
   - The Cause provides a key resource or necessary precondition for the Effect.
   - e.g. "Water" -> "Steam", "Electricity" -> "Light from bulbs".

Criteria for REJECTION (is_valid_link = false):

- Completely unrelated topics.
- Pure coincidence or loose association with no usable predictive power.
- Strong cross-domain jumps not supported by scientific knowledge.

Decision:
Is this a valid causal link based on scientific reasoning?

Output ONLY JSON:
{{"is_valid_link": true/false, "reasoning": "short explanation"}}
        """.strip()

        try:
            response = self._call_llm(prompt)
            response = self._clean_response(response)
            data = json.loads(response)
            is_valid = bool(data.get("is_valid_link", False))
            status = "✅ ACCEPTED" if is_valid else "❌ REJECTED"
            print(f"[Relevance] {status}: '{cause}' -> '{effect}'")
            return is_valid
        except Exception as e:
            # Fail-open: if the judge fails, don't kill potentially valid paths.
            print(f"_check_causal_relevance error: {e}")
            return True

    # Counterfactual test used by bridge_close_hits to detect 'substitution' edges.
    # We imagine removing the intermediate node C and see if the system
    # would need MORE of Y (or an alternative) to compensate; such edges
    # are treated as substitution rather than straightforward causal links.
    def _check_counterfactual_substitution(self, cause: str, effect: str, context: str) -> Dict[str, Any]:
        """
        使用反事实逻辑检测是否为"替代/竞争"关系，并返回置信度。
        """
        prompt = f"""
You are a Logic Consistency Checker.
We are distinguishing between two types of relationships: SUBSTITUTION vs. DEPENDENCY.

Question Context: {context}
A = "{cause}"
B = "{effect}"

Assumption: The graph currently says "More A -> More B".

TEST: Imagine A is completely REMOVED.
To achieve the Goal, do we now need **MORE** of B to compensate?

Type 1: SUBSTITUTION (The "Spare Tire" Logic) - RETURNS TRUE
- "A is gone, so I must use B instead."
- Example: No Pipes -> Need MORE Trucks.
- Example: No Coffee -> Need MORE Tea (to stay awake).
- Logic: A and B are COMPETITORS.
- Verdict: **TRUE** (Flip to Negative).

Type 2: DEPENDENCY / CAUSATION (The "Fuel" Logic) - RETURNS FALSE
- "A is gone, so B cannot happen or is useless."
- Example: No Soil -> No Germination. (Adding "more germination" makes no sense without soil).
- Example: No Sunlight -> No Evaporation.
- Logic: A causes/enables B.
- Verdict: **FALSE** (Keep Positive).

Task: Is A a SUBSTITUTE for B? (Does removing A *increase* the need/occurrence of B?)

Output ONLY JSON:
{{
  "identified_goal": "...",
  "is_substitute": true/false,          // Only true when clearly substitution
  "confidence": 0.0-1.0,                 // Conservative: use >=0.7 only if very certain
  "reasoning": "..."
}}
""".strip()

        try:
            response = self._call_llm(prompt)
            response = self._clean_response(response)
            data = json.loads(response)
            is_sub = bool(data.get("is_substitute", False))
            try:
                conf = float(data.get("confidence", 0.0))
            except Exception:
                conf = 0.5
            conf = max(0.0, min(1.0, conf))

            goal = data.get("identified_goal", "Unknown")
            reasoning = str(data.get("reasoning", "")).strip()
            print(f"[Counterfactual] A='{cause}', B='{effect}' | Goal: {goal} | Is Substitute? {is_sub} | conf={conf:.2f}")
            return {
                "is_substitute": is_sub,
                "confidence": conf,
                "reasoning": reasoning,
            }
        except Exception as e:
            print(f"_check_counterfactual_substitution error: {e}")
            return {
                "is_substitute": False,
                "confidence": 0.5,
                "reasoning": "LLM parsing failure or exception.",
            }

    def _verify_chain_plausibility(
        self,
        start: str,
        target: str,
        path_summary: str,
        context: str,
    ) -> bool:
        """
        Path-Aware Sanity Check (broad-minded, context-aware).

        Improvements:
        - Gives benefit of the doubt for skipped/implicit steps.
        - Focuses on blocking semantic drift and absurd logic,
          rather than penalizing indirect but reasonable chains.
        """
        prompt = f"""
You are a Broad-minded Scientific Reviewer.
We are evaluating a causal chain based on commonsense scientific knowledge.

Question Context:
{context}

Start Event: "{start}"
End Event: "{target}"

Proposed Chain:
{path_summary}

Task:
Determine if this chain represents a LOGICALLY COHERENT argument
based on scientific knowledge, even if it simplifies complex processes
and skips obvious steps.

*** CRITERIA FOR APPROVAL (is_plausible = true) ***

1. Implicit Steps are OK:
   - If the chain skips obvious intermediate steps
     (e.g. "Fewer plants -> Less seeds germinating" skipping pollination/dispersal),
     it is still VALID. Do not be a pedant.

2. General Causality:
   - If, based on scientific knowledge, it is reasonable that A usually leads to B
     (e.g. "Rain -> Wet ground", "More fire -> More heat"), it is VALID.

3. Negative Logic:
   - Chains explaining prevention or reduction are valid,
     e.g. "No clouds -> No rain", "Less fuel -> Less energy".

*** CRITERIA FOR REJECTION (is_plausible = false) ***

1. Semantic Drift:
   - A key word changes meaning halfway through the chain,
     e.g. "plant shoots" (botany) -> "shooting guns" (weapons).

2. Magical/Absurd Links:
   - Connects completely unrelated concepts without a mechanism,
     e.g. "Pollen -> Internet speed".

3. Extreme Butterfly Effect:
   - A long chain that relies on many weak, speculative jumps across unrelated domains
     without scientific support.

Verification Question:
"Is there a reasonable scientific or commonsense explanation
where this chain holds true as a causal explanation?"

Output ONLY JSON:
{{"is_plausible": true/false, "reasoning": "brief explanation"}}
        """.strip()

        try:
            response = self._call_llm(prompt)
            response = self._clean_response(response)
            data = json.loads(response)
            is_plausible = bool(data.get("is_plausible", True))
            status = "✅ PASS" if is_plausible else "❌ REJECT"
            print(f"[Sanity Check] {status}: {path_summary}")
            if not is_plausible:
                print(f"  Reason: {data.get('reasoning', '')}")
            return is_plausible
        except Exception as e:
            # Fail-open: if the sanity judge fails, don't over-filter.
            print(f"_verify_chain_plausibility error: {e}")
            return True


    def bridge_close_hits(
        self,
        triples: List[Tuple[str, str, str]],
        close_hits: List[Dict[str, Any]],
        Y: str,
        max_bridge_nodes: int = 3,
    ) -> List[Tuple[str, str, str]]:
        if not close_hits: return triples
        Y = (Y or "").strip()
        if not Y: return triples

        new_triples = list(triples)
        def _get_ht(entry):
            if isinstance(entry, dict):
                return entry.get("head", ""), entry.get("tail", "")
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                return entry[0], entry[2]
            return "", ""
        # 按深度排序，优先桥接浅层的节点
        def _bridge_priority(hit: Dict[str, Any]) -> int:
            bfs_eq = str((hit or {}).get("bfs_equivalence") or "").strip().lower()
            qty_rel = str((hit or {}).get("quantity_relation") or "").strip().lower()
            # "exact_target" with same/subset quantity is the highest priority bridge class.
            if bfs_eq == "exact_target" and qty_rel in {"same_quantity", "subset_or_component"}:
                return 4
            if bfs_eq == "exact_target":
                return 3
            # Component/stage close-hits are also very valuable for bridging.
            if bfs_eq == "close_hit" and qty_rel == "subset_or_component":
                return 3
            if bfs_eq == "close_hit":
                return 2
            if bfs_eq == "bridge_candidate":
                return 1
            return 0

        sorted_hits = sorted(
            close_hits,
            key=lambda x: (
                -_bridge_priority(x),
                x.get("depth", 0),
                -float(x.get("weight", 0.0) or 0.0),
            ),
        )
        used = 0
        
        # Helper to parse bridge response
        def _parse_bridge_json(raw: str) -> Optional[Dict[str, Any]]:
            cleaned = self._clean_response(raw or "")
            try:
                return json.loads(cleaned)
            except Exception:
                match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if match:
                    try: return json.loads(match.group(0))
                    except: return None
            return None

        def _norm_bridge_relation(label: Any) -> str:
            s = str(label or "").strip().upper()
            if "INCREASE" in s:
                return "INCREASES"
            if "DECREASE" in s:
                return "DECREASES"
            if "NO_CLEAR_EFFECT" in s or ("NO" in s and "EFFECT" in s):
                return "NO_CLEAR_EFFECT"
            return ""

        def _sanitize_cf_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            meta = meta or {}
            is_valid = bool(meta.get("is_valid", True))
            try:
                conf = float(meta.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            reasoning = str(meta.get("reasoning", "")).strip()
            suggested = _norm_bridge_relation(meta.get("suggested_relation", "")) or "NO_SUGGESTION"
            if suggested not in {"INCREASES", "DECREASES"}:
                suggested = "NO_SUGGESTION"
            return {
                "is_valid": is_valid,
                "confidence": conf,
                "reasoning": reasoning,
                "suggested_relation": suggested,
            }

        def _verify_and_maybe_fix_bridge_edge(head: str, relation: str, tail: str) -> Tuple[Optional[str], Dict[str, Any]]:
            relation = _norm_bridge_relation(relation)
            if relation not in {"INCREASES", "DECREASES"}:
                return None, {"is_valid": True, "confidence": 0.5, "reasoning": "", "suggested_relation": "NO_SUGGESTION"}

            cf_meta = {}
            try:
                cf_meta = self._verify_edge_counterfactual(head, relation, tail) or {}
            except Exception:
                cf_meta = {}
            cf_meta = _sanitize_cf_meta(cf_meta)

            # Only intervene when the judge is confident the sign is wrong.
            if cf_meta.get("is_valid") is False and float(cf_meta.get("confidence", 0.0) or 0.0) >= 0.7:
                suggested = str(cf_meta.get("suggested_relation", "") or "").strip().upper()
                if suggested in {"INCREASES", "DECREASES"} and suggested != relation:
                    # Re-check the suggested direction before flipping.
                    cf2 = {}
                    try:
                        cf2 = self._verify_edge_counterfactual(head, suggested, tail) or {}
                    except Exception:
                        cf2 = {}
                    cf2 = _sanitize_cf_meta(cf2)
                    if cf2.get("is_valid") is True:
                        return suggested, cf2
                # No safe flip -> drop this bridge edge.
                return None, cf_meta

            return relation, cf_meta

        def _add_bridge_edge(
            head: str,
            relation: str,
            tail: str,
            extra: Optional[Dict[str, Any]] = None,
        ) -> Tuple[Optional[str], Dict[str, Any]]:
            nonlocal used
            final_relation, cf_meta = _verify_and_maybe_fix_bridge_edge(head, relation, tail)
            if not final_relation:
                return None, cf_meta
            edge = {
                "head": head,
                "relation": final_relation,
                "tail": tail,
                "is_substitute": False,
                "substitute_confidence": 0.0,
                # Save the bridge-time verification results (Top-K stage may override later).
                "cf_is_valid": bool(cf_meta.get("is_valid", True)),
                "cf_confidence": float(cf_meta.get("confidence", 0.5) or 0.5),
                "cf_reasoning": str(cf_meta.get("reasoning", "")).strip(),
                "cf_suggested_relation": str(cf_meta.get("suggested_relation", "NO_SUGGESTION")),
                "source": "bridge",
            }
            if extra:
                edge.update(extra)
            new_triples.append(edge)
            if hasattr(self, "bridge_edges"):
                self.bridge_edges.add((head, final_relation, tail))
            used += 1
            return final_relation, cf_meta

        for hit in sorted_hits:
            if used >= max_bridge_nodes: break
            node = (hit.get("node") or "").strip()
            if not node: continue
            
            # 防止重复添加
            duplicate_found = False
            for entry in new_triples:
                h, t = _get_ht(entry)
                if h == node and t == Y:
                    duplicate_found = True
                    break
            if duplicate_found:
                continue

            # === 如果是同一概念 (identical)，可以安全地认为正相关直接直连 ===
            bfs_eq = str(hit.get("bfs_equivalence") or "").strip().lower()

            # Lexical helpers for safe auto-bridging of exact_target / strong opposites.
            def _norm(s: str) -> str:
                s = (s or "").strip().lower()
                s = re.sub(r"[^a-z0-9\s]+", " ", s)
                s = re.sub(r"\s+", " ", s).strip()
                return s

            node_norm = _norm(node)
            y_norm = _norm(Y)
            node_toks = set(node_norm.split()) if node_norm else set()
            y_toks = set(y_norm.split()) if y_norm else set()
            union = node_toks | y_toks
            jacc = (len(node_toks & y_toks) / len(union)) if union else 0.0

            neg_markers = {
                "impairment", "impairments",
                "deformity", "deformities",
                "abnormality", "abnormalities",
                "defect", "defects",
                "damage", "damaged",
                "mutation", "mutations",
                "pollution", "contamination", "toxicity", "toxic",
                "reduction", "reductions", "decline",
                "decrease", "decreases", "loss", "lack", "absence", "missing",
            }
            neg_mismatch = bool(node_toks & neg_markers) ^ bool(y_toks & neg_markers)

            # 1) Opposite-like forms with strong overlap: auto negative bridge.
            if neg_mismatch and jacc >= 0.60:
                final_rel, _cf = _add_bridge_edge(
                    node,
                    "DECREASES",
                    Y,
                    extra={"bridge_reason": "opposite_like"},
                )
                if final_rel:
                    print(f"[Bridge] Auto-bridging OPPOSITE-like node: '{node}' -> {final_rel} -> '{Y}' (jacc={jacc:.2f})")
                else:
                    print(f"[Bridge] Dropped OPPOSITE-like bridge: '{node}' -> DECREASES -> '{Y}' (jacc={jacc:.2f})")
                continue

            # 2) Exact-target synonyms / near-identical forms: safe positive bridge.
            if bfs_eq == "exact_target":
                measurement_tokens = {
                    "rate", "rates",
                    "level", "levels",
                    "probability", "likelihood", "frequency",
                    "occurrence", "success", "chance",
                }
                extra_node = node_toks - y_toks
                extra_y = y_toks - node_toks
                measure_diff = (
                    (extra_node and extra_node.issubset(measurement_tokens) and len(extra_node) <= 2 and len(extra_y) == 0)
                    or (extra_y and extra_y.issubset(measurement_tokens) and len(extra_y) <= 2 and len(extra_node) == 0)
                )
                if node_norm == y_norm or jacc >= 0.92 or jacc >= 0.75 or measure_diff:
                    final_rel, _cf = _add_bridge_edge(
                        node,
                        "INCREASES",
                        Y,
                        extra={"bridge_reason": "exact_target_lexical"},
                    )
                    if final_rel:
                        print(f"[Bridge] Auto-bridging EXACT_TARGET node: '{node}' -> {final_rel} -> '{Y}' (jacc={jacc:.2f})")
                    else:
                        print(f"[Bridge] Dropped EXACT_TARGET bridge: '{node}' -> INCREASES -> '{Y}' (jacc={jacc:.2f})")
                    continue
                # Even if surface forms differ, exact_target equivalence is a strong signal: bridge then verify.
                final_rel, _cf = _add_bridge_edge(
                    node,
                    "INCREASES",
                    Y,
                    extra={"bridge_reason": "exact_target_equivalence"},
                )
                if final_rel:
                    print(f"[Bridge] Bridging EXACT_TARGET by equivalence: '{node}' -> {final_rel} -> '{Y}'")
                else:
                    print(f"[Bridge] Dropped EXACT_TARGET by equivalence: '{node}' -> INCREASES -> '{Y}'")
                continue

            # 3) For close_hit / bridge_candidate, ask a minimal direction judge and bridge to base.
            if bfs_eq in {"close_hit", "bridge_candidate"}:
                effect_prompt = f"""
You are a causal direction judge.

We interpret every variable as a QUANTITY / LEVEL / FREQUENCY / PROBABILITY.
- If a variable is a bare noun (e.g., "frogs"), treat it as "number/abundance/probability of frogs (being present/surviving)".
- health / quality / suitability / stability / strength / fitness / habitat quality are POSITIVE improvements.
- toxicity / pollution / contamination / dangerous chemicals are NEGATIVE harms.

Monotonic priors (usually true):
- If X is health/quality/suitability/habitat quality of an organism/population, INCREASE in X usually INCREASES the population/number/probability.
- If X is toxicity/pollution/contamination/dangerous chemicals affecting an organism/population, INCREASE in X usually DECREASES the population/number/probability.

Base variable (Y): "{Y}"
Candidate node (X): "{node}"

Task: If X increases, what happens to Y?

Output ONLY JSON:
{{
  "edge_direction": "INCREASES" | "DECREASES" | "NO_CLEAR_EFFECT",
  "prior_consistency": true | false,
  "reasoning": "brief explanation"
}}
""".strip()
                try:
                    raw_dir = self._call_llm(effect_prompt)
                except Exception:
                    raw_dir = ""
                direction = ""
                prior_consistency = True
                dir_reason = ""
                dir_data = _parse_bridge_json(raw_dir or "")
                if isinstance(dir_data, dict) and dir_data:
                    direction = _norm_bridge_relation(dir_data.get("edge_direction") or dir_data.get("direction") or "")
                    try:
                        pc = dir_data.get("prior_consistency", True)
                        if isinstance(pc, str):
                            prior_consistency = pc.strip().lower() == "true"
                        else:
                            prior_consistency = bool(pc)
                    except Exception:
                        prior_consistency = True
                    dir_reason = str(dir_data.get("reasoning", "")).strip()
                else:
                    dir_clean = self._clean_response(raw_dir or "").upper()
                    m = re.search(r"\b(INCREASES|DECREASES|NO_CLEAR_EFFECT)\b", dir_clean)
                    if m:
                        direction = m.group(1)
                    else:
                        if "INCREASE" in dir_clean:
                            direction = "INCREASES"
                        elif "DECREASE" in dir_clean:
                            direction = "DECREASES"
                        elif "NO" in dir_clean and "EFFECT" in dir_clean:
                            direction = "NO_CLEAR_EFFECT"

                if direction in {"INCREASES", "DECREASES"}:
                    final_rel, _cf = _add_bridge_edge(
                        node,
                        direction,
                        Y,
                        extra={
                            "prior_consistency": bool(prior_consistency),
                            "direction_reasoning": dir_reason,
                            "bridge_reason": f"direction_judge_{bfs_eq}",
                        },
                    )
                    if final_rel:
                        print(f"[Bridge] Bridging {bfs_eq.upper()} node via direction-judge: '{node}' -> {final_rel} -> '{Y}'")
                    else:
                        print(f"[Bridge] Dropped {bfs_eq.upper()} bridge via direction-judge: '{node}' -> {direction} -> '{Y}'")
                    continue

            # Legacy fall-through (kept for safety).
            relation_type = (hit.get("relation_type") or "weakly_related").strip().lower()
            if relation_type == "identical":
                # Guard: only auto-bridge when strings are truly very close.
                # This prevents false "identical" like "precipitation variability" vs "precipitation".
                def _norm(s: str) -> str:
                    s = (s or "").strip().lower()
                    s = re.sub(r"[^a-z0-9\s]+", " ", s)
                    s = re.sub(r"\s+", " ", s).strip()
                    return s

                node_norm = _norm(node)
                y_norm = _norm(Y)
                node_toks = set(node_norm.split()) if node_norm else set()
                y_toks = set(y_norm.split()) if y_norm else set()
                union = node_toks | y_toks
                jacc = (len(node_toks & y_toks) / len(union)) if union else 0.0

                # If not close enough, fall through to the LLM bridge classifier.
                # Stricter auto-bridge to reduce false positives.
                if not (node_norm == y_norm or jacc >= 0.92):
                    relation_type = "weakly_related"
                else:
                    # 同一事物，意味着正相关 (A变多 等于 B变多)
                    final_rel, _cf = _add_bridge_edge(
                        node,
                        "INCREASES",
                        Y,
                        extra={"bridge_reason": "identical_lexical"},
                    )
                    if final_rel:
                        print(f"[Bridge] Auto-bridging IDENTICAL node: '{node}' -> {final_rel} -> '{Y}'")
                    else:
                        print(f"[Bridge] Dropped IDENTICAL bridge: '{node}' -> INCREASES -> '{Y}'")
                    continue
            if relation_type == "opposite":
                # Opposite concept: only auto-bridge when overlap is strong (reduce false positives).
                def _norm(s: str) -> str:
                    s = (s or "").strip().lower()
                    s = re.sub(r"[^a-z0-9\s]+", " ", s)
                    s = re.sub(r"\s+", " ", s).strip()
                    return s
                node_norm = _norm(node)
                y_norm = _norm(Y)
                node_toks = set(node_norm.split()) if node_norm else set()
                y_toks = set(y_norm.split()) if y_norm else set()
                union = node_toks | y_toks
                jacc = (len(node_toks & y_toks) / len(union)) if union else 0.0
                if jacc < 0.60:
                    # Not confident enough: let the LLM bridge classifier decide instead.
                    relation_type = "weakly_related"
                else:
                    # Opposite concept: more of C implies less of Y (negative bridge).
                    final_rel, _cf = _add_bridge_edge(
                        node,
                        "DECREASES",
                        Y,
                        extra={"bridge_reason": "opposite_lexical"},
                    )
                    if final_rel:
                        print(f"[Bridge] Auto-bridging OPPOSITE node: '{node}' -> {final_rel} -> '{Y}'")
                    else:
                        print(f"[Bridge] Dropped OPPOSITE bridge: '{node}' -> DECREASES -> '{Y}'")
                    continue
            if relation_type == "part_of":
                # Smart allow-list: part_of can be safely bridged ONLY when the surface forms are
                # almost identical (e.g., adding just "rate(s)"/"probability"/"likelihood").
                def _norm(s: str) -> str:
                    s = (s or "").strip().lower()
                    s = re.sub(r"[^a-z0-9\s]+", " ", s)
                    s = re.sub(r"\s+", " ", s).strip()
                    return s
                node_norm = _norm(node)
                y_norm = _norm(Y)
                node_toks = set(node_norm.split()) if node_norm else set()
                y_toks = set(y_norm.split()) if y_norm else set()
                union = node_toks | y_toks
                jacc = (len(node_toks & y_toks) / len(union)) if union else 0.0

                # Do NOT auto-bridge if a negative/impairment marker mismatch exists.
                neg_markers = {
                    "impairment", "impairments",
                    "deformity", "deformities",
                    "abnormality", "abnormalities",
                    "defect", "defects",
                    "damage", "damaged",
                    "mutation", "mutations",
                    "pollution", "contamination", "toxicity", "toxic",
                    "reduction", "reductions", "decline",
                }
                if (node_toks & neg_markers) ^ (y_toks & neg_markers):
                    relation_type = "weakly_related"
                else:
                    # Only allow small measurement-only differences.
                    measurement_tokens = {
                        "rate", "rates",
                        "level", "levels",
                        "probability", "likelihood", "frequency",
                        "occurrence", "success", "chance",
                    }
                    extra_node = node_toks - y_toks
                    extra_y = y_toks - node_toks
                    measure_diff = (
                        (extra_node and extra_node.issubset(measurement_tokens) and len(extra_node) <= 2 and len(extra_y) == 0)
                        or (extra_y and extra_y.issubset(measurement_tokens) and len(extra_y) <= 2 and len(extra_node) == 0)
                    )
                    # Also allow high overlap (e.g., only one extra token).
                    if jacc >= 0.75 or measure_diff:
                        final_rel, _cf = _add_bridge_edge(
                            node,
                            "INCREASES",
                            Y,
                            extra={"bridge_reason": "part_of_lexical"},
                        )
                        if final_rel:
                            print(f"[Bridge] Auto-bridging PART_OF (lexical) node: '{node}' -> {final_rel} -> '{Y}' (jacc={jacc:.2f})")
                        else:
                            print(f"[Bridge] Dropped PART_OF (lexical) bridge: '{node}' -> INCREASES -> '{Y}' (jacc={jacc:.2f})")
                        continue
                    # Not safe enough: fall through to the LLM bridge classifier.
                    relation_type = "weakly_related"
            # part_of 不能默认 INCREASES：例如 stability/strength/integrity 往往与裂缝发生是反向关系
            # 因此交给下面的桥接分类器判方向（INCREASES / DECREASES），避免系统性符号写反
            # =================================================

            prompt = f"""
You are a causal reasoning assistant.
Problem: {self.question}
C = "{node}", Y = "{Y}"
Classify ONLY the direct relationship between C and Y.
Output STRICT JSON:
{{
  "causal_type": "direct_step" | "shared_cause" | "multi_step" | "correlation_only",
  "direction": "C_increases_Y" | "C_decreases_Y" | "no_direct_effect",
  "is_local_to_question": true | false,
  "reasoning": "..."
}}
""".strip()
            try:
                raw = self._call_llm(prompt)
            except Exception: continue

            data = _parse_bridge_json(raw or "")
            if not data: continue

            causal_type = str(data.get("causal_type", "")).strip().lower()
            direction = str(data.get("direction", "")).strip().lower()
            is_local = str(data.get("is_local_to_question", "false")).lower() == "true"

            if direction not in {"c_increases_y", "c_decreases_y"}:
                continue
            # Stricter bridge: require a local direct step for all relation types.
            if causal_type != "direct_step" or not is_local:
                continue

            final_relation = "INCREASES" if direction == "c_increases_y" else "DECREASES"
            final_rel, _cf = _add_bridge_edge(
                node,
                final_relation,
                Y,
                extra={
                    "bridge_reason": f"llm_bridge_{relation_type}",
                    "bridge_causal_type": causal_type,
                    "bridge_local": bool(is_local),
                },
            )
            if final_rel:
                print(f"[Bridge] LLM-bridging {relation_type}: '{node}' -> {final_rel} -> '{Y}' (type={causal_type}, local={is_local})")
            else:
                print(f"[Bridge] Dropped LLM-bridge {relation_type}: '{node}' -> {final_relation} -> '{Y}' (type={causal_type}, local={is_local})")

        return new_triples


    def classify_variable_relation(self, a: str, b: str, context: str = "") -> VariableRelation:
        """
        Core comparison function returning a rich VariableRelation label.
        """
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return VariableRelation("weakly_related", "Empty input.")

        # --- Lightweight lexical guardrails ---
        # LLMs (esp. small ones) can over-label "identical" when two phrases share a head noun
        # but differ in a critical modifier (e.g., "precipitation variability" vs "precipitation").
        # We keep the LLM decision, but post-process "identical" to avoid systematic false positives.
        def _norm_tokens(s: str) -> Set[str]:
            s = (s or "").lower()
            s = re.sub(r"[^a-z0-9\s]+", " ", s)
            toks = [t for t in s.split() if t]
            stop = {
                "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "with",
                "between", "from", "at", "by", "as", "into", "over", "under",
                # very generic fillers that often appear in these phrases
                "levels", "level",
            }
            return set(t for t in toks if t not in stop)

        # Modifiers that usually indicate a different measurement/attribute, not the same variable.
        # If only one side contains them, we should NOT treat the phrases as identical.
        _identity_block_mods = {
            "variability", "intensity", "probability", "likelihood", "frequency",
            "concentration", "density", "distribution", "rate", "rates", "index", "indices",
            "margin", "margins", "potential", "risk", "chance",
        }
        # Negative/impairment markers: if one side has these and the other doesn't,
        # it often means "opposite" in WIQA-style variable semantics.
        _opposite_markers = {
            "impairment", "impairments",
            "deformity", "deformities",
            "abnormality", "abnormalities",
            "defect", "defects",
            "mutation", "mutations",
            "reduction", "reductions",
            "decline", "decrease", "decreases", "decreased",
            "loss", "lack", "absence", "missing",
            "failure", "fail", "failed",
            "damage", "damaged", "injury", "injuries",
            "inhibition", "inhibits", "inhibited",
            "suppression", "suppressed",
            "toxicity", "toxic", "pollution", "contamination",
        }

        # === Rule-first (deterministic) fast path ===
        # Only take rule decisions when confidence is high; otherwise fall back to LLM.
        def _norm_text(s: str) -> str:
            s = (s or "").strip().lower()
            s = re.sub(r"[^a-z0-9\s]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        a_norm = _norm_text(a)
        b_norm = _norm_text(b)
        ta0 = _norm_tokens(a)
        tb0 = _norm_tokens(b)
        union0 = ta0 | tb0
        jacc0 = (len(ta0 & tb0) / len(union0)) if union0 else 0.0

        # 1) Exact normalized match => identical
        if a_norm and a_norm == b_norm:
            return VariableRelation("identical", "[Rule] exact normalized match.")

        # 2) Strong overlap + no critical modifier mismatch => identical
        mods_a0 = ta0 & _identity_block_mods
        mods_b0 = tb0 & _identity_block_mods
        if jacc0 >= 0.90 and not (mods_a0 ^ mods_b0):
            return VariableRelation("identical", f"[Rule] high token overlap (jacc={jacc0:.2f}).")

        # 3) Strong overlap + one side has negative markers => opposite
        neg_a0 = ta0 & _opposite_markers
        neg_b0 = tb0 & _opposite_markers
        if jacc0 >= 0.45 and len(ta0 & tb0) >= 2 and (neg_a0 ^ neg_b0):
            return VariableRelation("opposite", f"[Rule] negative-marker mismatch with overlap (jacc={jacc0:.2f}).")

        # 4) Completely disjoint tokens => likely unrelated (rule-level)
        # Keep this conservative: only trigger when both sides have enough content.
        if len(ta0) >= 2 and len(tb0) >= 2 and len(ta0 & tb0) == 0:
            return VariableRelation("unrelated", "[Rule] no token overlap.")

        prompt = f"""
You are a scientific concept-matching assistant.

Task:
Given two short phrases A and B that refer to variables/events in a causal question,
classify their semantic relationship into EXACTLY ONE of:

- "identical": The two phrases describe the SAME underlying quantity/state/event
               (minor rephrasing is allowed).
- "part_of":   One is a sub-event/component of the other in the mechanism being discussed.
- "causal_but_separate": They are causally related but NOT the same quantity.
- "weakly_related": Vaguely or loosely related, but not clearly part-of or strongly causal.
- "unrelated": No meaningful relation.
- "opposite": Explicitly opposing meanings/states.

Guidance:
- If you are NOT clearly confident they are the same underlying quantity, pick a weaker label
  such as "causal_but_separate" or "weakly_related" instead of "identical" or "part_of".
- Do not answer the WIQA question itself; only classify the relation.

Question context (may be empty):
{context}

A = "{a}"
B = "{b}"

Output ONLY JSON:
{{"relation_type": "<one of the six labels>", "reasoning": "short explanation"}}
""".strip()

        allowed = {
            "identical",
            "part_of",
            "causal_but_separate",
            "weakly_related",
            "unrelated",
            "opposite",
        }

        try:
            resp = self._call_llm(prompt)
            resp = self._clean_response(resp)
            try:
                data = json.loads(resp)
            except Exception:
                match = re.search(r"\{.*\}", resp, flags=re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                else:
                    raise
            relation_type = str(data.get("relation_type", "")).strip().lower()
            reasoning = str(data.get("reasoning", "")).strip()
            if relation_type not in allowed:
                relation_type = "weakly_related"
            # Post-process overly broad "identical" labels using token overlap + modifier checks.
            if relation_type == "identical":
                ta = _norm_tokens(a)
                tb = _norm_tokens(b)
                # If there is a critical modifier on only one side, it's not identical.
                mods_a = ta & _identity_block_mods
                mods_b = tb & _identity_block_mods
                if mods_a ^ mods_b:
                    relation_type = "causal_but_separate"
                    if reasoning:
                        reasoning = f"[PostProcess] downgraded identical due to modifier mismatch {sorted(list(mods_a ^ mods_b))}. {reasoning}"
                    else:
                        reasoning = f"[PostProcess] downgraded identical due to modifier mismatch {sorted(list(mods_a ^ mods_b))}."
                else:
                    union = ta | tb
                    jacc = (len(ta & tb) / len(union)) if union else 0.0
                    # Require reasonably high lexical overlap for "identical".
                    if jacc < 0.60:
                        relation_type = "causal_but_separate" if jacc >= 0.25 else "weakly_related"
                        if reasoning:
                            reasoning = f"[PostProcess] downgraded identical (token Jaccard={jacc:.2f}). {reasoning}"
                        else:
                            reasoning = f"[PostProcess] downgraded identical (token Jaccard={jacc:.2f})."

            # Post-process: recover from "unrelated"/"weakly_related" when the phrases clearly overlap.
            # Also detect "opposite" when one side is an impairment/negative form of the other.
            ta2 = _norm_tokens(a)
            tb2 = _norm_tokens(b)
            overlap = ta2 & tb2
            union2 = ta2 | tb2
            jacc2 = (len(overlap) / len(union2)) if union2 else 0.0
            if len(overlap) >= 2 and jacc2 >= 0.45:
                neg_a = ta2 & _opposite_markers
                neg_b = tb2 & _opposite_markers
                if neg_a ^ neg_b:
                    # e.g., "tadpole limb development impairment" vs "tadpole limb development"
                    if relation_type in {"unrelated", "weakly_related", "causal_but_separate", "part_of", "identical"}:
                        relation_type = "opposite"
                        if reasoning:
                            reasoning = f"[PostProcess] upgraded to opposite (overlap={sorted(list(overlap))}, jacc={jacc2:.2f}). {reasoning}"
                        else:
                            reasoning = f"[PostProcess] upgraded to opposite (overlap={sorted(list(overlap))}, jacc={jacc2:.2f})."
                else:
                    if relation_type in {"unrelated", "weakly_related"}:
                        relation_type = "part_of" if jacc2 >= 0.75 else "causal_but_separate"
                        if reasoning:
                            reasoning = f"[PostProcess] upgraded relation (overlap={sorted(list(overlap))}, jacc={jacc2:.2f}). {reasoning}"
                        else:
                            reasoning = f"[PostProcess] upgraded relation (overlap={sorted(list(overlap))}, jacc={jacc2:.2f})."

            # Guardrail: avoid over-using "part_of"/"identical" when one side is clearly a POPULATION-style quantity.
            # Example: "frogs population" is NOT a component of "frogs mating"; it's upstream (causal_but_separate).
            population_markers = {"population", "density", "abundance", "availability", "scarcity"}
            if relation_type in {"part_of", "identical"}:
                has_pop_a = bool(ta2 & population_markers)
                has_pop_b = bool(tb2 & population_markers)
                if has_pop_a ^ has_pop_b:
                    relation_type = "causal_but_separate"
                    if reasoning:
                        reasoning = f"[PostProcess] downgraded {data.get('relation_type','')} due to population-marker mismatch. {reasoning}"
                    else:
                        reasoning = "[PostProcess] downgraded due to population-marker mismatch."
            relation = VariableRelation(relation_type, reasoning)
            print("is_same_variable: LLM 判定", a, "<->", b, "=", relation_type)
            return relation
        except Exception as e:
            print("is_same_variable: 解析 LLM 响应失败:", e)
            return VariableRelation("weakly_related", "Parse failure or exception.")

    def _derive_bfs_equivalence(
        self,
        core_entity_relation: str,
        quantity_relation: str,
        causal_or_structural_relation: str,
    ) -> str:
        """Derive bfs_equivalence from the 3 dimensions using WIQA-wide rules."""
        cer = (core_entity_relation or "").strip().lower()
        qr = (quantity_relation or "").strip().lower()
        csr = (causal_or_structural_relation or "").strip().lower()

        if (
            cer == "same_entity"
            and qr in {"same_quantity", "subset_or_component"}
            and csr == "same_state"
        ):
            return "exact_target"
        if (
            cer == "same_entity"
            and qr in {"subset_or_component", "aggregate_or_population"}
            and csr == "direct_cause_or_effect"
        ):
            return "close_hit"
        if (
            cer in {"same_entity", "overlapping_entities"}
            and csr in {"direct_cause_or_effect", "correlated_or_confounding"}
            and qr == "different_quantity"
        ):
            return "bridge_candidate"
        if cer == "different_entity" and csr == "independent_or_unknown":
            return "not_related"

        # Conservative fallback: avoid "not_related" unless clearly different entity.
        if cer == "different_entity":
            return "bridge_candidate" if csr != "independent_or_unknown" else "not_related"
        if cer == "same_entity":
            return "close_hit"
        return "bridge_candidate"

    def is_same_variable(self, a: str, b: str, context: str = "") -> Dict[str, Any]:
        """
        Multi-dimension variable comparison.
        Returns a dict with:
          core_entity_relation, quantity_relation, causal_or_structural_relation,
          bfs_equivalence, explanation
        """
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return {
                "core_entity_relation": "different_entity",
                "quantity_relation": "different_quantity",
                "causal_or_structural_relation": "independent_or_unknown",
                "bfs_equivalence": "not_related",
                "explanation": "Empty input.",
            }

        # --- Lightweight lexical fast paths ---
        def _norm_text(s: str) -> str:
            s = (s or "").strip().lower()
            s = re.sub(r"[^a-z0-9\s]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _norm_tokens(s: str) -> Set[str]:
            s = (s or "").lower()
            s = re.sub(r"[^a-z0-9\s]+", " ", s)
            toks = [t for t in s.split() if t]
            stop = {
                "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "with",
                "between", "from", "at", "by", "as", "into", "over", "under",
                "levels", "level",
            }
            return set(t for t in toks if t not in stop)

        neg_markers = {
            "impairment", "impairments",
            "deformity", "deformities",
            "abnormality", "abnormalities",
            "defect", "defects",
            "mutation", "mutations",
            "reduction", "reductions",
            "decline", "decrease", "decreases", "decreased",
            "loss", "lack", "absence", "missing",
            "failure", "fail", "failed",
            "damage", "damaged", "injury", "injuries",
            "inhibition", "inhibits", "inhibited",
            "suppression", "suppressed",
            "toxicity", "toxic", "pollution", "contamination",
        }

        a_norm = _norm_text(a)
        b_norm = _norm_text(b)
        if a_norm and a_norm == b_norm:
            return {
                "core_entity_relation": "same_entity",
                "quantity_relation": "same_quantity",
                "causal_or_structural_relation": "same_state",
                "bfs_equivalence": "exact_target",
                "explanation": "Exact normalized match.",
            }

        ta = _norm_tokens(a)
        tb = _norm_tokens(b)
        overlap = ta & tb
        union = ta | tb
        jacc = (len(overlap) / len(union)) if union else 0.0
        neg_mismatch = bool(ta & neg_markers) ^ bool(tb & neg_markers)
        if jacc >= 0.90 and not neg_mismatch:
            return {
                "core_entity_relation": "same_entity",
                "quantity_relation": "same_quantity",
                "causal_or_structural_relation": "same_state",
                "bfs_equivalence": "exact_target",
                "explanation": f"High lexical overlap (jacc={jacc:.2f}).",
            }

        prompt = f"""
You are judging the relationship between two variables in a causal graph.

Variable A: "{a}"
Variable B: "{b}"

Step 1: Extract for EACH variable:
- main entity or entities (who/what is this about?)
- what is being measured or described (quantity, property, process, event?)
- is it individual-level (per animal/person) or population-level (total count, density)?
- does it describe a state itself, or a cause/effect of some other state?

Step 2: Decide these fields:

1) core_entity_relation:
- "same_entity": if both mainly talk about the SAME type of entity (e.g., both about tadpoles, both about frogs).
- "overlapping_entities": if one talks about a group that contains the other (e.g., frogs vs amphibians).
- "different_entity": if they are about clearly different entities (e.g., human movement vs tadpoles).

2) quantity_relation:
- "same_quantity": if they describe the SAME underlying variable using different wording or granularity.
- "subset_or_component": if one is a PART, STAGE, or COMPONENT of the other.
- "aggregate_or_population": if one is an INDIVIDUAL-LEVEL quantity and the other is a POPULATION/AGGREGATE quantity for the same entity.
- "different_quantity": if they measure completely different aspects.

3) causal_or_structural_relation:
- "same_state": if they basically refer to the same state or its negation.
- "direct_cause_or_effect": if changes in one directly cause changes in the other or are a necessary condition.
- "correlated_or_confounding": if they tend to change together but are neither identical nor a clean cause–effect.
- "independent_or_unknown": if there is no clear structural or causal link.

4) bfs_equivalence:
Decide how B should be treated when we are searching for A as a target in a causal graph.

Rules:
- Use "exact_target" if:
  - core_entity_relation = "same_entity", AND
  - quantity_relation = "same_quantity" OR "subset_or_component", AND
  - causal_or_structural_relation = "same_state".
- Use "close_hit" if:
  - core_entity_relation = "same_entity", AND
  - quantity_relation ∈ {{"subset_or_component", "aggregate_or_population"}}, AND
  - causal_or_structural_relation = "direct_cause_or_effect".
- Use "bridge_candidate" if:
  - core_entity_relation ∈ {{"same_entity", "overlapping_entities"}}, AND
  - causal_or_structural_relation ∈ {{"direct_cause_or_effect", "correlated_or_confounding"}},
  but A and B are clearly different quantities.
- Use "not_related" ONLY if:
  - core_entity_relation = "different_entity", AND
  - causal_or_structural_relation = "independent_or_unknown".

Very important:
- If the entities are the same, DO NOT use "not_related" unless they are truly independent.

Question context (may be empty):
{context}

 Important:
 - Do NOT output bfs_equivalence. It will be derived deterministically in code from the 3 fields above.
 
 Output ONLY JSON in this format:
 {{
   "core_entity_relation": "...",
   "quantity_relation": "...",
   "causal_or_structural_relation": "...",
   "explanation": "short explanation"
 }}
""".strip()

        allowed_core = {"same_entity", "overlapping_entities", "different_entity"}
        allowed_qty = {"same_quantity", "subset_or_component", "aggregate_or_population", "different_quantity"}
        allowed_csr = {"same_state", "direct_cause_or_effect", "correlated_or_confounding", "independent_or_unknown"}

        def _norm_enum(v: Any, allowed: Set[str]) -> str:
            s = str(v or "").strip().lower()
            s = s.replace(" ", "_").replace("-", "_")
            aliases = {
                "same": "same_entity",
                "overlapping": "overlapping_entities",
                "overlap": "overlapping_entities",
                "different": "different_entity",
                "part_of": "subset_or_component",
                "subset": "subset_or_component",
                "component": "subset_or_component",
                "population": "aggregate_or_population",
                "aggregate": "aggregate_or_population",
                "correlated": "correlated_or_confounding",
                "confounding": "correlated_or_confounding",
                "independent": "independent_or_unknown",
                "unknown": "independent_or_unknown",
                "exact": "exact_target",
                "target": "exact_target",
                "close": "close_hit",
                "bridge": "bridge_candidate",
                "unrelated": "not_related",
            }
            s = aliases.get(s, s)
            return s if s in allowed else ""

        data: Dict[str, Any] = {}
        try:
            raw = self._call_llm(prompt)
            cleaned = self._clean_response(raw or "")
            try:
                data = json.loads(cleaned)
            except Exception:
                match = re.search(r"\{[\s\S]*?\}", cleaned)
                if match:
                    data = json.loads(match.group(0))
        except Exception:
            data = {}

        core_entity_relation = _norm_enum(data.get("core_entity_relation"), allowed_core) or "different_entity"
        quantity_relation = _norm_enum(data.get("quantity_relation"), allowed_qty) or "different_quantity"
        causal_or_structural_relation = _norm_enum(data.get("causal_or_structural_relation"), allowed_csr) or "independent_or_unknown"

        # Do NOT trust any bfs_equivalence label from the LLM.
        # Always derive it from the 3 dimensions to avoid inconsistent outputs
        # (e.g., same_entity + correlated -> not_related), which breaks BFS pruning.
        try:
            generic_noise = {
                "rate", "rates", "level", "levels", "amount", "number", "numbers",
                "probability", "likelihood", "chance", "risk", "potential",
                "frequency", "occurrence", "formation",
                "population", "density", "abundance", "size",
                "health", "quality", "suitability", "stability", "integrity", "availability", "scarcity",
            }
            entity_overlap = (ta & tb) - generic_noise
            if core_entity_relation == "different_entity" and entity_overlap:
                core_entity_relation = "same_entity"
        except Exception:
            pass

        bfs_equivalence = self._derive_bfs_equivalence(
            core_entity_relation, quantity_relation, causal_or_structural_relation
        )
        explanation = str(data.get("explanation") or data.get("reasoning") or "").strip()
        if not explanation:
            explanation = f"Derived bfs_equivalence={bfs_equivalence}."

        rel = {
            "core_entity_relation": core_entity_relation,
            "quantity_relation": quantity_relation,
            "causal_or_structural_relation": causal_or_structural_relation,
            "bfs_equivalence": bfs_equivalence,
            "explanation": explanation,
        }
        try:
            print(
                "is_same_variable_v2:",
                a,
                "<->",
                b,
                "=",
                bfs_equivalence,
                f"({core_entity_relation}, {quantity_relation}, {causal_or_structural_relation})",
            )
        except Exception:
            pass
        return rel

    def get_causal_chain(
        self,
        triples: List[Tuple[str, str, str]],
        start_X: str,
        target_Y: str,
        max_path_length: int = 5  # 限制最大长度为 5
    ) -> Dict[str, Any]:
        """
        [Modified]
        Extract chains with SCORING to prioritize semantic relevance and reduce noise.
        Implements:
        1. Generic Node Penalty (Punish 'Size', 'Stability', etc.)
        2. Top-K Selection (Keep only best 10 paths)
        3. Deterministic Sorting
        """
        start = (start_X or "").strip()
        target = (target_Y or "").strip()
        if not start or not target:
            return {"start": start, "target": target, "paths": [], "num_paths": 0, "shortest_path_length": None, "all_nodes_in_paths": set()}

        # 1. 构建图
        graph: Dict[str, List[Dict[str, str]]] = {}
        for triple in triples:
            if isinstance(triple, dict):
                h = triple.get("head")
                r = triple.get("relation")
                t = triple.get("tail")
                is_sub = bool(triple.get("is_substitute", False))
                sub_conf = float(triple.get("substitute_confidence", 0.0) or 0.0)
                cf_is_valid = bool(triple.get("cf_is_valid", True))
                try:
                    cf_conf = float(triple.get("cf_confidence", 0.5))
                except Exception:
                    cf_conf = 0.5
                cf_conf = max(0.0, min(1.0, cf_conf))
                cf_reason = str(triple.get("cf_reasoning", "") or "").strip()
                src = triple.get("source", "bfs")
            else:
                try:
                    h, r, t = triple
                except Exception:
                    continue
                is_sub = False
                sub_conf = 0.0
                cf_is_valid = True
                cf_conf = 0.5
                cf_reason = ""
                src = "bfs"
            edge = {
                "head": h,
                "relation": r,
                "tail": t,
                "is_substitute": is_sub,
                "substitute_confidence": sub_conf,
                "cf_is_valid": cf_is_valid,
                "cf_confidence": cf_conf,
                "cf_reasoning": cf_reason,
                "source": src,
            }
            if getattr(self, "bridge_edges", None) and (h, r, t) in self.bridge_edges:
                edge["source"] = "bridge"
            graph.setdefault(h, []).append(edge)
        if start not in graph:
            return {"start": start, "target": target, "paths": [], "num_paths": 0, "shortest_path_length": None, "all_nodes_in_paths": set()}

        # 2. 邻接表排序 (保证遍历顺序确定)
        for node in graph:
            graph[node].sort(key=lambda e: e["tail"])

        # === 定义“万金油”节点黑名单 ===
        # 这些词太过通用，如果路径包含它们，我们会降低该路径的权重
        generic_nodes = {"stability", "size", "growth rate", "water availability", "temperature", "pressure", "maturity"}

        # Optional drift metadata from BFS (if provided by expand_toward_target)
        rel_map = {}
        if isinstance(triples, dict):
            # (defensive; normally triples is a list)
            rel_map = (triples.get("node_rel_to_target") or {}) if isinstance(triples.get("node_rel_to_target"), dict) else {}
        else:
            rel_map = getattr(self, "last_node_rel_to_target", {}) or {}

        all_found_paths = []
        
        # 3. 带评分的 DFS
        def dfs(node: str, path_edges: List[Dict[str, str]], current_score: float):
            # 剪枝：超过长度直接丢弃
            if len(path_edges) >= max_path_length:
                return

            # 找到目标
            if node.lower() == target.lower() and path_edges:
                # === 计算路径分数 (分数越低越好) ===
                # 基础分 = 路径长度 (越短越好)
                score = len(path_edges)
                num_sub_edges = 0
                max_sub_conf = 0.0
                penalty = 0.0

                # Semantic drift counting: how many intermediate nodes were "unrelated" to the target?
                nodes_in_path = set()
                for ed in path_edges:
                    nodes_in_path.add(str(ed.get("head", "")).strip().lower())
                    nodes_in_path.add(str(ed.get("tail", "")).strip().lower())
                nodes_in_path.discard(start.lower())
                nodes_in_path.discard(target.lower())
                num_unrelated = 0
                num_weak = 0
                for n in nodes_in_path:
                    rel_t = str(rel_map.get(n, "")).strip().lower()
                    if rel_t in {"not_related", "unrelated"}:
                        num_unrelated += 1
                    elif rel_t == "bridge_candidate":
                        num_weak += 1
                
                # 惩罚项：每经过一个通用节点，分数 +0.5 (变差)
                for edge in path_edges:
                    h_lower = edge["head"].lower()
                    t_lower = edge["tail"].lower()
                    
                    # 检查由通用节点组成的边
                    if any(g in h_lower for g in generic_nodes):
                        score += 0.5
                    if any(g in t_lower for g in generic_nodes):
                        score += 0.5
                    # 反事实 Judge 软惩罚：如果某条边被判定为“符号/逻辑可疑”，降低该路径分数
                    # （不做硬过滤，避免因 Judge 偏差导致整题无路径）
                    if edge.get("cf_is_valid") is False:
                        try:
                            _cf_conf = float(edge.get("cf_confidence", 0.5))
                        except Exception:
                            _cf_conf = 0.5
                        _cf_conf = max(0.0, min(1.0, _cf_conf))
                        score += 0.6 * _cf_conf
                    # If the bridge direction contradicts monotonic priors, apply a soft penalty.
                    if edge.get("prior_consistency") is False:
                        score += 0.35
                    if edge.get("is_substitute"):
                        num_sub_edges += 1
                        conf_val = float(edge.get("substitute_confidence", 0.0) or 0.0)
                        max_sub_conf = max(max_sub_conf, conf_val)
                        penalty += 0.5 * conf_val
                
                score -= penalty
                # Add drift penalty (soft); hard-drop happens after sorting.
                score += 0.75 * num_unrelated + 0.35 * num_weak
                
                all_found_paths.append({
                    "edges": list(path_edges),
                    "score": score,
                    "num_substitute_edges": num_sub_edges,
                    "max_substitute_confidence": max_sub_conf,
                    "num_unrelated_to_target": num_unrelated,
                    "num_weakly_related_to_target": num_weak,
                })
                return
            
            for edge in graph.get(node, []):
                tail = edge["tail"]
                # 防止环路
                if any(e["tail"].lower() == tail.lower() for e in path_edges): continue
                
                path_edges.append(edge)
                dfs(tail, path_edges, current_score)
                path_edges.pop()

        dfs(start, [], 0)
        
        # 4. 排序与截断
        # 优先按分数排序(越低越好)，分数相同按字符串排序(保证确定性)
        all_found_paths.sort(key=lambda x: (x["score"], str(x["edges"])))
        
        # Hard-drop: if a path contains 2+ nodes explicitly "unrelated" to the target,
        # it is likely semantic drift (e.g., erosion -> hazards -> volcano -> lava).
        kept_paths = [p for p in all_found_paths if int(p.get("num_unrelated_to_target", 0) or 0) < 2][:10]

        # === Top-K path verification stage (Counterfactual + Substitution) ===
        # We ONLY run expensive/brittle checks for edges that actually appear in kept_paths.
        # This improves robustness vs. edge-level filtering during BFS.
        if kept_paths:
            # Gather unique edges from Top-K paths
            uniq_edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
            for p in kept_paths:
                for e in p.get("edges", []) or []:
                    hh = str(e.get("head", "") or "").strip()
                    rr = str(e.get("relation", "") or "").strip().upper()
                    tt = str(e.get("tail", "") or "").strip()
                    if not hh or not rr or not tt:
                        continue
                    if "INCREASE" in rr:
                        rr = "INCREASES"
                    elif "DECREASE" in rr:
                        rr = "DECREASES"
                    key = (hh, rr, tt)
                    if key not in uniq_edges:
                        uniq_edges[key] = {"cf": None, "sub": None}

            # Run counterfactual verification (edge-level, but only on Top-K path edges)
            for (hh, rr, tt) in uniq_edges.keys():
                try:
                    cf = self._verify_edge_counterfactual(hh, rr, tt) or {"is_valid": True, "confidence": 0.5, "reasoning": ""}
                except Exception as e:
                    cf = {"is_valid": True, "confidence": 0.5, "reasoning": f"Exception: {e}"}
                try:
                    cf_conf = float(cf.get("confidence", 0.5))
                except Exception:
                    cf_conf = 0.5
                cf_conf = max(0.0, min(1.0, cf_conf))
                uniq_edges[(hh, rr, tt)]["cf"] = {
                    "is_valid": bool(cf.get("is_valid", True)),
                    "confidence": cf_conf,
                    "reasoning": str(cf.get("reasoning", "")).strip(),
                }

            # Run substitution detection (edge-level, only on Top-K path edges)
            # NOTE: this is treated as a SOFT tag; we do NOT drop edges here.
            for (hh, rr, tt) in uniq_edges.keys():
                try:
                    sub = self._check_counterfactual_substitution(hh, tt, getattr(self, "question", ""))
                except Exception:
                    sub = {"is_substitute": False, "confidence": 0.0, "reasoning": ""}
                try:
                    sub_conf = float(sub.get("confidence", 0.0))
                except Exception:
                    sub_conf = 0.0
                sub_conf = max(0.0, min(1.0, sub_conf))
                uniq_edges[(hh, rr, tt)]["sub"] = {
                    "is_substitute": bool(sub.get("is_substitute", False)),
                    "confidence": sub_conf,
                    "reasoning": str(sub.get("reasoning", "")).strip(),
                }

            # Attach metadata back onto each edge in each path, and adjust path score.
            for p in kept_paths:
                edges = p.get("edges", []) or []
                cf_penalty = 0.0
                sub_penalty = 0.0
                for e in edges:
                    hh = str(e.get("head", "") or "").strip()
                    rr = str(e.get("relation", "") or "").strip().upper()
                    tt = str(e.get("tail", "") or "").strip()
                    if "INCREASE" in rr:
                        rr = "INCREASES"
                    elif "DECREASE" in rr:
                        rr = "DECREASES"
                    key = (hh, rr, tt)
                    meta = uniq_edges.get(key, {})

                    cf_meta = meta.get("cf") or {"is_valid": True, "confidence": 0.5, "reasoning": ""}
                    e["cf_is_valid"] = bool(cf_meta.get("is_valid", True))
                    e["cf_confidence"] = float(cf_meta.get("confidence", 0.5) or 0.5)
                    e["cf_reasoning"] = str(cf_meta.get("reasoning", "")).strip()
                    if e["cf_is_valid"] is False:
                        # Penalize suspicious edges; do not hard-drop to avoid 0-path failures.
                        cf_penalty += 0.6 * float(e["cf_confidence"] or 0.0)

                    sub_meta = meta.get("sub") or {"is_substitute": False, "confidence": 0.0, "reasoning": ""}
                    e["is_substitute"] = bool(sub_meta.get("is_substitute", False))
                    e["substitute_confidence"] = float(sub_meta.get("confidence", 0.0) or 0.0)
                    if e["is_substitute"]:
                        sub_penalty += 0.5 * float(e["substitute_confidence"] or 0.0)

                p["cf_penalty"] = cf_penalty
                p["sub_penalty"] = sub_penalty
                p["score"] = float(p.get("score", 0.0) or 0.0) + cf_penalty - sub_penalty

            # Re-sort after verification, keep Top-K
            kept_paths.sort(key=lambda x: (x.get("score", 0.0), str(x.get("edges"))))
            kept_paths = kept_paths[:10]
        
        all_nodes = set()
        for p in kept_paths:
            for e in p["edges"]:
                all_nodes.add(e["head"])
                all_nodes.add(e["tail"])

        shortest = min(len(p["edges"]) for p in kept_paths) if kept_paths else None
        
        return {
            "start": start, 
            "target": target, 
            "paths": kept_paths, 
            "num_paths": len(kept_paths), 
            "shortest_path_length": shortest, 
            "all_nodes_in_paths": all_nodes
        }
    
    def show_causal_graph(self):
        """
        Display the constructed causal graph.
        """
        print("self.X:", self.X)
        print("self.Y:", self.Y)
        print("self.Z:", self.Z)
        print("self.A:", self.A)
        print("self.D:", self.D)
        print("self.W:", self.W)
        print("self.U:", self.U)
        print("self.V:", self.V)

    def _edge_relation_to_sign(self, relation: str) -> int:
        """
        [Modified] explicit support for INCREASES/DECREASES
        """
        if not relation:
            return 0
        r = str(relation).strip().upper()
        
        # 扩展词表
        POSITIVE = {"INCREASES", "RESULTS_IN", "CAUSES", "PROMOTES", "ENABLES"}
        NEGATIVE = {"DECREASES", "NOT_RESULTS_IN", "PREVENTS", "INHIBITS", "BLOCKS", "REDUCES"}

        if r in POSITIVE: return 1
        if r in NEGATIVE: return -1
        
        # 模糊匹配兜底
        if "INCREASE" in r or "MORE" in r: return 1
        if "DECREASE" in r or "LESS" in r or "PREVENT" in r: return -1
        
        return 0

    def calculate_chain_sign(self, path: List[Dict[str, Any]]) -> int:
        """
        Compute the net causal sign of a path.

        Strategy:
        - For each edge, map its relation to a sign in {-1, 0, +1}
          using `_edge_relation_to_sign`.
        - Ignore neutral edges (sign == 0).
        - Combine non-zero edge signs by multiplication (parity):
            * an odd number of negative edges => net negative (-1)
            * an even number of negative edges => net positive (+1)
        - If all edges are neutral, return 0.
        """
        if not path:
            return 0

        if isinstance(path, dict):
            edges = path.get("edges") or []
        else:
            edges = path

        # Collect non-zero signs.
        signs: List[int] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            # Try common keys for the relation label.
            relation = edge.get("relation") or edge.get("label") or ""
            s = self._edge_relation_to_sign(relation)
            if s != 0:
                signs.append(s)

        if not signs:
            # All edges were neutral / unknown.
            return 0

        net = 1
        for s in signs:
            net *= s

        return 1 if net > 0 else -1

    def _check_path_consistency(self, start_node: str, path: List[Dict[str, str]]) -> bool:
        """
        前提一致性检查 (Premise Consistency Check)。

        目的：防止因果链在推导过程中“篡改”了起点设定的前提。
        例子：起点是 "No clouds" (无云)，路径中间却出现了 "Cloud formation" (云形成)。
        虽然物理上这是可能的（长期循环），但在回答“无云的影响”这一因果问题时，
        这种路径属于逻辑悖论（为了得到结果，必须先否定前提）。

        返回:
        - True: 路径一致（未违反前提）。
        - False: 路径违规（中间节点与起点矛盾）。
        """
        # 提取路径中的中间节点（不包含起点和终点，因为终点是我们要预测的，起点是已知的）
        # 我们只关心中间过程是否推翻了起点
        if len(path) < 2:
            return True

        intermediate_nodes = [edge.get("tail", "") for edge in path[:-1]]
        nodes_str = ", ".join(intermediate_nodes)

        prompt = f"""
You are a Logic Consistency Validator. We are analyzing a causal path to answer a "What if" question.

Start Event (The Premise): "{start_node}"
Intermediate Steps in Path: [{nodes_str}]

Task:
Check if any of the Intermediate Steps logically CONTRADICT or NEGATE the Start Event in the *immediate* context.

**Critical Rules:**
1. **No "Undoing" the Premise:** If the Start Event says something is ABSENT (e.g., "No clouds", "No food"), and the steps rely on PRODUCING or HAVING that thing (e.g., "Cloud formation", "Eating food") to proceed, this is INVALID.
2. **Immediate vs. Long-term:** We are looking for the *direct consequence* of the Start Event. Do not accept long-term feedback loops that restore the missing factor (e.g., "No clouds -> Evaporation -> Clouds form"). This is a logical loop, not a direct effect.

Example 1:
- Start: "No clouds"
- Steps: "Sunlight reaches ground", "Water evaporates", "Cloud formation"
- Verdict: INVALID. (Step 'Cloud formation' contradicts the premise 'No clouds'. You can't assume clouds form to answer what happens when there are no clouds).

Example 2:
- Start: "No clouds"
- Steps: "Sunlight reaches ground", "Soil dries out"
- Verdict: VALID. (Consistent with no clouds).

Is this path logically consistent with maintaining the premise?

Output ONLY JSON:
{{"is_consistent": true/false, "reasoning": "Explain violation if any"}}
""".strip()

        try:
            response = self._call_llm(prompt)
            response = self._clean_response(response)
            data = json.loads(response)
            is_consistent = bool(data.get("is_consistent", True))

            if not is_consistent:
                print(f"[Consistency] REJECTED Path: Start='{start_node}' -> Steps={nodes_str}")
                print(f"  Reason: {data.get('reasoning', '')}")

            return is_consistent
        except Exception as e:
            print(f"_check_path_consistency error: {e}")
            return True  # 出错时默认放行，避免因解析失败导致整条路径丢失

    def _format_paths_for_llm(
        self,
        kept_paths: List[List[Dict[str, Any]]],
    ) -> str:
        """
        Deprecated formatting helper (kept for backward compatibility if needed).
        New logic uses _build_paths_payload_for_llm instead.
        """
        return "Structured path payload is provided separately."

    def _filter_paths_for_llm(
        self,
        paths: List[List[Dict[str, Any]]],
        max_len: int = 6,
        max_paths: int = 8,
    ) -> List[List[Dict[str, Any]]]:
        """
        Light-weight filtering before sending paths to the final LLM:
        - Drop paths that are too long (len > max_len).
        - Sort by length (shorter first).
        - Deduplicate by the (head, tail, relation) sequence.
        - Keep at most max_paths paths.
        We keep diversity and do NOT try to do any voting or scoring here.
        """
        if not paths:
            return []

        def _edges_of(path_obj: Any) -> List[Dict[str, Any]]:
            if isinstance(path_obj, dict):
                edges_val = path_obj.get("edges")
                return edges_val if isinstance(edges_val, list) else []
            return path_obj if isinstance(path_obj, list) else []

        path_edge_pairs = [(p, _edges_of(p)) for p in paths]

        # 1) filter by length
        filtered = [(p, e) for p, e in path_edge_pairs if len(e) > 0 and len(e) <= max_len]
        if not filtered:
            # if all paths are too long, fall back to the original ones (shortest few)
            paths_sorted = sorted(path_edge_pairs, key=lambda pe: len(pe[1]) if pe[1] else float("inf"))
            return [pe[0] for pe in paths_sorted[:max_paths]]

        # 2) sort by length ascending
        filtered = sorted(filtered, key=lambda pe: len(pe[1]))

        # 3) deduplicate by edge sequence
        seen = set()
        unique_paths: List[List[Dict[str, Any]]] = []
        for p, edges in filtered:
            key = tuple((e.get("head", ""), e.get("relation", ""), e.get("tail", "")) for e in edges)
            if key in seen:
                continue
            seen.add(key)
            unique_paths.append(p)

        # 4) cap total number
        return unique_paths[:max_paths]

    def _build_paths_payload_for_llm(self, paths: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Convert internal path representation into a JSON-friendly payload for the final LLM.

        Each item in the returned list:
        - id: int
        - length: int
        - net_sign: +1 / -1 / 0 (parity product of edge.sign, ignoring zeros)
        - edges: list of {head, tail, relation, sign, source}
        """
        payload: List[Dict[str, Any]] = []

        def _edges_of(path_obj: Any) -> List[Dict[str, Any]]:
            if isinstance(path_obj, dict):
                edges_val = path_obj.get("edges")
                return edges_val if isinstance(edges_val, list) else []
            return path_obj if isinstance(path_obj, list) else []

        for idx, path in enumerate(paths, start=1):
            edges = _edges_of(path)
            edges_info: List[Dict[str, Any]] = []
            # Compute net sign by PARITY (multiplication), consistent with calculate_chain_sign().
            # This avoids the classic bug: DECREASES then DECREASES => net INCREASES (positive),
            # which sum-based scoring would get wrong.
            sign_product = 1
            has_nonzero = False
            sub_edges = 0
            max_sub_conf = 0.0
            for e in edges:
                # Prefer an existing sign field; otherwise infer from relation.
                s = e.get("sign")
                if s is None:
                    s = self._edge_relation_to_sign(e.get("relation", ""))
                try:
                    sign = int(s)
                except Exception:
                    sign = 0
                # Normalize sign to -1/0/+1
                if sign > 0:
                    sign = 1
                elif sign < 0:
                    sign = -1
                else:
                    sign = 0
                if sign != 0:
                    has_nonzero = True
                    sign_product *= sign
                is_sub = bool(e.get("is_substitute", False))
                sub_conf = float(e.get("substitute_confidence", 0.0) or 0.0)
                if is_sub:
                    sub_edges += 1
                    max_sub_conf = max(max_sub_conf, sub_conf)
                edges_info.append({
                    "head": e.get("head", ""),
                    "tail": e.get("tail", ""),
                    "relation": e.get("relation", ""),
                    "sign": sign,
                    "source": e.get("source", "bfs"),
                    "is_substitute": is_sub,
                    "substitute_confidence": sub_conf,
                })

            net_sign = 0
            if has_nonzero:
                net_sign = 1 if sign_product > 0 else -1

            payload.append({
                "id": idx,
                "length": len(edges),
                "net_sign": net_sign,
                "score": path.get("score") if isinstance(path, dict) else None,
                "num_substitute_edges": path.get("num_substitute_edges") if isinstance(path, dict) else sub_edges,
                "max_substitute_confidence": path.get("max_substitute_confidence") if isinstance(path, dict) else max_sub_conf,
                "edges": edges_info,
            })
        return payload

    def causal_chain_to_text(self, chain_result: Dict[str, Any], bfs_result: Optional[Dict[str, Any]] = None) -> str:
        """
        Convert chains to text description.
        [Fixed] Display logic to handle INCREASES/DECREASES correctly.
        """
        start = chain_result.get("start", "")
        target = chain_result.get("target", "")
        paths = chain_result.get("paths", [])
        
        lines = []
        if paths:
            lines.append(f"From '{start}' to '{target}', the system found {len(paths)} causal path(s).")
            for i, path in enumerate(paths[:5], 1):
                path_edges = path.get("edges", path) if isinstance(path, dict) else path
                steps = []
                for e in path_edges:
                    r = e["relation"]
                    # 使用 helper 判断符号，而不是字符串匹配
                    sign = self._edge_relation_to_sign(r)
                    if sign > 0:
                        rel_txt = "INCREASES"
                    elif sign < 0:
                        rel_txt = "DECREASES"
                    else:
                        rel_txt = r # Fallback
                        
                    steps.append(f"({e['head']}) -> [{rel_txt}] -> ({e['tail']})")
                lines.append(f"Path {i}: " + " ; ".join(steps))
            
            pos = sum(1 for p in paths if self.calculate_chain_sign(p) > 0)
            neg = sum(1 for p in paths if self.calculate_chain_sign(p) < 0)
            neu = len(paths) - pos - neg
            lines.append(f"Statistical Summary: {pos} positive chains, {neg} negative chains, {neu} neutral/unclear.")
        else:
            lines.append(
                f"The graph did not explicitly contain a complete path from '{start}' to the base variable '{target}'. "
                "This may reflect missing knowledge rather than true independence."
            )

            if bfs_result and bfs_result.get("close_hits"):
                lines.append("Related variables near the base that may indirectly affect it:")
                for i, hit in enumerate(bfs_result["close_hits"][:10], 1):
                    node = hit.get("node", "")
                    bfs_eq = hit.get("bfs_equivalence", "")
                    expl = hit.get("explanation") or hit.get("reasoning") or ""
                    if expl:
                        lines.append(f"- {node} ({bfs_eq}): {expl}")
                    else:
                        lines.append(f"- {node} ({bfs_eq})")
            if bfs_result and bfs_result.get("triples"):
                lines.append("Observed 1-hop relations:")
                for i, triple in enumerate(bfs_result["triples"][:10], 1):
                    if isinstance(triple, dict):
                        h, r, t = triple.get("head", ""), triple.get("relation", ""), triple.get("tail", "")
                    else:
                        try:
                            h, r, t = triple
                        except Exception:
                            continue
                    # 同样的修复
                    sign = self._edge_relation_to_sign(r)
                    rel_txt = "INCREASES" if sign > 0 else ("DECREASES" if sign < 0 else r)
                    lines.append(f"Edge {i}: {h} --[{rel_txt}]--> {t}")
            else:
                lines.append("No causal relations found in the graph.")
        
        return "\n".join(lines)
    
    
    def _fallback_parse_llm_output(self, raw: str) -> Dict[str, Any]:
        """
        Fallback parser when JSON decoding of the LLM output fails.

        We do NOT try to parse the full JSON structure.
        Instead, we use regex to extract a few key fields that are most
        important for downstream logic:
        - effect_on_base
        - confidence
        - reasoning

        This is meant to be robust even if the 'paths_eval' array is malformed
        or the overall JSON wrapper is slightly broken.
        """

        def _extract_str_field(key: str, default: str) -> str:
            # Match patterns like: "key": "value"
            pattern = rf'"{key}"\s*:\s*"([^"]+)"'
            m = re.search(pattern, raw)
            if m:
                return m.group(1).strip()
            return default

        effect = _extract_str_field("effect_on_base", "unknown")
        confidence = _extract_str_field("confidence", "low")
        reasoning = _extract_str_field(
            "reasoning",
            "Failed to parse full LLM JSON; used regex fallback.",
        )

        # Normalize to lowercase where appropriate; 'reasoning' can keep case.
        effect = (effect or "unknown").strip().lower()
        confidence = (confidence or "low").strip().lower()

        return {
            "effect_on_base": effect,
            "confidence": confidence,
            "reasoning": reasoning,
            "paths_eval": [],
        }

    def _final_llm_decision(
        self,
        question: str,
        cause_event: str,
        outcome_event: str,
        outcome_base: str,
        paths: List[Dict[str, Any]],
        choices: Optional[Dict[str, List[str]]] = None,
        graph_summary: Optional[Dict[str, Any]] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Final LLM aggregation for WIQA-style causal reasoning.
        """

        # 1. 将路径转化为自然语言描述 (Evidence Chains)
        # 我们把符号化的 sign 转换成自然语言，方便 8B 模型理解
        path_texts = []
        for p in paths or []:
            chain_parts = []
            p_edges = p.get("edges", []) or []
            bridge_edges = 0
            for e in p_edges:
                s = e.get("sign", 0)
                # 使用明确的动词
                rel_word = "INCREASES" if s > 0 else ("DECREASES" if s < 0 else "AFFECTS")
                if str(e.get("source", "")).strip().lower() == "bridge":
                    bridge_edges += 1
                chain_parts.append(f"({e.get('head')}) -> {rel_word} -> ({e.get('tail')})")
            
            # 计算这一条路径的净影响
            net = p.get("net_sign", 0)
            net_str = "POSITIVE (Causes Increase)" if net > 0 else ("NEGATIVE (Causes Decrease)" if net < 0 else "UNCLEAR")
            
            path_str = " -> ".join(chain_parts)
            path_texts.append(f"- Chain: {path_str} [Net Effect: {net_str}] [bridge_edges: {bridge_edges}/{len(p_edges)}]")
        
        evidence_block = "\n".join(path_texts) if path_texts else "(No clear causal paths found)"
        summary_json = json.dumps(graph_summary or {}, ensure_ascii=False, indent=2)

        # 2. Prompt：强调即使无路径也要根据常识作答，且避免默认 no_effect
        prompt = f"""
You are solving a WIQA-style causal reasoning problem.
Your job is to decide how the CAUSE affects the BASE VARIABLE.

Question: "{question}"
Cause event: "{cause_event}"
Outcome event (surface text): "{outcome_event}"
BASE variable (outcome_base, the only quantity you judge): "{outcome_base}"

Causal graph summary (may be incomplete or noisy):
{summary_json}

Evidence chains from cause → BASE (system-computed net effects; DO NOT re-multiply signs yourself):
{evidence_block}

Raw chain trace (may be noisy; optional for intuition):
{description}

IMPORTANT:
- You must decide the direction of change for the BASE VARIABLE only
  ("{outcome_base}"), NOT for the surface wording of the outcome sentence.
  Ignore "more"/"less" in the question text; mapping back to choices is
  handled outside of you.
 - In Evidence chains, "[Net Effect: POSITIVE]" means the BASE VARIABLE INCREASES.
   "[Net Effect: NEGATIVE]" means the BASE VARIABLE DECREASES.
 - The graph can be wrong or incomplete. Use it as evidence, but you may
   override it using commonsense and the question text.
 - If the graph summary indicates bridge-heavy evidence (e.g., "bridge_heavy": true),
   treat graph evidence as weak and rely more on general commonsense and the question itself.
 - "no_effect" is a rare answer. Only use "no_effect" if it is truly
   reasonable to treat the cause as approximately independent of the BASE
   VARIABLE, even after considering plausible mechanisms.
- NEVER choose "no_effect" just because:
  * there is no path in the graph, or
  * some chains are positive and some are negative, or
  * you feel uncertain.
  In those cases, choose "more" or "less" and lower your confidence.
- If most strong/credible chains are positive and commonsense agrees,
  choose "more".
- If most strong/credible chains are negative and commonsense agrees,
  choose "less".
- If evidence is mixed but biased (e.g., 2 positive vs 1 negative and
  commonsense supports the positive story), still choose that biased
  direction, but with lower confidence.
- Only if the cause is clearly irrelevant or orthogonal to the BASE
  VARIABLE, choose "no_effect".

Choice mapping for this dataset:
- "more"    → answer choice "A"
- "less"    → answer choice "B"
- "no_effect" → answer choice "C"

Output ONLY strict JSON:
{{
  "effect_on_base": "more" | "less" | "no_effect",
  "predicted_choice": "A" | "B" | "C",
  "confidence": "high" | "medium" | "low" | "very_low",
  "reasoning": "Short explanation of why the BASE VARIABLE goes up, down, or stays effectively unchanged."
}}
""".strip()

        # 3. 调用 LLM
        raw = self._call_llm(prompt)
        cleaned = self._clean_response(raw or "")

        # 4. 鲁棒解析 (Regex + Fallback)
        import re
        data = {}
        
        # 尝试提取 JSON 块（8B 经常会在 JSON 前后加废话）
        match = re.search(r'\{[\s\S]*\}', cleaned)
        if match:
            try:
                data = json.loads(match.group(0))
            except:
                pass 
        
        # 提取字段
        effect = str(data.get("effect_on_base", "unknown")).strip().lower()
        reasoning = str(data.get("reasoning", "Extracted via fallback."))
        confidence = str(data.get("confidence", "low"))
        confidence_score = self._normalize_confidence_score(confidence)

        # 5. 关键词兜底 (Text Fallback)
        # 如果 JSON 解析失败，或者模型输出了 "unknown"，则在文本里找关键词
        allowed = ["more", "less", "no_effect"]
        if effect not in allowed:
            lowered_raw = cleaned.lower()
            if '"more"' in lowered_raw or "'more'" in lowered_raw:
                effect = "more"
            elif '"less"' in lowered_raw or "'less'" in lowered_raw:
                effect = "less"
            

        return {
            "effect_on_base": effect,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "reasoning": reasoning,
            # 兼容旧接口，返回空字典
            "scores": {},
            "paths_eval": []
        }

    def _map_effect_to_answer(
        self,
        effect_on_base: str,
        question_direction: str,
    ) -> str:
        """
        Map (effect_on_base, question_direction) to one of:
        'more', 'less', 'no_effect'.

        effect_on_base: 'more' / 'less' / 'no_effect' / 'unknown'
        question_direction: 'MORE' / 'LESS' / 'NONE'
            - 'MORE': the question is of the form "MORE X"
            - 'LESS': the question is of the form "LESS X"
            - 'NONE': the question statement uses the base variable directly,
                      or has no explicit MORE/LESS marker.

        Returns:
            predicted_effect: 'more' / 'less' / 'no_effect'
        """

        # Normalize
        e = (effect_on_base or "unknown").lower()
        q = (question_direction or "NONE").upper()

        if e not in ("more", "less", "no_effect"):
            e = "no_effect"

        # No modifier: just use the base effect as-is
        if q == "NONE":
            return e

        # Question asks about "MORE X":
        # If base outcome increases -> answer 'more'
        # If base outcome decreases -> answer 'less'
        # If no_effect -> 'no_effect'
        if q == "MORE":
            return e

        # Question asks about "LESS X":
        # Intuition: think of outcome = "less X".
        # If base increases -> "less X" becomes less likely -> answer 'less'.
        # If base decreases -> "less X" becomes more likely -> answer 'more'.
        # If no_effect -> 'no_effect'.
        if q == "LESS":
            if e == "more":
                return "less"
            if e == "less":
                return "more"
            return "no_effect"

        # Fallback
        return e

    def _effect_to_choice(
        self,
        label: str,
        choices: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """
        Map WIQA textual answer (more/less/no_effect) -> label (A/B/C).
        """
        choices = choices or {}
        if choices and "text" in choices and "label" in choices:
            lab_norm = (label or "").strip().lower().replace(" ", "_")
            for lbl, txt in zip(choices["label"], choices["text"]):
                txt_norm = str(txt).strip().lower().replace(" ", "_")
                if lab_norm == txt_norm:
                    return str(lbl).strip().upper()
        default_map = {"more": "A", "less": "B", "no_effect": "C"}
        return default_map.get(label, "")

    def _normalize_confidence_score(self, conf: Any) -> float:
        """
        Normalize textual/numeric confidence to [0,1].
        """
        if conf is None:
            return 0.0
        try:
            val = float(conf)
            return max(0.0, min(1.0, val))
        except Exception:
            pass
        conf_str = str(conf).strip().lower()
        mapping = {
            "very_high": 0.95,
            "high": 0.85,
            "medium": 0.65,
            "low": 0.4,
            "very_low": 0.25,
            "unknown": 0.5,
        }
        return mapping.get(conf_str, 0.5)

    def resolve_effect_conflict(self, graph_effect: str, graph_conf: float,
                                llm_effect: str, llm_conf: float) -> Dict[str, Any]:
        """
        Resolve conflicts between graph signal and LLM judgement.
        """
        graph_effect = (graph_effect or "no_effect").strip().lower()
        llm_effect = (llm_effect or "no_effect").strip().lower()
        graph_conf = max(0.0, min(1.0, float(graph_conf or 0.0)))
        llm_conf = max(0.0, min(1.0, float(llm_conf or 0.0)))

        allowed = {"more", "less", "no_effect"}
        if graph_effect not in allowed:
            graph_effect = "no_effect"
        if llm_effect not in allowed:
            llm_effect = "no_effect"

        if graph_effect == llm_effect:
            final_effect = graph_effect
            final_conf = max(graph_conf, llm_conf)
            source = "agree"
        # Lower the threshold so we are more willing to take a directional decision.
        elif graph_conf >= 0.7 and llm_conf <= 0.55:
            final_effect = graph_effect
            final_conf = graph_conf
            source = "graph_dominant"
        elif llm_conf >= 0.7 and graph_conf <= 0.55:
            final_effect = llm_effect
            final_conf = llm_conf
            source = "llm_dominant"
        else:
            # Make "no_effect" a true last resort:
            # - If either side proposes a direction (more/less), keep that direction.
            # - Only output no_effect when BOTH sides output no_effect.
            if graph_effect in {"more", "less"} and llm_effect == "no_effect":
                final_effect = graph_effect
                final_conf = min(graph_conf, 0.55)  # keep confidence conservative
                source = "direction_over_no_effect_graph"
            elif llm_effect in {"more", "less"} and graph_effect == "no_effect":
                final_effect = llm_effect
                final_conf = min(llm_conf, 0.55)
                source = "direction_over_no_effect_llm"
            elif graph_effect in {"more", "less"} and llm_effect in {"more", "less"}:
                # Both directional but disagree.
                # If the graph has a reasonably confident, sign-consistent path signal,
                # do NOT let a slightly-higher-confidence LLM overturn it.
                # Only override the graph when the LLM is substantially more confident.
                if graph_conf >= 0.55:
                    # Strong, sign-consistent graph signal: always trust the graph.
                    final_effect = graph_effect
                    final_conf = min(graph_conf, 0.55)
                    source = "conflict_pick_graph"
                else:
                    # Graph weak: fall back to higher-confidence; tie -> prefer graph.
                    if graph_conf >= llm_conf:
                        final_effect = graph_effect
                        final_conf = min(graph_conf, 0.55)
                        source = "conflict_pick_graph"
                    else:
                        final_effect = llm_effect
                        final_conf = min(llm_conf, 0.55)
                        source = "conflict_pick_llm"
            else:
                # Both are no_effect (or invalid normalized to no_effect): keep no_effect.
                final_effect = "no_effect"
                final_conf = min(max(graph_conf, llm_conf), 0.4)
                source = "both_no_effect"

        return {
            "effect_on_base": final_effect,
            "confidence": final_conf,
            "decision_source": source,
        }

    def _summarize_graph_signal(self, chain_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute a coarse graph-based effect and confidence from chain_result.
        """
        if not chain_result:
            return {
                "graph_effect": "no_effect",
                "graph_confidence": 0.2,
                "best_path": None,
                "num_paths": 0,
                "shortest": None,
            }
        paths = chain_result.get("paths") or []
        num_paths = len(paths)
        shortest = chain_result.get("shortest_path_length")

        if num_paths == 0:
            return {
                "graph_effect": "no_effect",
                "graph_confidence": 0.2,
                "best_path": None,
                "num_paths": 0,
                "shortest": shortest,
            }

        best_path = paths[0]
        edges = best_path.get("edges") if isinstance(best_path, dict) else best_path
        if shortest is None and edges:
            shortest = len(edges)
        best_sign = self.calculate_chain_sign(best_path)

        # Path-level reliability: downweight bridge-heavy paths when summarizing sign.
        import math

        def _edges_of(path_obj: Any) -> List[Dict[str, Any]]:
            if isinstance(path_obj, dict):
                edges_val = path_obj.get("edges") or []
                return edges_val if isinstance(edges_val, list) else []
            return path_obj if isinstance(path_obj, list) else []

        weighted_pos = 0.0
        weighted_neg = 0.0
        bridge_counts: List[int] = []
        last_edge_is_bridge: List[bool] = []

        for p in paths:
            p_edges = _edges_of(p)
            if not p_edges:
                continue
            num_bridge = sum(
                1
                for e in p_edges
                if isinstance(e, dict) and str(e.get("source", "")).strip().lower() == "bridge"
            )
            bridge_counts.append(int(num_bridge))
            try:
                last_edge_is_bridge.append(
                    isinstance(p_edges[-1], dict)
                    and str(p_edges[-1].get("source", "")).strip().lower() == "bridge"
                )
            except Exception:
                last_edge_is_bridge.append(False)

            # Weighting option: exp(-num_bridge_edges) (keeps "all-bridge" paths as weak evidence, not zero).
            w = float(math.exp(-float(num_bridge)))
            s = self.calculate_chain_sign(p)
            if s > 0:
                weighted_pos += w
            elif s < 0:
                weighted_neg += w

        if weighted_pos > weighted_neg:
            graph_effect = "more"
        elif weighted_neg > weighted_pos:
            graph_effect = "less"
        else:
            # Mixed / weak signal: treat as no_effect at the graph-summary level.
            graph_effect = "no_effect"

        # Confidence heuristic
        base_conf = 0.5
        if num_paths >= 3:
            base_conf += 0.15
        elif num_paths >= 1:
            base_conf += 0.1
        if shortest:
            if shortest <= 2:
                base_conf += 0.2
            elif shortest <= 4:
                base_conf += 0.1
        sub_edges = 0
        max_sub_conf = 0.0
        if isinstance(best_path, dict):
            sub_edges = best_path.get("num_substitute_edges") or 0
            max_sub_conf = best_path.get("max_substitute_confidence") or 0.0
        if sub_edges == 0 and edges:
            sub_edges = sum(1 for e in edges if isinstance(e, dict) and e.get("is_substitute"))
            max_sub_conf = max([float(e.get("substitute_confidence", 0.0) or 0.0) for e in edges if isinstance(e, dict)] or [0.0])
        penalty = 0.15 * sub_edges + 0.25 * max_sub_conf
        graph_conf = max(0.0, min(1.0, base_conf - penalty))

        # Penalize mixed-sign evidence so many weak chains don't create fake confidence.
        mix_penalty = 0.0
        if weighted_pos > 0.0 and weighted_neg > 0.0:
            mix = min(weighted_pos, weighted_neg) / max(weighted_pos + weighted_neg, 1e-9)  # [0, 0.5]
            mix_penalty = 0.25 * (mix / 0.5)  # max 0.25 when perfectly split
            graph_conf = max(0.0, graph_conf - mix_penalty)

        # Bridge-heavy evidence cap: if every path relies on bridge edges (esp. into the base), keep graph_conf low.
        all_paths_have_bridge = bool(bridge_counts) and all(c > 0 for c in bridge_counts)
        all_paths_last_edge_bridge = bool(last_edge_is_bridge) and all(last_edge_is_bridge)
        bridge_heavy = bool(all_paths_have_bridge or all_paths_last_edge_bridge)
        if all_paths_have_bridge:
            graph_conf = min(graph_conf, 0.4)
        if all_paths_last_edge_bridge:
            graph_conf = min(graph_conf, 0.35)

        if graph_effect == "no_effect":
            graph_conf = min(graph_conf, 0.4)

        return {
            "graph_effect": graph_effect,
            "graph_confidence": graph_conf,
            "best_path": best_path,
            "num_paths": num_paths,
            "shortest": shortest,
            "num_substitute_edges": sub_edges,
            "max_substitute_confidence": max_sub_conf,
            "weighted_pos_score": weighted_pos,
            "weighted_neg_score": weighted_neg,
            "mix_penalty": mix_penalty,
            "bridge_heavy": bridge_heavy,
            "all_paths_have_bridge": all_paths_have_bridge,
            "all_paths_last_edge_bridge": all_paths_last_edge_bridge,
            "avg_bridge_edges_per_path": (sum(bridge_counts) / len(bridge_counts)) if bridge_counts else 0.0,
            "max_bridge_edges_in_path": max(bridge_counts) if bridge_counts else 0,
            "best_path_sign": best_sign,
        }

    def reason_with_description(
        self,
        description: str,
        chain_result: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        choices: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Final reasoning logic.

        New design:
        - Use the graph builder to produce structured causal evidence:
          paths from the cause to the outcome base, after
          bridge filtering, consistency checks, and Soft Occam.
        - Compute per-path signs and aggregated path-level statistics.
        - Delegate the final decision (more / less / no_effect) to a
          single LLM aggregator that sees the question and the selected causal chains.

        The graph builder + filters ensure that the chains are as
        causally plausible as possible; the LLM uses them as the
        main evidence for the final answer.
        """
        question = question or self.question
        question = question or self.question
        choices = choices or self.datapoint.get("choices", {}) or {}
        base = getattr(self, "outcome_base", "") or ""

        # Extract paths and start node from chain_result
        paths: List[List[Dict[str, Any]]] = []
        start_node = ""
        if chain_result and isinstance(chain_result, dict):
            paths = chain_result.get("paths", []) or []
            start_node = chain_result.get("start", "") or ""

        graph_signal = self._summarize_graph_signal(chain_result)
        best_path = graph_signal.get("best_path")
        best_score = best_path.get("score") if isinstance(best_path, dict) else None

        # Graph-level summary shown to the LLM (even when num_paths == 0)
        graph_summary = {
            "has_path": bool((chain_result or {}).get("num_paths", 0) > 0),
            "num_paths": (chain_result or {}).get("num_paths", 0),
            "shortest_path_length": (chain_result or {}).get("shortest_path_length"),
            "start": start_node,
            "target": (chain_result or {}).get("target", ""),
            "original_target": (chain_result or {}).get("original_target", ""),
            "mapped_target": (chain_result or {}).get("mapped_target", ""),
            "all_nodes_in_paths": list((chain_result or {}).get("all_nodes_in_paths", []))[:20],
            "graph_effect_hint": graph_signal.get("graph_effect"),
            "graph_confidence_estimate": graph_signal.get("graph_confidence"),
            "bridge_heavy": graph_signal.get("bridge_heavy"),
            "all_paths_have_bridge": graph_signal.get("all_paths_have_bridge"),
            "all_paths_last_edge_bridge": graph_signal.get("all_paths_last_edge_bridge"),
            "avg_bridge_edges_per_path": graph_signal.get("avg_bridge_edges_per_path"),
            "max_bridge_edges_in_path": graph_signal.get("max_bridge_edges_in_path"),
            "weighted_pos_score": graph_signal.get("weighted_pos_score"),
            "weighted_neg_score": graph_signal.get("weighted_neg_score"),
            "mix_penalty": graph_signal.get("mix_penalty"),
            "best_path_score": best_score,
            "best_path_num_substitute_edges": graph_signal.get("num_substitute_edges"),
            "best_path_max_substitute_confidence": graph_signal.get("max_substitute_confidence"),
        }
        if graph_summary["shortest_path_length"] is None:
            graph_summary["shortest_path_length"] = graph_signal.get("shortest")

        # Light-weight filtering for the LLM
        paths_for_llm = self._filter_paths_for_llm(paths, max_len=6, max_paths=8)
        if not paths_for_llm:
            # If filtering removed everything, fall back to original paths (shortest few)
            paths_for_llm = self._filter_paths_for_llm(paths, max_len=999, max_paths=4)

        # Build structured payload to show the LLM
        paths_payload = self._build_paths_payload_for_llm(paths_for_llm)

        # Call the final LLM decision
        llm_result = self._final_llm_decision(
            question=question,
            cause_event=getattr(self, "cause_event", ""),
            outcome_event=getattr(self, "outcome_event", "") or "",
            outcome_base=base,
            paths=paths_payload,
            choices=choices,
            graph_summary=graph_summary,
            description=description,
        )

        llm_effect = (llm_result.get("effect_on_base") or "unknown").strip().lower()
        if llm_effect not in {"more", "less", "no_effect"}:
            llm_effect = "unknown"
        reasoning = llm_result.get("reasoning", "")
        confidence = llm_result.get("confidence", "unknown")
        llm_conf_score = llm_result.get("confidence_score", self._normalize_confidence_score(confidence))
        # Conflict resolution between graph signal and LLM
        resolution = self.resolve_effect_conflict(
            graph_signal.get("graph_effect"),
            graph_signal.get("graph_confidence"),
            llm_effect,
            llm_conf_score,
        )
        final_effect = resolution.get("effect_on_base")
        final_confidence = resolution.get("confidence", 0.0)
        decision_source = resolution.get("decision_source", "llm_only")

        # Map effect_on_base + question direction -> predicted textual answer
        predicted_effect = self.map_effect_on_base_to_wiqa_label(final_effect)
        predicted_choice = self._effect_to_choice(predicted_effect, choices)

        combined_reason = f"[LLM] {reasoning} | [Graph] effect={graph_signal.get('graph_effect')} conf={graph_signal.get('graph_confidence')} | decision={decision_source}"
        graph_summary["final_effect_on_base"] = final_effect
        graph_summary["final_confidence_after_resolution"] = final_confidence
        graph_summary["decision_source"] = decision_source

        return {
            "predicted_answer": predicted_effect,
            "predicted_choice": predicted_choice,
            "effect_on_base": final_effect,
            "reasoning": combined_reason,
            "confidence": f"{final_confidence:.2f}",
            "debug_paths_used": paths_payload,
            "graph_summary": graph_summary,
            "llm_effect": llm_effect,
            "llm_confidence": confidence,
            "llm_confidence_score": llm_conf_score,
            "graph_effect": graph_signal.get("graph_effect"),
            "graph_confidence": graph_signal.get("graph_confidence"),
            "decision_source": decision_source,
        }
    def _verify_edge_counterfactual(self, head: str, relation: str, tail: str) -> Dict[str, Any]:
        """
        [Modified] Softer Counterfactual Verification.
        Accepts the link if 'head' is a CONTRIBUTING factor, even if not the sole cause.
        """
        # 转换关系词
        rel_str = relation.upper()
        if "INCREASE" in rel_str:
            action = "INCREASE / PROMOTE"
            cf_expect = "DECREASE / REDUCE"
            direct_q = (
                f'If "{head}" HAPPENS (vs. does NOT happen), does "{tail}" tend to be HIGHER / more likely?'
            )
            cf_q = (
                f'If "{head}" does NOT happen (vs. happens), does "{tail}" tend to be LOWER / less likely?'
            )
        elif "DECREASE" in rel_str:
            action = "DECREASE / SUPPRESS"
            cf_expect = "INCREASE / ALLOW"
            direct_q = (
                f'If "{head}" HAPPENS (vs. does NOT happen), does "{tail}" tend to be LOWER / less likely?'
            )
            cf_q = (
                f'If "{head}" does NOT happen (vs. happens), does "{tail}" tend to be HIGHER / more likely?'
            )
        else:
            # 非增减关系：默认放行（软信号）
            return {
                "is_valid": True,
                "confidence": 0.5,
                "reasoning": "Non-signed relation; skipped counterfactual sign check.",
                "suggested_relation": "NO_SUGGESTION",
            }

        prompt = f"""
You are a Common Sense Judge.
We need to verify if a causal link is reasonable.

Link: "{head}" -> {action} -> "{tail}"

Task: Check this logic using two tests. You MUST judge the SIGN (direction) as well.
1. **Direct Sense (Sign Check)**: {direct_q}
2. **Counterfactual (Sign Check)**: {cf_q}

 Important:
 - Ignore the original WIQA question/goal. Judge ONLY whether this generic link is reasonable.
 - If the link is plausible but the SIGN is backwards, output is_valid = false.
 - "{cf_expect}" is what we expect under removal if the SIGN is correct.
 - If you are UNSURE / not confident, output is_valid = true (fail-open) and explain uncertainty briefly.
 - If is_valid = false because the SIGN is backwards, set suggested_relation to the correct one ("INCREASES" or "DECREASES").
 - If is_valid = false for other reasons (nonsense/unrelated), set suggested_relation = "NO_SUGGESTION".

**Rule for "Valid"**: 
- Output TRUE if the stated SIGN is generally correct and "{head}" is a PLAUSIBLE CONTRIBUTING factor.
- Output TRUE even if other factors also affect "{tail}".
- Output FALSE if the link is clearly WRONG, BACKWARDS (sign wrong), or NONSENSE.

Example:
- "Predators" -> INCREASE -> "Mortality": TRUE (Predators are a major cause of death).
- "Good Environment" -> DECREASE -> "Survival": FALSE (Contradicts common sense).

Is this link valid?

 Output ONLY JSON:
 {{
   "is_valid": true/false,
   "confidence": 0.0-1.0,
   "reasoning": "short explanation",
   "suggested_relation": "INCREASES" | "DECREASES" | "NO_SUGGESTION"
 }}
 """
        try:
            # 增加一点 num_predict 防止截断，但保持 temperature=0
            response = ollama.generate(
                model=self.model_name, 
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 256}
            )['response']
            
            import re
            match = re.search(r'\{[\s\S]*?\}', response)
            if match:
                data = json.loads(match.group(0))
                is_valid = bool(data.get("is_valid", True))
                try:
                    conf = float(data.get("confidence", 0.5))
                except Exception:
                    conf = 0.5
                conf = max(0.0, min(1.0, conf))
                rsn = str(data.get("reasoning", "")).strip()
                suggested = str(data.get("suggested_relation", "NO_SUGGESTION") or "").strip().upper()
                if "INCREASE" in suggested:
                    suggested = "INCREASES"
                elif "DECREASE" in suggested:
                    suggested = "DECREASES"
                else:
                    suggested = "NO_SUGGESTION"
                return {"is_valid": is_valid, "confidence": conf, "reasoning": rsn, "suggested_relation": suggested}
            # 无法解析到 JSON：默认放行（软信号）
            return {
                "is_valid": True,
                "confidence": 0.5,
                "reasoning": "No JSON object found in judge output.",
                "suggested_relation": "NO_SUGGESTION",
            }
        except Exception as e:
            # 出错时默认放行，避免 Judge 把整条图杀死
            return {
                "is_valid": True,
                "confidence": 0.5,
                "reasoning": f"Exception in counterfactual judge: {e}",
                "suggested_relation": "NO_SUGGESTION",
            }
        
    def _return_no_effect(self, reason: str) -> Dict[str, Any]:
        """
        Helper to return a standardized 'no_effect' prediction.
        """
        return {
            "predicted_answer": "no_effect",
            "predicted_choice": "C",
            "effect_on_base": "no_effect",
            "reasoning": reason,
            "confidence": "low",
        }


    def run_wiqa_pipeline(
        self,
        *,
        bfs_max_depth: int = 5,
        bfs_max_relations_per_node: int = 5,
        bfs_max_nodes: int = 50,
        bfs_beam_width: int = 10,
        bridge_max_bridge_nodes: Optional[int] = 3,
        seed_max_parents: Optional[int] = 6,
        chain_max_path_length: int = 5,
    ) -> bool:
        """
        Run the full WIQACausalBuilder pipeline for one WIQA datapoint.

        This is the programmatic version of the steps executed in ``main()``,
        so it can be imported and called from other scripts.

        Parameters
        ----------
        bfs_max_depth : int
            Max BFS depth when expanding toward the target.
        bfs_max_relations_per_node : int
            Max outgoing relations generated per expanded node.
        bfs_max_nodes : int
            Max total visited nodes during BFS expansion.
        bfs_beam_width : int
            Beam width used to prune the frontier during BFS.
        bridge_max_bridge_nodes : Optional[int]
            Max number of close-hit nodes to bridge directly into the target.
            Set to 0/None to disable bridging.
        seed_max_parents : Optional[int]
            Max number of direct parent causes to seed for the target.
            Set to 0/None to disable seeding.
        chain_max_path_length : int
            Max path length when extracting causal chains from the graph.

        Returns
        -------
        bool
            Whether the predicted answer matches the gold label.
        """


        print("\n" + "="*80)
        print("步骤 0: 问题信息")
        print("="*80)
        print(f"\n问题: {self.datapoint['question_stem']}")
        print(f"正确答案: {self.datapoint['answer_label']} ({self.datapoint['answer_label_as_choice']})")

        # ========== 步骤 1: 提取起点和终点 ==========
        print("\n" + "="*80)
        print("步骤 1: 提取起点和终点")
        print("="*80)

        info = self.extract_start_entity()
        start = info["cause_event"]
        target = info["outcome_base"]

        print(f"\n提取结果:")
        print(f"  - 原因事件 (cause_event): '{start}'")
        print(f"  - 结果事件 (outcome_event): '{info['outcome_event']}'")
        print(f"  - 结果基底 (outcome_base): '{target}'")
        print(f"  - 方向词 (outcome_direction_in_question): {info['outcome_direction_in_question']}")
        print(f"  - 是否否定 (outcome_is_negated): {info.get('outcome_is_negated', False)}")

        # ========== 步骤 2: BFS 因果图扩展 ==========
        print("\n" + "="*80)
        print("步骤 2: BFS 因果图扩展")
        print("="*80)

        bfs = self.expand_toward_target(
            start_X=start,
            target_Y=target,
            max_depth=bfs_max_depth,
            max_relations_per_node=bfs_max_relations_per_node,
            max_nodes=bfs_max_nodes,
            beam_width=bfs_beam_width,
        )
        # Save drift metadata for downstream path filtering.
        try:
            self.last_node_rel_to_target = bfs.get("node_rel_to_target", {}) if isinstance(bfs, dict) else {}
        except Exception:
            self.last_node_rel_to_target = {}

        print(f"\nBFS 扩展结果:")
        print(f"  - 发现三元组数量: {len(bfs['triples'])}")
        print(f"  - 访问节点数量: {len(bfs['visited'])}")
        print(f"  - 是否找到目标: {bfs['found_target']}")
        print(f"  - 搜索深度: {bfs['depth_reached']}")
        print(f"  - Close hits 数量: {len(bfs['close_hits'])}")

        # 显示 close_hits 详情
        if bfs['close_hits']:
            print(f"\n  Close Hits 详情:")
            for i, hit in enumerate(bfs['close_hits'], 1):
                print(f"    {i}. 节点: '{hit['node']}' (深度: {hit['depth']})")

        # 显示所有三元组
        print(f"\n  发现的所有三元组:")
        for i, triple in enumerate(bfs['triples'], 1):
            if isinstance(triple, dict):
                h, r, t = triple.get("head", ""), triple.get("relation", ""), triple.get("tail", "")
            else:
                try:
                    h, r, t = triple
                except Exception:
                    continue
            print(f"    {i}. {h} --[{r}]--> {t}")

        # ========== 步骤 3: 基于 close hits 的桥接 + 提取因果链 ==========
        print("\n" + "="*80)
        print("步骤 3: 基于 close hits 的桥接 + 提取因果链")
        print("="*80)

        # 使用 bridge_close_hits 自动对 close 节点与 Y 之间的直接因果关系做一次筛选：
        # 仅当 LLM 明确给出 RESULTS_IN / NOT_RESULTS_IN 时才添加 v -> Y 的桥接边。
        if bfs["close_hits"] and int(bridge_max_bridge_nodes or 0) > 0:
            triples_with_bridges = self.bridge_close_hits(
                triples=bfs["triples"],
                close_hits=bfs["close_hits"],
                Y=target,
                max_bridge_nodes=int(bridge_max_bridge_nodes),
            )
            added_bridges = len(triples_with_bridges) - len(bfs["triples"])
            print(f"\n  通过 bridge_close_hits 添加了 {added_bridges} 条桥接边。")
        elif bfs["close_hits"]:
            triples_with_bridges = bfs["triples"]
        else:
            triples_with_bridges = bfs["triples"]
            print("\n  无 close hits，可用于桥接的节点为空。")

        # === [NEW] Target seeding: add a few plausible DIRECT parents of the target ===
        # This improves robustness when forward search never reaches the outcome domain.
        seed_edges = []
        if int(seed_max_parents or 0) > 0:
            try:
                seed_edges = self.find_target_parents(target, max_parents=int(seed_max_parents))
            except Exception:
                seed_edges = []
        if seed_edges:
            before = len(triples_with_bridges)
            existing = set()
            for entry in triples_with_bridges:
                if isinstance(entry, dict):
                    existing.add((entry.get("head", ""), entry.get("relation", ""), entry.get("tail", "")))
                elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    existing.add((entry[0], entry[1], entry[2]))
            added = 0
            for e in seed_edges:
                key = (e.get("head", ""), e.get("relation", ""), e.get("tail", ""))
                if key in existing:
                    continue
                triples_with_bridges.append(e)
                existing.add(key)
                added += 1
            if added > 0:
                print(f"\n  通过 find_target_parents 额外添加了 {added} 条目标父节点边 (target_seed)。")

        chain_result = self.get_causal_chain(
            triples_with_bridges,
            start_X=start,
            target_Y=target,
            max_path_length=chain_max_path_length,
        )

        # 如果未找到路径，尝试使用 close hit 中的 identical/part_of 节点作为替代目标再搜一次
        if chain_result.get("num_paths", 0) == 0 and bfs.get("close_hits"):
            best_alt = None
            for hit in bfs["close_hits"]:
                bfs_eq = str(hit.get("bfs_equivalence") or "").strip().lower()
                if bfs_eq != "exact_target":
                    continue
                alt_target = (hit.get("node") or "").strip()
                if not alt_target:
                    continue
                alt_chain = self.get_causal_chain(
                    triples_with_bridges,
                    start_X=start,
                    target_Y=alt_target,
                    max_path_length=chain_max_path_length,
                )
                if alt_chain.get("num_paths", 0) > 0:
                    alt_chain["mapped_target"] = alt_target
                    alt_chain["original_target"] = target
                    if best_alt is None:
                        best_alt = alt_chain
                    else:
                        current_len = best_alt.get("shortest_path_length")
                        new_len = alt_chain.get("shortest_path_length")
                        if current_len is None or (new_len is not None and new_len < current_len):
                            best_alt = alt_chain
            if best_alt:
                print(f"\n  使用 close hit '{best_alt.get('mapped_target', '')}' 作为替代目标找到了因果路径。")
                chain_result = best_alt

        print(f"\n因果链提取结果:")
        print(f"  - 找到路径数量: {chain_result['num_paths']}")
        if chain_result['shortest_path_length']:
            print(f"  - 最短路径长度: {chain_result['shortest_path_length']}")
        print(f"  - 涉及节点数: {len(chain_result['all_nodes_in_paths'])}")

        # 打印所有路径
        if chain_result['num_paths'] > 0:
            print(f"\n  所有因果路径:")
            for i, path in enumerate(chain_result["paths"], 1):
                path_edges = path.get("edges", path) if isinstance(path, dict) else path
                print(f"\n  路径 {i} (长度 {len(path_edges)}):")
                for e in path_edges:
                    print(f"    {e['head']} --[{e['relation']}]--> {e['tail']}")
        else:
            print("\n  未找到完整因果路径")

        # ========== 步骤 4: 生成文字描述 ==========
        print("\n" + "="*80)
        print("步骤 4: 生成因果分析描述")
        print("="*80)

        description = self.causal_chain_to_text(chain_result, bfs)

        print(f"\n自然语言描述:")
        print("-" * 80)
        print(description)
        print("-" * 80)

        # ========== 步骤 5: LLM 推理 ==========
        print("\n" + "="*80)
        print("步骤 5: 基于描述的 LLM 推理")
        print("="*80)

        reasoning_result = self.reason_with_description(description, chain_result=chain_result)

        print(f"\n推理结果:")
        print(f"  - 对基础变量的影响 (effect_on_base): {reasoning_result.get('effect_on_base', 'N/A')}")
        print(f"  - 最终预测答案 (predicted_answer): {reasoning_result['predicted_answer']}")
        print(f"  - 预测选项 (predicted_choice): {reasoning_result['predicted_choice']}")
        print(f"  - 置信度 (confidence): {reasoning_result['confidence']}")

        print(f"\n推理过程:")
        print("-" * 80)
        print(reasoning_result['reasoning'])
        print("-" * 80)

        # ========== 验证结果 ==========
        print("\n" + "="*80)
        print("最终验证")
        print("="*80)

        gold_label = str(self.datapoint['answer_label'])
        pred_label = str(reasoning_result['predicted_answer'])
        # 归一化后再比较，避免 'no effect' vs 'no_effect' 这种纯格式差异导致误判
        gold_norm = gold_label.strip().lower().replace(" ", "_")
        pred_norm = pred_label.strip().lower().replace(" ", "_")

        print(f"\n正确答案: {gold_label}")
        print(f"预测答案: {pred_label}")

        if pred_norm == gold_norm:
            print("\n✓ 预测正确!")
            result = True
        else:
            print(f"\n✗ 预测错误!")
            result = False

        return result
        


def main():
    # 示例数据点
    datapoint = {'question_stem': 'suppose there are no dangerous chemicals in their environment happens, how will it affect more frogs.',
 'answer_label': 'more',
 'answer_label_as_choice': 'A',
 'choices': {'text': ['more', 'less', 'no_effect'], 'label': ['A', 'B', 'C']},}

    wiqa = WIQACausalBuilder(datapoint)
    result = wiqa.run_wiqa_pipeline()
    print(result)
    
if __name__ == "__main__":
    main()
