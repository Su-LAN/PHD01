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
        model_name: str = "gemma2:27b"
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
                "num_predict": 128  # 限制输出长度，防止废话
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
        从 question 中抽取 X_q 和 outcome 相关字段。
        
        返回的基础字段（直接来自 LLM 抽取结果）：
        - "cause_event"  : 问题中的扰动事件（记作 X_q）
        - "outcome_event": 问题文本里出现的完整结果表达（例如 "LESS sound being detected",
                           或 "seeds will not be transported by the animals"）
        - "outcome_base" : 去掉 MORE/LESS 等方向词，以及显式否定词后的“基础结果变量”
                           （例如 "sound being detected", "seeds being transported by the animals"）
        - "outcome_direction_in_question": 问题中提到的方向词（例如 "MORE" / "LESS" / "NONE"）
        - "outcome_is_negated": 问题中的 OUTCOME_EVENT 是否包含显式否定
                                （例如含 "not", "no", "never", "without", "fail to", "lack of" 等）

        同时在对象上缓存：
        - self.X_q  : cause_event
        - self.A_q  : "MORE <outcome_base>"
        - self.D_q  : "LESS <outcome_base>"
        """
        prompt = f"""You are an information-extraction assistant for scientific causal questions.

Your task:
Given ONE question of the following form, extract TWO clearly separated layers of the outcome:

LAYER 1: OUTCOME_EVENT (text-level phrase)
- This is the full outcome expression as it appears in the QUESTION.
- It may include direction words like "MORE", "LESS", "GREATER amount of", "SMALLER",
  or other modifiers.
- It may also include explicit negation words like "not", "no", "never", "without",
  "fail to", "lack of", "will not be transported", etc.

LAYER 2: OUTCOME_BASE (underlying variable)
- This should represent the underlying physical/process variable that the question cares about.
- Remove direction modifiers such as "MORE", "LESS", "GREATER amount of", "SMALLER",
  "HELPING", "HURTING", etc.
- Also remove explicit negation tokens such as "not", "no", "never", "without",
  "fail to", "lack of", "will not", "cannot", etc.
- Example:
    QUESTION: "LESS sound being detected"
      OUTCOME_EVENT = "LESS sound being detected"
      OUTCOME_BASE  = "sound being detected"
    QUESTION: "seeds will not be transported by the animals"
      OUTCOME_EVENT = "seeds will not be transported by the animals"
      OUTCOME_BASE  = "seeds being transported by the animals"

Direction vs Negation:
- Direction words (for OUTCOME_DIRECTION_IN_QUESTION) include:
  "MORE", "LESS", "GREATER", "SMALLER", "HELPING", "HURTING".
- Negation words (for OUTCOME_IS_NEGATED) include:
  "not", "no", "never", "without", "fail to", "lack of", "will not", "cannot", etc.
- IMPORTANT:
  - "LESS X" is a DIRECTION (LESS), NOT a negation.
  - "no X", "not X", "X will not happen" ARE negation.

Fields to extract:
1. "cause_event"  : the cause event in the question (CAUSE_EVENT).
2. "outcome_event": the full outcome expression phrase from the question (OUTCOME_EVENT).
3. "outcome_base" : the underlying base outcome variable (OUTCOME_BASE),
                    with NO direction modifiers and NO explicit negation tokens.
4. "outcome_direction_in_question":
   - The main direction word mentioned about the outcome:
     "MORE", "LESS", "GREATER", "SMALLER", "HELPING", "HURTING",
     or "NONE" if no such directional modifier is present.
5. "outcome_is_negated":
   - "true"  if OUTCOME_EVENT is explicitly negated (has "not", "no", "never",
     "without", "fail to", "lack of", "will not", etc.).
   - "false" otherwise.
   - Example:
       "seeds will not be transported by the animals" -> outcome_is_negated = "true"
       "LESS sound being detected"                    -> outcome_is_negated = "false"

Very important rules:
- ONLY use the text of the QUESTION itself. Do NOT invent extra details.
- Do NOT answer the question (do NOT predict more/less/no effect). Only extract the fields.
- Always return a single JSON object with exactly these keys:
  "cause_event",
  "outcome_event",
  "outcome_base",
  "outcome_direction_in_question",
  "outcome_is_negated".
- For outcome_direction_in_question:
  - Use UPPERCASE tokens like "MORE", "LESS", "GREATER", "SMALLER", "HELPING", "HURTING".
  - If the direction is in a phrase (e.g. "a GREATER amount of ..."), capture the main direction word ("GREATER").
  - If no clear direction word appears, use "NONE".
- For outcome_is_negated:
  - Return the lowercase string "true" or "false" ONLY (no other values).

Output format:
Return ONLY a JSON object, with no extra text or explanation.

Question:
{self.question}"""
        # 调用 LLM 并做基础清洗
        response = self._call_llm(prompt)
        response = self._clean_response(response)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return None

        # 基础字段
        cause_event = (data.get("cause_event") or "").strip()
        outcome_event = (data.get("outcome_event") or "").strip()
        outcome_base = (data.get("outcome_base") or "").strip()
        direction = (data.get("outcome_direction_in_question") or "").strip().upper() or "NONE"

        # 解析并归一化显式否定标记
        raw_neg = str(data.get("outcome_is_negated", "false")).strip().lower()
        self.outcome_is_negated = (raw_neg == "true")

        # 保险起见：再做一层规则化，确保 outcome_base 去掉了 MORE/LESS 等方向词和显式否定前缀
        # 例如:
        #   QUESTION: "LESS sound being detected" -> outcome_base: "sound being detected"
        #   QUESTION: "seeds will not be transported by the animals"
        #       -> outcome_base 应类似 "seeds being transported by the animals"
        lowered = outcome_base.lower()
        for prefix in [
            "more ",
            "less ",
            "greater ",
            "smaller ",
            "a greater amount of ",
            "a smaller amount of ",
            "helping ",
            "hurting ",
            "no ",
            "not ",
            "never ",
            "without ",
            "lack of ",
        ]:
            if lowered.startswith(prefix):
                outcome_base = outcome_base[len(prefix) :].strip()
                break

        # ====== Question-level nodes (X_q, A_q, D_q) ======
        # 扰动事件：X
        self.cause_event = cause_event
        self.X.append(cause_event)

        # 结果基底及方向
        self.outcome_event = outcome_event
        self.outcome_base = outcome_base
        self.Y = outcome_base
        self.outcome_direction_in_question = direction

        # A_q / D_q 是基于 outcome_base 的 MORE / LESS 节点表示
        if outcome_base:
            self.A.append(f"MORE {outcome_base}")
            self.D.append(f"LESS {outcome_base}")


        # 返回原始抽取结果
        return {
            "cause_event": cause_event,
            "outcome_event": outcome_event,
            "outcome_base": outcome_base,
            "outcome_direction_in_question": direction,
            "outcome_is_negated": self.outcome_is_negated,
        }

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

        # 如果基础变量的符号本身就不确定，直接视为 no_effect
        if e not in {"more", "less", "no_effect"}:
            return "no_effect"
        if e == "no_effect":
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
        max_relations: int = 3,
    ) -> Dict[str, Any]:
        """
        使用 LLM 来拓展围绕 X 的一跳因果关系（local expansion）。

        返回:
            {
                "triples": List[Tuple[str, str, str]],  # (X, relation, new_entity)
                "new_entities": Set[str]               # 新出现的 tail 节点
            }
        """
        X = (X or "").strip()
        Y = (Y or "").strip() if Y is not None else None
        result: Dict[str, Any] = {"triples": [], "new_entities": set()}
        if not X:
            return result

        target_hint = Y if Y else "NONE"
        prompt = f"""
You are a causal edge finder.

Input:
- CAUSE_NODE (X): "{X}"
- TARGET_HINT (Y): "{target_hint}"

Task:
- Propose up to {max_relations} SINGLE-HOP causal effects that start directly from X.
- Each effect must be something that X itself immediately changes (one step away).
- Stay at the level of everyday science and commonsense mechanisms.
- Prefer short, neutral noun/verb phrases for effect nodes (no pronouns).
- Avoid baking "more/less" inside the node; encode direction via the relation sign.
- **Very important:** When choosing the effect node,
  * prefer nodes that reuse important nouns/phrases from TARGET_HINT
    (e.g. "control rods", "seeds", "lungs", "air sacs", etc.), or
  * nodes that are natural direct precursors/parts of TARGET_HINT.
- If there are several plausible effects, make sure that at least some of them
  clearly point toward the kind of process described in TARGET_HINT.

Signs:
- Use "RESULTS_IN" when X makes the effect node more likely/stronger/more frequent.
- Use "NOT_RESULTS_IN" when X makes the effect node less likely/weaker/less frequent.

If X has NO clear direct causal effects under normal conditions,
return an empty list of triples.

Output format:
Return ONLY one JSON object exactly in this shape:
{{
  "triples": [
    ["{X}", "RESULTS_IN" | "NOT_RESULTS_IN", "<direct effect node>"],
    ...
  ]
}}
""".strip()

        response = self._call_llm(prompt)
        response = self._clean_response(response)

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # 第一次兜底：尝试从响应中截取第一个 {...} 片段
            match = re.search(r'\{.*\}', response, re.DOTALL)
            candidate = match.group(0) if match else response

            # 尝试严格 JSON 解析
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                # 第二次兜底：使用 ast.literal_eval 解析"类 Python"字面量（支持单引号等）
                try:
                    data = ast.literal_eval(candidate)
                    if not isinstance(data, dict):
                        raise ValueError("literal_eval result is not a dict")
                except Exception:
                    # 第三次兜底：直接从文本中用正则提取三元组行  ["X", "RESULTS_IN", "Y"]
                    triple_matches = re.findall(
                        r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]',
                        response,
                        re.DOTALL,
                    )
                    if not triple_matches:
                        return result
                    data = {"triples": [list(t) for t in triple_matches]}

        raw_triples = data.get("triples", [])
        triples: List[Tuple[str, str, str]] = []
        new_entities: Set[str] = set()
        seen = set()

        if isinstance(raw_triples, list):
            for t in raw_triples:
                if not (isinstance(t, list) and len(t) == 3):
                    continue
                h, r, tail = t
                if str(h).strip() != X:
                    continue
                relation = str(r).strip().upper()
                if relation not in ("RESULTS_IN", "NOT_RESULTS_IN"):
                    continue
                tail_clean = str(tail).strip()
                if not tail_clean:
                    continue

                key = (X, relation, tail_clean.lower())
                if key in seen:
                    continue
                seen.add(key)

                triples.append((X, relation, tail_clean))
                new_entities.add(tail_clean)
                if len(triples) >= max_relations:
                    break

        result["triples"] = triples
        result["new_entities"] = new_entities
        self.last_causal_relations = result
        return result

    def expand_toward_target(
        self,
        start_X: str,
        target_Y: Optional[str] = None,
        max_depth: int = 2,
        max_relations_per_node: int = 3,
        max_nodes: int = 50,
    ) -> Dict[str, Any]:
        """
        循环展开：从 start_X 出发，朝着 target_Y 方向做多层一跳扩展，直到命中 target_Y 或达到深度/节点限制。

        返回:
            {
                "triples": List[Tuple[str, str, str]],  # 累积的所有一跳三元组
                "visited": Set[str],                    # 已访问/创建的节点
                "found_target": bool,                   # 是否在扩展中命中 target_Y
                "depth_reached": int,                   # 实际展开的层数
                "close_hits": List[Dict[str, Any]],     # 与 target 语义上 close 的节点记录
            }
        """
        start = (start_X or "").strip()
        target = (target_Y or "").strip() if target_Y else None

        if not start:
            return {
                "triples": [],
                "visited": set(),
                "found_target": False,
                "depth_reached": 0,
                "close_hits": [],
            }

        visited: Set[str] = set([start])
        triples_acc: List[Tuple[str, str, str]] = []
        frontier = [start]
        found = False
        depth = 0
        close_hits: List[Dict[str, Any]] = []

        while frontier and depth < max_depth and len(visited) < max_nodes:
            next_frontier: List[str] = []
            for node in frontier:
                rels = self.find_causal_relations(node, target, max_relations_per_node)
                for h, r, tail in rels.get("triples", []):
                    triples_acc.append((h, r, tail))
                    if tail not in visited and len(visited) < max_nodes:
                        visited.add(tail)
                        next_frontier.append(tail)
                    # 命中目标（精确匹配）
                    if target and tail.lower() == target.lower():
                        found = True
                        break
                    # 语义上是同一个变量：successful mailing vs. the letter being mailed successfully
                    if target:
                        relation = self.classify_variable_relation(tail, target, self.question)
                        if self._is_strong_close_relation(relation.relation_type):
                            # Only treat 'identical'/'part_of' as true close hits.
                            close_hits.append(
                                {
                                    "node": tail,
                                    "depth": depth + 1,
                                    "relation_type": relation.relation_type,
                                    "reasoning": relation.reasoning,
                                }
                            )
                if found or len(visited) >= max_nodes:
                    break
            if found or len(visited) >= max_nodes:
                depth += 1
                break
            frontier = next_frontier
            depth += 1

        return {
            "triples": triples_acc,
            "visited": visited,
            "found_target": found,
            "depth_reached": depth,
            "close_hits": close_hits,
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
            return is_valid
        except Exception:
            # Fail-open: if the judge fails, don't kill potentially valid paths.
            return True

    # Counterfactual test used by bridge_close_hits to detect 'substitution' edges.
    # We imagine removing the intermediate node C and see if the system
    # would need MORE of Y (or an alternative) to compensate; such edges
    # are treated as substitution rather than straightforward causal links.
    def _check_counterfactual_substitution(self, cause: str, effect: str, context: str) -> bool:
        """
        使用反事实逻辑检测是否为"替代/竞争"关系。
        返回 True 表示是替代关系（即原本被判为正向 RESULTS_IN 的关系，实际上是负向的）。
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
  "is_substitute": true/false,
  "reasoning": "..."
}}
""".strip()

        try:
            response = self._call_llm(prompt)
            response = self._clean_response(response)
            data = json.loads(response)
            is_sub = bool(data.get("is_substitute", False))
            return is_sub
        except Exception:
            return False

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
            return is_plausible
        except Exception:
            # Fail-open: if the sanity judge fails, don't over-filter.
            return True


    def bridge_close_hits(
        self,
        triples: List[Tuple[str, str, str]],
        close_hits: List[Dict[str, Any]],
        Y: str,
        max_bridge_nodes: int = 3,
    ) -> List[Tuple[str, str, str]]:
        if not close_hits:
            return triples
        Y = (Y or "").strip()
        if not Y:
            return triples

        new_triples = list(triples)
        sorted_hits = sorted(close_hits, key=lambda x: x.get("depth", 0))
        used = 0

        question_text = self.question
        context_start = triples[0][0] if triples else "The System"

        def _parse_bridge_json(raw: str) -> Optional[Dict[str, Any]]:
            cleaned = self._clean_response(raw or "")
            try:
                return json.loads(cleaned)
            except Exception:
                match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except Exception:
                        return None
            return None

        for hit in sorted_hits:
            if used >= max_bridge_nodes:
                break
            node = (hit.get("node") or "").strip()
            if not node:
                continue
            if any(h == node and t == Y for (h, _, t) in new_triples):
                continue

            prompt = f"""
You are a causal reasoning assistant.

Problem statement and question:
{question_text}


We consider a candidate close-hit variable C = "{node}" and the outcome variable Y = "{Y}".
Classify ONLY the direct relationship between C and Y using the JSON schema below.

Output STRICT JSON (no extra text):
{{
  "causal_type": "direct_step" | "shared_cause" | "multi_step" | "correlation_only",
  "direction": "C_increases_Y" | "C_decreases_Y" | "no_direct_effect",
  "is_local_to_question": true | false,
  "reasoning": "short natural language explanation"
}}

Definitions:
- "direct_step": C is essentially a single-step cause or component of Y in the causal mechanism.
- "shared_cause": C and Y mostly share an upstream cause; they do NOT directly cause each other.
- "multi_step": C may influence Y only via multiple intermediate steps (not 1-hop).
- "correlation_only": merely correlated/loosely associated with no robust causal direction.
- "C_increases_Y": more/stronger C tends to increase Y.
- "C_decreases_Y": more/stronger C tends to decrease Y.
- "no_direct_effect": no robust direct monotonic effect from C to Y.
- is_local_to_question = true if the direct C -> Y link is clearly implied by the question context or scientific knowledge; false if it relies mainly on weak speculation.
""".strip()

            try:
                raw = self._call_llm(prompt)
            except Exception:
                continue

            data = _parse_bridge_json(raw or "")
            if not data:
                continue

            causal_type = str(data.get("causal_type", "")).strip().lower()
            direction = str(data.get("direction", "")).strip()
            direction_norm = direction.lower()
            is_local_raw = data.get("is_local_to_question", False)
            is_local = bool(is_local_raw)
            if isinstance(is_local_raw, str):
                is_local = is_local_raw.strip().lower() == "true"

            if (
                causal_type != "direct_step"
                or not is_local
                or direction_norm not in {"c_increases_y", "c_decreases_y"}
            ):
                continue

            final_relation = "RESULTS_IN" if direction_norm == "c_increases_y" else "NOT_RESULTS_IN"
            new_triples.append((node, final_relation, Y))
            if hasattr(self, "bridge_edges"):
                self.bridge_edges.add((node, final_relation, Y))
            used += 1

        return new_triples


    def classify_variable_relation(self, a: str, b: str, context: str = "") -> VariableRelation:
        """
        Core comparison function returning a rich VariableRelation label.
        """
        a = (a or "").strip()
        b = (b or "").strip()
        if not a or not b:
            return VariableRelation("weakly_related", "Empty input.")

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
            relation = VariableRelation(relation_type, reasoning)
            return relation
        except Exception:
            return VariableRelation("weakly_related", "Parse failure or exception.")

    def _is_strong_close_relation(self, relation_type: str) -> bool:
        """
        Determine if a relation_type counts as a true close hit to the outcome.
        """
        return (relation_type or "").strip().lower() in {"identical", "part_of"}

    def is_same_variable(self, a: str, b: str, context: str = "") -> str:
        """
        Backward-compatible wrapper returning only the relation_type string.
        """
        relation = self.classify_variable_relation(a, b, context)
        return relation.relation_type

    def get_causal_chain(
        self,
        triples: List[Tuple[str, str, str]],
        start_X: str,
        target_Y: str,
    ) -> Dict[str, Any]:
        """
        仅基于给定的三元组列表，在图里从 start_X 链接到 target_Y，抽取显式因果链。

        参数:
            triples: 形如 (head, relation, tail) 的一跳因果三元组列表
            start_X: 起点节点
            target_Y: 终点节点（必须提供）

        返回:
            {
                "start": str,
                "target": str,
                "paths": List[List[Dict[str, str]]],  # 每条路径是若干 edge 字典
                "num_paths": int,                     # 找到的路径数量
                "shortest_path_length": Optional[int], # 最短路径的长度
                "all_nodes_in_paths": Set[str],       # 所有路径中出现的节点
            }
        """
        start = (start_X or "").strip()
        target = (target_Y or "").strip()

        if not start or not target:
            return {
                "start": start,
                "target": target,
                "paths": [],
                "num_paths": 0,
                "shortest_path_length": None,
                "all_nodes_in_paths": set(),
            }

        # 构建邻接表：head -> list[edge_dict]
        graph: Dict[str, List[Dict[str, str]]] = {}
        for h, r, t in triples:
            edge: Dict[str, str] = {"head": h, "relation": r, "tail": t}
            # 如果这条三元组是由 bridge_close_hits 创建，则标记其来源为 bridge
            if getattr(self, "bridge_edges", None) and (h, r, t) in self.bridge_edges:
                edge["source"] = "bridge"
            graph.setdefault(h, []).append(edge)

        paths: List[List[Dict[str, str]]] = []
        target_lower = target.lower()
        all_nodes: Set[str] = set()

        def dfs(node: str, path_edges: List[Dict[str, str]]) -> None:
            if node.lower() == target_lower and path_edges:
                paths.append(list(path_edges))
                # 收集路径中的所有节点
                for edge in path_edges:
                    all_nodes.add(edge["head"])
                    all_nodes.add(edge["tail"])
                return

            for edge in graph.get(node, []):
                nxt = edge["tail"]
                # 避免在同一路径中形成环
                if any(e["tail"].lower() == nxt.lower() for e in path_edges):
                    continue
                path_edges.append(edge)
                dfs(nxt, path_edges)
                path_edges.pop()

        dfs(start, [])

        # 计算最短路径长度
        shortest_length = None
        if paths:
            shortest_length = min(len(p) for p in paths)

        return {
            "start": start,
            "target": target,
            "paths": paths,
            "num_paths": len(paths),
            "shortest_path_length": shortest_length,
            "all_nodes_in_paths": all_nodes,
        }
    
    def show_causal_graph(self):
        """
        Display the constructed causal graph.
        """
        pass

    def _edge_relation_to_sign(self, relation: str) -> int:
        """
        Map an edge relation label to a causal sign:
        +1: increasing / promoting / enabling
        -1: decreasing / preventing / inhibiting
         0: neutral / unknown

        This mapping is intentionally generic so it can be reused across
        datasets, as long as relation strings follow a causal convention.
        """
        if not relation:
            return 0

        r = str(relation).strip().upper().replace(" ", "_")

        # Primary mapping for known relation names.
        POSITIVE = {
            "RESULTS_IN",
            "CAUSES",
            "LEADS_TO",
            "INCREASES",
            "PROMOTES",
            "ALLOWS",
            "ENABLES",
            "PRODUCES",
            "CREATES",
        }
        NEGATIVE = {
            "NOT_RESULTS_IN",
            "PREVENTS",
            "REDUCES",
            "DECREASES",
            "INHIBITS",
            "BLOCKS",
            "STOPS",
            "LOWERS",
        }

        if r in POSITIVE:
            return 1
        if r in NEGATIVE:
            return -1

        # Fallback: use keyword patterns inside the relation string.
        # This allows some robustness to slightly different labels.
        tokens = r.split("_")
        if any(t in ("INCREASE", "INCREASES", "MORE", "HIGHER", "RAISES") for t in tokens):
            return 1
        if any(t in ("DECREASE", "DECREASES", "LESS", "LOWER", "REDUCE", "REDUCES", "REDUCED") for t in tokens):
            return -1
        if any(t in ("PREVENTS", "PREVENT", "INHIBITS", "BLOCKS", "STOPS") for t in tokens):
            return -1

        # Unknown / neutral.
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

        # Collect non-zero signs.
        signs: List[int] = []
        for edge in path:
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
            return is_consistent
        except Exception:
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

        # 1) filter by length
        filtered = [p for p in paths if len(p) > 0 and len(p) <= max_len]
        if not filtered:
            # if all paths are too long, fall back to the original ones (shortest few)
            paths_sorted = sorted(paths, key=lambda p: len(p))
            return paths_sorted[:max_paths]

        # 2) sort by length ascending
        filtered = sorted(filtered, key=lambda p: len(p))

        # 3) deduplicate by edge sequence
        seen = set()
        unique_paths: List[List[Dict[str, Any]]] = []
        for p in filtered:
            key = tuple((e.get("head", ""), e.get("relation", ""), e.get("tail", "")) for e in p)
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
        - net_sign: +1 / -1 / 0 (sum of edge.sign)
        - edges: list of {head, tail, relation, sign, source}
        """
        payload: List[Dict[str, Any]] = []
        for idx, path in enumerate(paths, start=1):
            edges_info: List[Dict[str, Any]] = []
            sign_sum = 0
            for e in path:
                # Prefer an existing sign field; otherwise infer from relation.
                s = e.get("sign")
                if s is None:
                    s = self._edge_relation_to_sign(e.get("relation", ""))
                try:
                    sign = int(s)
                except Exception:
                    sign = 0
                sign_sum += sign
                edges_info.append({
                    "head": e.get("head", ""),
                    "tail": e.get("tail", ""),
                    "relation": e.get("relation", ""),
                    "sign": sign,
                    "source": e.get("source", "bfs"),
                })

            net_sign = 0
            if sign_sum > 0:
                net_sign = 1
            elif sign_sum < 0:
                net_sign = -1

            payload.append({
                "id": idx,
                "length": len(path),
                "net_sign": net_sign,
                "edges": edges_info,
            })
        return payload

    def causal_chain_to_text(
        self,
        chain_result: Dict[str, Any],
        bfs_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        将因果链/三元组集合转化为“结构化的描述文字”。

        修改：跳过 LLM 润色，直接返回包含
        [INCREASES / PROMOTES] / [DECREASES / SUPPRESSES] 等关键词的结构化文本，
        以防止 LLM 在摘要过程中“幻觉”出错误的因果方向。

        参数:
            chain_result: get_causal_chain 的返回结果
            bfs_result: expand_toward_target 的返回结果（可选）

        返回:
            描述性文字
        """
        start = chain_result.get("start", "")
        target = chain_result.get("target", "")
        num_paths = chain_result.get("num_paths", 0)

        raw_parts: List[str] = []

        if num_paths > 0:
            paths = chain_result["paths"]

            raw_parts.append(
                f"From '{start}' to '{target}', the system found {num_paths} causal path(s)."
            )

            # 可以稍微多展示几条，比如前 5 条路径
            for i, path in enumerate(paths[:5], 1):
                steps = []
                for edge in path:
                    r = edge["relation"]
                    h = edge["head"]
                    t = edge["tail"]
                    if r == "RESULTS_IN":
                        # 保持这些大写的关键词，reason_with_description 的 Prompt 会专门识别它们
                        rel_text = "INCREASES / PROMOTES"
                    else:  # NOT_RESULTS_IN
                        rel_text = "DECREASES / SUPPRESSES"
                    steps.append(f"({h}) -> [{rel_text}] -> ({t})")
                raw_parts.append(f"Path {i}: " + " ; ".join(steps))

            # 统计信息：多少正向/负向边
            positive_count = sum(1 for p in paths for e in p if e["relation"] == "RESULTS_IN")
            negative_count = sum(1 for p in paths for e in p if e["relation"] != "RESULTS_IN")
            raw_parts.append(
                f"Statistical Summary: {positive_count} positive edges, {negative_count} negative edges."
            )

        else:
            # 无路径时的 fallback 描述
            if bfs_result and bfs_result.get("triples"):
                triples = bfs_result["triples"]
                raw_parts.append(
                    f"No complete causal path from '{start}' to '{target}' was found."
                )
                raw_parts.append("Observed 1-hop relations:")
                for i, (h, r, t) in enumerate(triples[:10], 1):
                    relation_text = "INCREASES" if r == "RESULTS_IN" else "DECREASES"
                    raw_parts.append(f"Edge {i}: {h} --[{relation_text}]--> {t}")
            else:
                raw_parts.append("No causal relations found.")

        # 关键修改：直接返回结构化文本，不再调用 LLM 润色
        return "\n".join(raw_parts)

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
    ) -> Dict[str, Any]:
        """
        Final decision by LLM, using the extracted causal paths as 'reasoning scratchpad'.

        New design:
          - We do NOT hard-code 'long chains => no_effect' in Python.
          - Instead we compute path statistics and show them to the LLM.
          - The LLM must evaluate three hypotheses:
              H_more: cause increases outcome_base
              H_less: cause decreases outcome_base
              H_no_effect: cause has no clear effect on outcome_base
            and assign a plausibility score to each.
          - The LLM must also perform a counterfactual test:
              "If the cause is removed/prevented, does the base outcome
               tend to go up, down, or stay essentially unchanged?"
        """

        # 1) Convert paths to human-readable text
        path_blocks: List[str] = []
        for p in paths or []:
            edges = p.get("edges", [])
            edges_txt_parts: List[str] = []
            for e in edges:
                sign = e.get("sign", 0)
                if sign > 0:
                    tag = "PROMOTES(+)"
                elif sign < 0:
                    tag = "SUPPRESSES(-)"
                else:
                    tag = "NEUTRAL(0)"
                edges_txt_parts.append(
                    f"({e.get('head', '')}) -[{tag}]-> ({e.get('tail', '')})"
                )
            edges_txt = " ; ".join(edges_txt_parts) if edges_txt_parts else "(empty path)"
            path_blocks.append(
                f"Path {p.get('id')} "
                f"(length={p.get('length')}, net_sign={p.get('net_sign')}): "
                f"{edges_txt}"
            )

        paths_text = "\n".join(path_blocks) if path_blocks else "(no paths)"

        # 2) Compute simple path statistics for the LLM
        num_paths = len(paths) if paths else 0
        lengths = [int(p.get("length", 0) or 0) for p in paths] if paths else []
        shortest_len = min(lengths) if lengths else None
        longest_len = max(lengths) if lengths else None
        num_pos = sum(1 for p in (paths or []) if int(p.get("net_sign", 0) or 0) > 0)
        num_neg = sum(1 for p in (paths or []) if int(p.get("net_sign", 0) or 0) < 0)
        num_neu = num_paths - num_pos - num_neg

        # "long_paths_only" means there is evidence but all chains are long (soft Occam).
        long_paths_only = bool(shortest_len is not None and shortest_len > 4)

        # 3) Construct the LLM prompt
        prompt = f"""
You are a causal reasoning assistant.

You are given:
- A WIQA-style question about how changing a cause event affects an outcome.
- A set of causal paths from the cause event to the *base* outcome variable
  (without any MORE/LESS wording).
- Simple statistics about these paths.

[Question]
{question}

[Events]
Cause event: "{cause_event}"
Outcome in the question: "{outcome_event}"
Base outcome variable (without MORE/LESS modifiers): "{outcome_base}"

[Causal paths]
Each path connects the cause event to the base outcome variable.
Each edge is annotated as PROMOTES(+) or SUPPRESSES(-).

{paths_text}

[Path statistics]
- num_paths: {num_paths}
- shortest_length: {shortest_len}
- longest_length: {longest_len}
- positive_paths (net_sign > 0): {num_pos}
- negative_paths (net_sign < 0): {num_neg}
- neutral_paths (net_sign == 0): {num_neu}
- long_paths_only (all paths longer than 4 edges?): {str(long_paths_only).lower()}

Your task:

1. Reason ONLY about the base outcome variable "{outcome_base}".
2. Consider three hypotheses about the effect of the cause on this base variable:
   - H_more: the cause tends to INCREASE "{outcome_base}" overall.
   - H_less: the cause tends to DECREASE "{outcome_base}" overall.
   - H_no_effect: the cause has NO CLEAR EFFECT on "{outcome_base}".
3. Use the causal paths AND the path statistics as evidence:
   - Shorter paths are usually stronger evidence than extremely long multi-step paths.
   - If *all* paths are very long (for example shortest_length > 4),
     treat "no_effect" as a serious candidate unless the paths are extremely consistent.
   - If paths strongly disagree in sign (many positive AND many negative),
     this is evidence for "no_effect" or "unknown".
4. Use a COUNTERFACTUAL test:
   - Imagine the cause event is completely removed or prevented.
   - Would this make the base outcome larger, smaller, or essentially unchanged?
   - Use this to help decide between MORE, LESS, and NO_EFFECT.
5. For each hypothesis H_more, H_less, H_no_effect, assign a PLAUSIBILITY SCORE
   between 0 and 1. These scores MUST sum to 1.0 (within rounding).
6. Then:
   - Choose "effect_on_base" as one of "more", "less", "no_effect", or "unknown".
     Normally this should be the hypothesis with the highest score, unless the
     evidence is so weak that "unknown" is more honest.
   - Set "confidence" to "high", "medium", or "low" based on how strong and
     consistent the evidence is.
   - Briefly explain your reasoning, referring to the most important paths or
     statistics.

Output format (VERY IMPORTANT):

Return ONLY a single JSON object with EXACTLY this structure and NO extra text:

{{
  "effect_on_base": "more" | "less" | "no_effect" | "unknown",
  "confidence": "high" | "medium" | "low",
  "reasoning": "short natural language explanation",
  "scores": {{
    "more": <float between 0 and 1>,
    "less": <float between 0 and 1>,
    "no_effect": <float between 0 and 1>
  }},
  "paths_eval": [
    {{
      "path_id": <int>,
      "plausible": true | false,
      "direction": "more" | "less" | "no_effect" | "unknown",
      "comment": "short comment about this path"
    }}
  ]
}}
""".strip()

        # 4) Call LLM and parse JSON robustly
        raw = self._call_llm(prompt)
        raw = self._clean_response(raw or "")

        data: Dict[str, Any]

        try:
            json_text = raw.strip()
            data = json.loads(json_text)
        except Exception:
            # Attempt to extract the first JSON object from the raw text.
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                try:
                    json_text = m.group(0)
                    data = json.loads(json_text)
                except Exception:
                    data = self._fallback_parse_llm_output(raw)
            else:
                data = self._fallback_parse_llm_output(raw)

        # 5) Normalize effect_on_base and handle scores-based fallback
        effect = (data.get("effect_on_base") or "unknown")
        if isinstance(effect, str):
            effect_norm = effect.strip().lower()
        else:
            effect_norm = "unknown"

        scores = data.get("scores")
        if not isinstance(scores, dict):
            scores = {}
        data["scores"] = scores  # ensure field exists

        allowed_labels = ["more", "less", "no_effect"]

        if effect_norm not in allowed_labels:
            # If LLM returned "unknown" or something weird, fall back to scores
            best_label = None
            best_score = -1.0
            for lbl in allowed_labels:
                try:
                    sc = float(scores.get(lbl, 0.0))
                except Exception:
                    sc = 0.0
                if sc > best_score:
                    best_label = lbl
                    best_score = sc
            if best_label is not None and best_score > 0.0:
                effect_norm = best_label
            else:
                # absolute fallback
                effect_norm = "no_effect"

        data["effect_on_base"] = effect_norm

        return data

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

        if not chain_result:
            return self._return_no_effect("No causal chain result provided.")

        # Extract paths and start node from chain_result
        paths: List[List[Dict[str, Any]]] = []
        start_node = ""
        if chain_result and isinstance(chain_result, dict):
            paths = chain_result.get("paths", []) or []
            start_node = chain_result.get("start", "") or ""

        # If no causal paths at all, default to no_effect
        if not paths:
            return self._return_no_effect("No causal paths found between cause and outcome_base.")

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
        )

        effect_on_base = (llm_result.get("effect_on_base") or "unknown").strip().lower()
        if effect_on_base not in {"more", "less", "no_effect"}:
            effect_on_base = "unknown"
        reasoning = llm_result.get("reasoning", "")
        confidence = llm_result.get("confidence", "unknown")

        # Map effect_on_base + question direction -> predicted textual answer
        predicted_effect = self.map_effect_on_base_to_wiqa_label(effect_on_base)
        predicted_choice = self._effect_to_choice(predicted_effect, choices)

        return {
            "predicted_answer": predicted_effect,
            "predicted_choice": predicted_choice,
            "effect_on_base": effect_on_base,
            "reasoning": reasoning,
            "confidence": confidence,
            "debug_paths_used": paths_payload,
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


    def run_wiqa_pipeline(self):
        """
        Run the full WIQACausalBuilder pipeline for one WIQA datapoint.

        This is the programmatic version of the steps executed in ``main()``,
        so it can be imported and called from other scripts.

        Parameters
        ----------
        datapoint : dict
            A WIQA example with at least the keys
            ``'question_stem'``, ``'answer_label'``, ``'answer_label_as_choice'``
            and ``'choices'``.

        Returns
        -------
        dict
            A dictionary containing intermediate and final results:
            - 'info'             : output of extract_start_entity()
            - 'bfs'              : output of expand_toward_target()
            - 'chain_result'     : output of get_causal_chain()
            - 'description'      : string from causal_chain_to_text()
            - 'reasoning_result' : output of reason_with_description()
        """
        # STEP 1: extract start / target entities
        info = self.extract_start_entity()
        start = info["cause_event"]
        target = info["outcome_base"]

        # STEP 2: expand toward target with BFS
        bfs = self.expand_toward_target(
            start_X=start,
            target_Y=target,
            max_depth=5,
            max_relations_per_node=5,
        )

        # STEP 3: optionally insert bridges via close hits
        if bfs["close_hits"]:
            triples_with_bridges = self.bridge_close_hits(
                triples=bfs["triples"],
                close_hits=bfs["close_hits"],
                Y=target,
                max_bridge_nodes=3,
            )
        else:
            triples_with_bridges = bfs["triples"]

        # STEP 4: search for causal chain
        chain_result = self.get_causal_chain(
            triples_with_bridges,
            start_X=start,
            target_Y=target,
        )

        # STEP 5: verbalize chain and reason with LLM
        description = self.causal_chain_to_text(chain_result, bfs)
        reasoning_result = self.reason_with_description(
            description,
            chain_result=chain_result,
        )

        return {
            "info": info,
            "bfs": bfs,
            "chain_result": chain_result,
            "description": description,
            "reasoning_result": reasoning_result,
        }
        


def main():
    # 示例数据点
    datapoint = {
        'question_stem': 'suppose during summer happens, how will it affect more water molecules less spacing.',
        'answer_label': 'less',
        'answer_label_as_choice': 'B',
        'choices': {'text': ['more', 'less', 'no_effect'], 'label': ['A', 'B', 'C']},
    }

    wiqa = WIQACausalBuilder(datapoint)
    result = wiqa.run_wiqa_pipeline()

    # 返回预测结果
    return result['reasoning_result']

if __name__ == "__main__":
    result = main()
    if result:
        print(f"Predicted: {result['predicted_answer']} (choice: {result['predicted_choice']})")
        print(f"Confidence: {result['confidence']}")
