# WIQACausalBuilder 完整流程分析

## 整体流程概览

```
Question → 步骤1: 提取起止点 → 步骤2: BFS扩展因果图 → 步骤3: 桥接+提取链 → 步骤4: 文字描述 → 步骤5: LLM推理 → 最终答案
```

---

## 详细流程说明

### 📥 **输入数据结构**
```python
datapoint = {
    'question_stem': '问题文本',  # 例如: "suppose X happens, how will it affect LESS Y"
    'question_para_step': [...],  # 背景段落（过程描述）
    'answer_label': 'more/less/no_effect',
    'choices': {'text': [...], 'label': [...]},
    ...
}
```

---

## 步骤 1: 提取起点和终点 (extract_start_entity)

### 📍 **位置**: 86-248行

### 🎯 **目的**
从问题中抽取因果推理的关键要素：
- **起点**: cause_event (扰动事件 X)
- **终点**: outcome_base (结果基底变量 Y)
- **方向**: outcome_direction_in_question (MORE/LESS/NONE)
- **否定**: outcome_is_negated (是否包含否定词)

### 📝 **提示词结构**
```
You are an information-extraction assistant for scientific causal questions.

Given ONE question, extract TWO layers:
LAYER 1: OUTCOME_EVENT (完整表达，含MORE/LESS/否定词)
LAYER 2: OUTCOME_BASE (去掉方向词和否定词的基础变量)

Fields to extract:
1. cause_event: 原因事件
2. outcome_event: 完整结果表达
3. outcome_base: 基础结果变量
4. outcome_direction_in_question: MORE/LESS/NONE
5. outcome_is_negated: true/false

Question: {self.question}
```

### 🔄 **输入/输出**
- **输入**: `self.question` (问题文本)
- **输出**:
```python
{
    "cause_event": "no sunlight for the tree to grow",
    "outcome_event": "LESS rain",
    "outcome_base": "rain",
    "outcome_direction_in_question": "LESS",
    "outcome_is_negated": False
}
```

### 💾 **缓存到对象**
```python
self.cause_event = cause_event
self.X.append(cause_event)  # 扰动节点
self.Y = outcome_base
self.A.append(f"MORE {outcome_base}")
self.D.append(f"LESS {outcome_base}")
```

---

## 步骤 2: BFS 因果图扩展 (expand_toward_target)

### 📍 **位置**: 484-574行

### 🎯 **目的**
从 start_X 出发，通过多层一跳扩展（BFS），朝着 target_Y 构建因果图，直到找到目标或达到深度限制。

### 🔁 **循环调用**: find_causal_relations (355-482行)

#### **子函数: find_causal_relations**

##### 📝 **提示词结构**
```
You are a causal edge finder.

Input:
- CAUSE_NODE (X): "{X}"
- TARGET_HINT (Y): "{target_hint}"
- PARAGRAPH: "{paragraph}"

Task:
- Propose up to {max_relations} SINGLE-HOP causal effects from X
- Each effect must be direct (one step away)
- Prefer nodes that reuse key nouns from TARGET_HINT or PARAGRAPH

Signs:
- "RESULTS_IN": X makes effect more likely/stronger
- "NOT_RESULTS_IN": X makes effect less likely/weaker

Output format:
{
  "triples": [
    ["{X}", "RESULTS_IN" | "NOT_RESULTS_IN", "<effect_node>"],
    ...
  ]
}
```

##### 🔄 **输入/输出**
- **输入**:
  - `X`: 当前节点
  - `Y`: 目标节点（提示方向）
  - `max_relations`: 最多返回几条边
- **输出**:
```python
{
    "triples": [
        ("no sunlight", "RESULTS_IN", "tree cannot photosynthesize"),
        ("no sunlight", "NOT_RESULTS_IN", "tree growth"),
        ...
    ],
    "new_entities": {"tree cannot photosynthesize", "tree growth", ...}
}
```

### 🔍 **BFS 扩展过程**
```python
frontier = [start_X]
depth = 0

while frontier and depth < max_depth:
    for node in frontier:
        rels = find_causal_relations(node, target_Y, max_relations_per_node)
        for (h, r, tail) in rels["triples"]:
            triples_acc.append((h, r, tail))

            # 精确匹配目标
            if tail.lower() == target.lower():
                found = True
                break

            # 语义匹配 (调用 is_same_variable)
            label = is_same_variable(tail, target, question)
            if label == "same":
                triples_acc.append((tail, "RESULTS_IN", target))
                found = True
            elif label == "opposite":
                triples_acc.append((tail, "NOT_RESULTS_IN", target))
                found = True
            elif label == "close":
                close_hits.append({"node": tail, "depth": depth+1})

            if tail not in visited:
                visited.add(tail)
                next_frontier.append(tail)

    frontier = next_frontier
    depth += 1
```

#### **子函数: is_same_variable** (914-978行)

##### 📝 **提示词结构**
```
You are a scientific concept-matching assistant.

Classify relationship into:
- "same": 同一个变量，方向一致
- "opposite": 同一个变量，方向相反 (e.g., success vs failure)
- "close": 强相关但非同一变量 (part-of, subtype)
- "different": 明显不同

A = "{a}"
B = "{b}"

Output: {"label": "same" | "opposite" | "close" | "different"}
```

### 🔄 **输入/输出**
- **输入**:
  - `start_X`: "no sunlight for the tree to grow"
  - `target_Y`: "rain"
  - `max_depth`: 5
  - `max_relations_per_node`: 5
- **输出**:
```python
{
    "triples": [
        ("no sunlight", "NOT_RESULTS_IN", "tree growth"),
        ("tree growth", "NOT_RESULTS_IN", "transpiration"),
        ...
    ],
    "visited": {"no sunlight", "tree growth", "transpiration", ...},
    "found_target": False,
    "depth_reached": 5,
    "close_hits": [
        {"node": "water vapor", "depth": 3},
        {"node": "evaporation", "depth": 2}
    ]
}
```

---

## 步骤 3: 桥接 + 提取因果链

### 3A: bridge_close_hits (793-912行)

#### 🎯 **目的**
对于 BFS 中发现的 "close" 节点，用 LLM 判断它们与目标 Y 之间是否存在直接因果关系。

#### 📝 **提示词结构**
```
You are a causal reasoning assistant.

Context:
- System driven by "{context_start}"
- Candidate variable V = "{node}"
- Target variable Y = "{Y}"

Decide: Does increasing V directly increase/decrease/not affect Y?

Logic Guide:
1. Fuel Rule: V is raw material/upstream cause → more V = MORE Y → "RESULTS_IN"
2. Brake Rule: Y is inhibitor of V → more V = LESS Y → "NOT_RESULTS_IN"

Output: {"relation": "RESULTS_IN" | "NOT_RESULTS_IN" | "NONE", "reasoning": "..."}
```

#### 🛡️ **两道防线**

##### **防线1: _check_causal_relevance** (576-651行)
```
You are a Scientific Logic Judge.

Does knowing Cause helps predict Effect within this paragraph context?

Criteria for ACCEPTANCE:
1. State Exclusion (Strong Negative)
2. Mechanism
3. Indirect Dependency / Necessary Resource

Output: {"is_valid_link": true/false, "reasoning": "..."}
```

##### **防线2: _check_counterfactual_substitution** (653-706行)
```
Distinguish SUBSTITUTION vs. DEPENDENCY.

TEST: If A is REMOVED, do we need MORE B to compensate?

Type 1: SUBSTITUTION (Spare Tire Logic)
  - "No Pipes -> Need MORE Trucks"
  - Verdict: TRUE (Flip to Negative)

Type 2: DEPENDENCY (Fuel Logic)
  - "No Soil -> No Germination"
  - Verdict: FALSE (Keep Positive)

Output: {"is_substitute": true/false, "reasoning": "..."}
```

#### 🔄 **输入/输出**
- **输入**:
  - `triples`: BFS 得到的三元组
  - `close_hits`: [{"node": "water vapor", "depth": 3}, ...]
  - `Y`: "rain"
- **输出**: 扩展后的三元组列表，可能新增桥接边：
```python
[
    ...原有三元组...,
    ("water vapor", "RESULTS_IN", "rain"),  # 新增桥接边
    ...
]
```

### 3B: get_causal_chain (980-1063行)

#### 🎯 **目的**
从给定的三元组列表中，用 DFS 提取从 start_X 到 target_Y 的所有路径。

#### 🔄 **输入/输出**
- **输入**:
  - `triples`: [(h, r, t), ...]
  - `start_X`: "no sunlight"
  - `target_Y`: "rain"
- **输出**:
```python
{
    "start": "no sunlight",
    "target": "rain",
    "paths": [
        [
            {"head": "no sunlight", "relation": "NOT_RESULTS_IN", "tail": "tree growth"},
            {"head": "tree growth", "relation": "NOT_RESULTS_IN", "tail": "transpiration"},
            {"head": "transpiration", "relation": "RESULTS_IN", "tail": "water vapor"},
            {"head": "water vapor", "relation": "RESULTS_IN", "tail": "rain"}
        ],
        ...
    ],
    "num_paths": 3,
    "shortest_path_length": 4,
    "all_nodes_in_paths": {...}
}
```

---

## 步骤 4: 生成文字描述 (causal_chain_to_text)

### 📍 **位置**: 1332-1401行

### 🎯 **目的**
将结构化的因果路径转换为带符号标记的文字描述（不经过 LLM 润色，避免幻觉）。

### 🔄 **输入/输出**
- **输入**: `chain_result` (步骤3B的输出)
- **输出**:
```text
From 'no sunlight' to 'rain', the system found 3 causal path(s).
Path 1: (no sunlight) -> [DECREASES / SUPPRESSES] -> (tree growth) ; (tree growth) -> [DECREASES / SUPPRESSES] -> (transpiration) ; ...
Path 2: ...
Statistical Summary: 2 positive edges, 4 negative edges.
```

---

## 步骤 5: LLM 推理 (reason_with_description)

### 📍 **位置**: 1651-1731行

### 🎯 **目的**
基于提取的因果路径，让 LLM 最终决定 cause 对 **outcome_base** 的影响。

### 🔁 **核心调用**: _final_llm_decision (1444-1578行)

#### 📝 **提示词结构**
```
You are a causal reasoning assistant.

[Question]
{question}

[Paragraph]
{paragraph_steps}

[Base outcome variable]
"{outcome_base}"

[Candidate causal paths]
Each path connects "{cause_event}" to "{outcome_base}".
Edges annotated as PROMOTES(+) or SUPPRESSES(-).

Path 1 (length=4): (no sunlight) -[SUPPRESSES(-)]-> (tree growth) ; ...
Path 2 (length=3): ...

[Your job]
1. Look ONLY at base outcome: "{outcome_base}"
2. Decide net effect: more/less/no_effect/unknown
3. IGNORE question phrasing like "MORE X" or "LESS X"

Output ONLY JSON:
{
  "effect_on_base": "more" | "less" | "no_effect" | "unknown",
  "confidence": "high" | "medium" | "low",
  "reasoning": "...",
  "paths_eval": [
    {"path_id": 1, "plausible": true, "direction": "less", "comment": "..."},
    ...
  ]
}
```

#### 🔄 **输入/输出**
- **输入**:
  - `question`: 原问题
  - `paragraph`: 背景段落
  - `cause_event`: "no sunlight"
  - `outcome_base`: "rain"
  - `paths`: 结构化路径列表
- **输出**:
```python
{
    "effect_on_base": "less",  # 对基础变量的影响
    "confidence": "medium",
    "reasoning": "No sunlight -> less tree growth -> less transpiration -> less water vapor -> less rain",
    "paths_eval": [...]
}
```

### 🔀 **映射逻辑**: map_effect_on_base_to_wiqa_label (250-353行)

将 **effect_on_base** + **outcome_direction_in_question** 映射到最终答案：

```python
# 例子1: 题目问 "affect LESS rain"
outcome_direction_in_question = "LESS"
effect_on_base = "less"  # 基础变量 rain 减少

# LESS 方向逻辑：基础变量减少 → "LESS rain" 事件更常发生 → 答案 "more"
predicted_answer = "more"

# 例子2: 题目问 "affect rain"
outcome_direction_in_question = "NONE"
effect_on_base = "less"

# 无方向修饰：直接返回基础效果
predicted_answer = "less"
```

### 📤 **最终输出**
```python
{
    "predicted_answer": "more",  # 最终 WIQA 答案
    "predicted_choice": "A",     # 对应选项
    "effect_on_base": "less",    # 对基础变量的影响
    "reasoning": "...",
    "confidence": "medium",
    "debug_paths_used": [...]
}
```

---

## 🔍 辅助检查函数（在 BFS/桥接中使用）

### 1. _check_path_consistency (1171-1234行)
**作用**: 防止路径推导过程中违反前提（例如起点是"无云"，中间却出现"云形成"）

**提示词关键**:
```
Check if Intermediate Steps CONTRADICT the Start Event.

Critical Rules:
1. No "Undoing" the Premise
2. Immediate vs. Long-term (避免循环论证)

Output: {"is_consistent": true/false, "reasoning": "..."}
```

### 2. _verify_chain_plausibility (708-791行)
**作用**: 路径整体合理性检查（防止语义漂移、荒谬联系）

**提示词关键**:
```
CRITERIA FOR APPROVAL:
1. Implicit Steps are OK (允许跳过明显中间步骤)
2. General Causality
3. Negative Logic

CRITERIA FOR REJECTION:
1. Semantic Drift (概念中途变义)
2. Magical/Absurd Links
3. Extreme Butterfly Effect

Output: {"is_plausible": true/false, "reasoning": "..."}
```

---

## 📊 完整流程数据流示意

```
输入问题: "suppose no sunlight happens, how will it affect LESS rain"
    ↓
步骤1 (extract_start_entity):
    cause_event = "no sunlight"
    outcome_base = "rain"
    outcome_direction = "LESS"
    ↓
步骤2 (expand_toward_target):
    BFS 从 "no sunlight" 扩展
    → 发现 50 个节点，100 条边
    → close_hits = ["water vapor", "evaporation"]
    ↓
步骤3A (bridge_close_hits):
    检查 "water vapor" → "rain": RESULTS_IN ✓
    检查 "evaporation" → "rain": NOT_RESULTS_IN ✗ (被过滤)
    → 新增 1 条桥接边
    ↓
步骤3B (get_causal_chain):
    DFS 提取路径
    → 找到 3 条完整路径从 "no sunlight" 到 "rain"
    ↓
步骤4 (causal_chain_to_text):
    生成结构化描述文本
    → "Path 1: (no sunlight) -[SUPPRESSES]-> ... -[PROMOTES]-> (rain)"
    ↓
步骤5 (_final_llm_decision):
    LLM 分析路径 → effect_on_base = "less"
    ↓
映射 (map_effect_on_base_to_wiqa_label):
    outcome_direction = "LESS"
    effect_on_base = "less"
    → "LESS rain" 事件更常发生
    → predicted_answer = "more"
    ↓
输出: {"predicted_answer": "more", "predicted_choice": "A"}
```

---

## 🎯 关键设计思想

1. **两层抽象**:
   - `outcome_base`: 去掉方向和否定的基础变量
   - `outcome_event`: 原问题中的完整表达
   - LLM 只需判断对基础变量的影响，由代码映射到最终答案

2. **多道防线**:
   - Relevance Filter (相关性)
   - Substitution Filter (替代关系检测)
   - Consistency Check (前提一致性)
   - Plausibility Check (路径合理性)

3. **避免 LLM 幻觉**:
   - 结构化路径不经润色，直接传递符号
   - 使用 PROMOTES(+) / SUPPRESSES(-) 等显式标记
   - 最终 LLM 只做聚合决策，不做因果发现

4. **温度=0 确定性**:
   - 所有 LLM 调用强制 `temperature=0`, `seed=42`
   - 保证可复现性
