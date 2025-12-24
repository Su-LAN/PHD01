# WIQACausalBuilder 完整流程分析

## 概述

`WIQACausalBuilder` 是一个用于处理 WIQA 数据集的因果推理系统。它通过构建因果图并使用 LLM 进行推理来回答"如果 X 发生，会如何影响 Y"这类问题。

## 整体架构

```
输入: WIQA 问题 (cause → outcome?)
  ↓
步骤1: 提取起点和终点
  ↓
步骤2: BFS 扩展因果图
  ↓
步骤3: 提取因果链
  ↓
步骤4: 转化为文字描述
  ↓
步骤5: LLM 推理
  ↓
输出: more/less/no_effect + 推理过程
```

---

## 详细流程分解

### 步骤 1: 初始化和信息提取

#### 1.1 初始化 (`__init__`)
```python
wiqa = WIQACausalBuilder(datapoint, model_name="gemma2:27b")
```

**目的**: 创建因果构建器实例

**输入**:
- `datapoint`: WIQA 数据点，包含：
  - `question_stem`: 问题文本
  - `answer_label`: 正确答案 (more/less/no_effect)
  - `choices`: 选项
  - `question_para_step`: 过程步骤（可选）

**初始化的状态**:
- `self.question`: 存储问题
- `self.model_name`: LLM 模型名称
- `self.X, self.Y, self.Z, etc.`: 节点集合（预留用于图构建）

#### 1.2 提取起点和终点 (`extract_start_entity`)
```python
info = wiqa.extract_start_entity()
start = info["cause_event"]          # 起点：原因事件
target = info["outcome_base"]        # 终点：结果基底
direction = info["outcome_direction_in_question"]  # 方向词
```

**目的**: 从问题中提取因果推理的起点和终点

**LLM Prompt 要求**:
- 提取 cause_event: 问题中的扰动/原因事件
- 提取 outcome_event: 完整的结果表达
- 提取 outcome_base: 去掉方向词后的结果基底
- 提取 outcome_direction_in_question: MORE/LESS/HURTING 等方向词

**示例**:
```
问题: "suppose less oxygen is inhaled happens, how will it affect MORE oxygen enters the blood."

提取结果:
{
  "cause_event": "less oxygen is inhaled",
  "outcome_event": "MORE oxygen enters the blood",
  "outcome_base": "oxygen enters the blood",
  "outcome_direction_in_question": "MORE"
}
```

**副作用**:
- 更新 `self.X`, `self.A`, `self.D` 等节点集合

---

### 步骤 2: BFS 因果图扩展

#### 2.1 主扩展函数 (`expand_toward_target`)
```python
bfs = wiqa.expand_toward_target(
    start_X=start,
    target_Y=target,
    max_depth=3,
    max_relations_per_node=3
)
```

**目的**: 从起点 X 出发，通过 BFS 方式扩展因果图，尝试到达目标 Y

**工作流程**:

```
1. 初始化
   frontier = [start]
   visited = {start}
   triples = []
   close_hits = []

2. BFS 循环 (max_depth 层)
   For each node in frontier:
     ├─ 调用 find_causal_relations(node, target)
     │  └─ LLM 生成从 node 出发的一跳因果关系
     │
     ├─ 对每个新发现的 tail 节点:
     │  ├─ 添加三元组 (node, relation, tail)
     │  ├─ 检查是否命中 target?
     │  │  ├─ 精确匹配 → found=True, 停止
     │  │  ├─ is_same_variable(tail, target) = "same"
     │  │  │   → 添加桥接边, found=True
     │  │  ├─ is_same_variable(tail, target) = "opposite"
     │  │  │   → 添加反向桥接边, found=True
     │  │  └─ is_same_variable(tail, target) = "close"
     │  │      → 记录到 close_hits, 继续搜索
     │  └─ 将 tail 加入下一层 frontier
     │
     └─ 如果 found=True，停止扩展

3. 返回结果
```

**返回值**:
```python
{
    "triples": [(h1, r1, t1), (h2, r2, t2), ...],  # 所有发现的因果三元组
    "visited": {node1, node2, ...},                 # 访问过的节点
    "found_target": True/False,                     # 是否找到目标
    "depth_reached": 3,                             # 实际搜索深度
    "close_hits": [{"node": "...", "depth": 2}, ...] # 与目标接近的节点
}
```

#### 2.2 一跳关系发现 (`find_causal_relations`)
```python
result = wiqa.find_causal_relations(
    X="current node",
    Y="target",  # 作为提示
    max_relations=3
)
```

**目的**: 使用 LLM 发现从当前节点出发的直接因果关系

**LLM Prompt 要求**:
- 从 X 出发，找出 X 直接影响的事物（一跳）
- 使用 RESULTS_IN (正向) 或 NOT_RESULTS_IN (负向) 表示关系
- 最多返回 max_relations 条关系

**示例**:
```
输入: X = "less oxygen is inhaled"
输出: [
  ["less oxygen is inhaled", "NOT_RESULTS_IN", "oxygen in lungs"],
  ["less oxygen is inhaled", "NOT_RESULTS_IN", "oxygen saturation"]
]
```

#### 2.3 变量相似度判断 (`is_same_variable`)
```python
label = wiqa.is_same_variable(
    a="oxygen saturation",
    b="oxygen enters the blood",
    context=question
)
# 返回: "same" / "opposite" / "close" / "different"
```

**目的**: 判断两个短语是否表示同一个因果变量

**判断标准**:
- **same**: 同一变量，方向一致
- **opposite**: 同一变量，方向相反 (如 more X vs less X)
- **close**: 强相关但不完全相同 (如 part-of, subtype)
- **different**: 明显不同

**作用**:
- 帮助 BFS 识别何时"命中"目标
- 即使措辞不同，也能建立因果链

---

### 步骤 3: 因果链提取

#### 3.1 提取路径 (`get_causal_chain`)
```python
chain_result = wiqa.get_causal_chain(
    triples=bfs["triples"],
    start_X=start,
    target_Y=target
)
```

**目的**: 从三元组集合中提取从起点到终点的所有路径

**工作原理**:
```
1. 构建邻接表图
   graph = {
     "node1": [edge1, edge2, ...],
     "node2": [edge3, edge4, ...],
     ...
   }

2. DFS 搜索所有路径
   function dfs(current_node, path):
     if current_node == target:
       保存 path
       return

     for each edge from current_node:
       if edge.tail not in path:  # 避免环
         path.append(edge)
         dfs(edge.tail, path)
         path.pop()

3. 统计信息
   - 路径数量
   - 最短路径长度
   - 路径中涉及的所有节点
```

**返回值**:
```python
{
    "start": "less oxygen is inhaled",
    "target": "oxygen enters the blood",
    "paths": [
        [
            {"head": "less oxygen is inhaled", "relation": "NOT_RESULTS_IN", "tail": "oxygen in lungs"},
            {"head": "oxygen in lungs", "relation": "RESULTS_IN", "tail": "oxygen enters the blood"}
        ],
        # ... 更多路径
    ],
    "num_paths": 2,
    "shortest_path_length": 2,
    "all_nodes_in_paths": {"less oxygen is inhaled", "oxygen in lungs", ...}
}
```

#### 3.2 Close Hits 桥接 (`attach_outcome_via_close`)
```python
# 如果 BFS 发现了 close_hits
if bfs["close_hits"]:
    triples_with_bridge = wiqa.attach_outcome_via_close(
        triples=bfs["triples"],
        close_hits=bfs["close_hits"],
        Y=target,
        max_bridge_nodes=3
    )
```

**目的**: 当没有找到直接路径时，利用 "close" 节点创建桥接边

**工作原理**:
```
1. 按深度排序 close_hits（优先选择距离起点近的）

2. 对前 max_bridge_nodes 个 close 节点:
   添加边: (close_node, RESULTS_IN, target)

3. 这样可能产生新的路径，增加找到因果链的机会
```

**示例**:
```
假设 BFS 发现:
- "oxygen saturation" 与 target "oxygen enters blood" 是 close

添加桥接边:
  ("oxygen saturation", RESULTS_IN, "oxygen enters blood")

现在可能形成新路径:
  start → ... → "oxygen saturation" → "oxygen enters blood"
```

---

### 步骤 4: 生成文字描述

#### 4.1 因果链转文字 (`causal_chain_to_text`)
```python
description = wiqa.causal_chain_to_text(
    chain_result=chain_result,
    bfs_result=bfs
)
```

**目的**: 将因果链或 BFS 结果转化为自然语言描述

**两种情况**:

**情况 A: 找到因果链**
```
原始摘要:
- From 'X' to 'Y', the system found 2 causal path(s).
- Path 1: X --[RESULTS_IN]--> intermediate --[NOT_RESULTS_IN]--> Y
- Path 2: X --[NOT_RESULTS_IN]--> Z --[NOT_RESULTS_IN]--> Y
- In all paths, there are 1 RESULTS_IN edges and 3 NOT_RESULTS_IN edges.

↓ (交给 LLM 润色)

自然语言描述:
"The causal analysis reveals two pathways from X to Y. The first path shows X
increases an intermediate factor which then decreases Y. The second path
demonstrates X decreasing Z, which in turn decreases Y. Overall, the pathways
contain predominantly negative causal links..."
```

**情况 B: 未找到因果链**
```
原始摘要:
- No complete causal path from 'X' to 'Y' was found.
- During local expansion around 'X', the system collected 5 one-hop relations.
- Edge 1: X --[RESULTS_IN]--> A
- Edge 2: X --[RESULTS_IN]--> B
- ...

↓ (交给 LLM 润色)

自然语言描述:
"No direct causal path connecting X to Y was discovered. The exploration around
X identified several immediate effects including A and B, but none of these
connected to Y through further causal chains. This suggests X likely has no
effect on Y..."
```

**设计原则**:
- 客观描述图结构（路径、关系、节点）
- **不在这一层判断答案** (more/less/no_effect)
- 描述交给 LLM 生成，更自然易读

---

### 步骤 5: LLM 推理

#### 5.1 基于描述推理 (`reason_with_description`)
```python
reasoning_result = wiqa.reason_with_description(
    description=description,
    question=question,
    choices=choices
)
```

**目的**: 基于因果分析描述，让 LLM 推理答案

**LLM Prompt 结构**:
```
QUESTION:
suppose less oxygen is inhaled happens, how will it affect MORE oxygen enters the blood.

CAUSAL ANALYSIS:
[生成的自然语言描述]

ANSWER CHOICES:
A. more
B. less
C. no effect

Your task:
1. Read the causal analysis
2. Determine how cause affects outcome
3. Select appropriate answer
4. Provide reasoning

Output JSON:
{
  "predicted_answer": "less",
  "predicted_choice": "B",
  "reasoning": "...",
  "confidence": "high"
}
```

**推理指导原则**:
- RESULTS_IN (正向关系) → 通常导致 "more"
- NOT_RESULTS_IN (负向关系) → 通常导致 "less"
- 无因果路径或效果平衡 → "no_effect"
- 注意方向词 (MORE, LESS, HURTING 等)

**答案归一化**:
```python
# LLM 可能返回各种形式，需要归一化
"more" → "more"
"less" / "fewer" → "less"
"no effect" / "no_effect" / "no change" / "none" → "no_effect"
```

**返回值**:
```python
{
    "predicted_answer": "less",       # 归一化的答案
    "predicted_choice": "B",          # 选项字母
    "reasoning": "The causal analysis shows that less oxygen inhaled leads to
                  decreased oxygen in the blood through a direct negative
                  causal chain. Since the question asks about MORE oxygen,
                  and we have a decrease, the answer is 'less'.",
    "confidence": "high"               # high/medium/low
}
```

---

## 关键机制详解

### 机制 1: 语义匹配 (is_same_variable)

**为什么需要**:
- LLM 生成的节点名称可能与目标不完全匹配
- 例如: "oxygen in blood" vs "oxygen enters the blood"

**如何工作**:
```python
# BFS 过程中，每发现一个新节点
if target:
    label = is_same_variable(new_node, target)

    if label == "same":
        # 同一变量，同向 → 直接连接
        add_edge(new_node, RESULTS_IN, target)
        found = True

    elif label == "opposite":
        # 同一变量，反向 → 反向连接
        add_edge(new_node, NOT_RESULTS_IN, target)
        found = True

    elif label == "close":
        # 相关但不完全相同 → 记录为 close_hit
        close_hits.append(new_node)
```

**示例**:
```
Target: "oxygen enters the blood"

节点1: "blood oxygenation"
→ is_same_variable → "same"
→ 添加桥接: blood oxygenation --[RESULTS_IN]--> oxygen enters the blood

节点2: "oxygen depletion in blood"
→ is_same_variable → "opposite"
→ 添加桥接: oxygen depletion --[NOT_RESULTS_IN]--> oxygen enters the blood

节点3: "hemoglobin saturation"
→ is_same_variable → "close"
→ 记录到 close_hits (可选择性桥接)
```

### 机制 2: Close Hits 桥接

**目的**: 增加找到因果路径的机会

**两阶段策略**:

**阶段 1 - BFS 中标记**:
```python
# BFS 扩展时
if is_same_variable(node, target) == "close":
    close_hits.append({"node": node, "depth": current_depth})
    # 但不立即视为找到目标，继续搜索
```

**阶段 2 - 选择性桥接**:
```python
# BFS 完成后
if chain_result["num_paths"] == 0 and close_hits:
    # 没找到路径，但有 close 节点
    # 添加桥接边，重新尝试找路径
    for hit in close_hits[:max_bridge_nodes]:
        triples.append((hit["node"], RESULTS_IN, target))
```

**权衡**:
- **优点**: 提高覆盖率，处理语义相似情况
- **风险**: 可能引入不准确的桥接
- **控制**: max_bridge_nodes 限制数量

### 机制 3: 两阶段描述生成

**阶段 1 - 结构化摘要**:
```python
# 确定性逻辑生成原始摘要
raw_summary = """
From 'X' to 'Y', found 2 paths.
Path 1: X --[RESULTS_IN]--> Z --[NOT_RESULTS_IN]--> Y
...
3 positive edges, 1 negative edge.
"""
```

**阶段 2 - LLM 润色**:
```python
# 交给 LLM 生成自然语言
prompt = f"""
RAW CAUSAL ANALYSIS:
{raw_summary}

Task: Write a concise, objective description.
Do NOT predict the answer, just describe the structure.
"""

natural_description = llm.generate(prompt)
```

**为什么两阶段**:
1. **可靠性**: 结构化摘要确保关键信息不丢失
2. **可读性**: LLM 润色提高自然度
3. **可控性**: 明确指示 LLM 不要预测答案
4. **回退机制**: LLM 失败时可使用原始摘要

---

## 完整示例运行

### 输入问题
```
Question: "suppose less oxygen is inhaled happens, how will it affect MORE oxygen enters the blood."
Answer: less
```

### 步骤 1: 提取
```python
info = extract_start_entity()
# {
#   "cause_event": "less oxygen is inhaled",
#   "outcome_base": "oxygen enters the blood",
#   "outcome_direction_in_question": "MORE"
# }
```

### 步骤 2: BFS 扩展
```python
bfs = expand_toward_target(start="less oxygen is inhaled", target="oxygen enters the blood")

# BFS 过程:
# Depth 0: ["less oxygen is inhaled"]
#
# Depth 1: find_causal_relations("less oxygen is inhaled", ...)
#   → ("less oxygen is inhaled", NOT_RESULTS_IN, "oxygen in lungs")
#   → ("less oxygen is inhaled", NOT_RESULTS_IN, "oxygen available")
#
# Depth 2: find_causal_relations("oxygen in lungs", ...)
#   → ("oxygen in lungs", RESULTS_IN, "oxygen diffusion to blood")
#   → is_same_variable("oxygen diffusion to blood", "oxygen enters the blood") = "same"
#   → 添加桥接: ("oxygen diffusion to blood", RESULTS_IN, "oxygen enters the blood")
#   → found = True!

# 返回:
# {
#   "triples": [
#     ("less oxygen is inhaled", NOT_RESULTS_IN, "oxygen in lungs"),
#     ("oxygen in lungs", RESULTS_IN, "oxygen diffusion to blood"),
#     ("oxygen diffusion to blood", RESULTS_IN, "oxygen enters the blood")
#   ],
#   "found_target": True,
#   "close_hits": []
# }
```

### 步骤 3: 提取路径
```python
chain_result = get_causal_chain(bfs["triples"], ...)

# {
#   "paths": [
#     [
#       {"head": "less oxygen is inhaled", "relation": "NOT_RESULTS_IN", "tail": "oxygen in lungs"},
#       {"head": "oxygen in lungs", "relation": "RESULTS_IN", "tail": "oxygen diffusion to blood"},
#       {"head": "oxygen diffusion to blood", "relation": "RESULTS_IN", "tail": "oxygen enters the blood"}
#     ]
#   ],
#   "num_paths": 1
# }
```

### 步骤 4: 生成描述
```python
description = causal_chain_to_text(chain_result, bfs)

# 输出:
# "From 'less oxygen is inhaled' to 'oxygen enters the blood', one causal pathway
#  was identified. The path demonstrates that reduced oxygen intake decreases the
#  oxygen present in the lungs, which subsequently affects oxygen diffusion and
#  ultimately impacts the amount entering the bloodstream. The pathway contains
#  one negative link followed by two positive links."
```

### 步骤 5: LLM 推理
```python
result = reason_with_description(description, question, choices)

# LLM 推理过程:
# - 看到路径: less O2 → [-] → O2 in lungs → [+] → diffusion → [+] → blood
# - 一个负向关系 (NOT_RESULTS_IN) + 两个正向关系 (RESULTS_IN)
# - 负向在开头: less O2 inhaled → less O2 in lungs
# - 最终效果: less O2 进入血液
# - 问题问的是"MORE oxygen enters blood"的影响
# - 答案: "less" (因为实际是减少)

# 返回:
# {
#   "predicted_answer": "less",
#   "predicted_choice": "B",
#   "reasoning": "The causal chain shows that less oxygen inhaled leads to
#                 decreased oxygen in the lungs. Although there are positive
#                 causal links afterward, the initial decrease propagates through,
#                 resulting in less oxygen entering the blood. Since the question
#                 asks about MORE oxygen, the effect is 'less'.",
#   "confidence": "high"
# }
```

### 验证
```python
predicted: "less"
actual: "less"
✓ 正确!
```

---

## 系统特点总结

### 优势

1. **LLM 驱动的灵活性**
   - 自动生成因果关系（无需预定义知识库）
   - 处理开放域问题
   - 语义理解和匹配

2. **多层推理**
   - BFS 扩展: 探索因果空间
   - 路径提取: 结构化因果链
   - 文字描述: 可解释的中间表示
   - LLM 推理: 最终决策

3. **鲁棒性机制**
   - is_same_variable: 处理措辞差异
   - close_hits: 增加覆盖率
   - 两阶段描述: 可靠性+可读性
   - 答案归一化: 容错处理

4. **可解释性**
   - 完整的因果链
   - 自然语言描述
   - 推理过程
   - 置信度评分

### 局限性

1. **LLM 依赖**
   - 需要多次 LLM 调用（慢、贵）
   - 质量依赖模型能力
   - 可能出现幻觉

2. **搜索空间**
   - BFS 深度和宽度受限
   - 可能错过复杂路径
   - 计算成本随深度增加

3. **桥接风险**
   - close_hits 桥接可能不准确
   - 语义"接近"判断主观
   - 可能引入噪声

4. **方向推理**
   - 复杂路径的正负效果累积可能出错
   - 依赖 prompt 设计

---

## 关键参数影响

| 参数 | 作用 | 推荐值 | 影响 |
|------|------|--------|------|
| `max_depth` | BFS 最大深度 | 3-5 | 太小找不到路径，太大噪声增加 |
| `max_relations_per_node` | 每个节点的扩展边数 | 3-5 | 平衡覆盖率和计算成本 |
| `max_bridge_nodes` | 使用的 close_hits 数量 | 1-3 | 控制桥接的激进程度 |
| `model_name` | LLM 模型 | gemma2:27b | 更强模型 → 更好推理，但更慢 |

---

## 总结

`WIQACausalBuilder` 是一个**混合符号-神经**的因果推理系统:

- **符号层**: BFS 搜索、图结构、路径提取
- **神经层**: LLM 生成关系、判断相似度、最终推理

通过将结构化推理与 LLM 灵活性结合，系统在保持可解释性的同时实现了开放域因果推理。
