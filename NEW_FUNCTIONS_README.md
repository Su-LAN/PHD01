# WIQACausalBuilder 新功能说明

本文档介绍 `WIQACausalBuilder` 类中新增的两个功能函数。

## 新增功能

### 1. `causal_chain_to_text()` - 因果链转文字描述

将因果链或 BFS 结果转化为易读的描述性文字。

#### 函数签名

```python
def causal_chain_to_text(
    self,
    chain_result: Dict[str, Any],
    bfs_result: Optional[Dict[str, Any]] = None,
) -> str
```

#### 参数说明

- `chain_result`: `get_causal_chain()` 的返回结果
- `bfs_result`: `expand_toward_target()` 的返回结果（可选）

#### 返回值

返回一个描述性字符串，包含：

**情况 1: 找到因果链时**
- 因果路径的数量
- 每条路径的详细描述（最多显示前 3 条）
- 正向关系和负向关系的统计
- 总体影响方向的判断

**情况 2: 没有找到因果链时**
- 说明未找到直接因果路径
- 列出 BFS 探索过程中发现的因果关系（最多显示前 5 个）
- 说明探索了哪些中间节点
- 推断为"无影响"

#### 使用示例

```python
wiqa = WIQACausalBuilder(datapoint)
info = wiqa.extract_start_entity()
start = info["cause_event"]
target = info["outcome_base"]

# 执行 BFS 扩展
bfs = wiqa.expand_toward_target(start_X=start, target_Y=target, max_depth=3)

# 提取因果链
chain_result = wiqa.get_causal_chain(bfs["triples"], start_X=start, target_Y=target)

# 转化为文字描述
description = wiqa.causal_chain_to_text(chain_result, bfs)
print(description)
```

#### 输出示例

**有因果链时：**
```
Based on causal analysis, starting from 'the seal is sealed less securely', we found 2 causal path(s) leading to 'the letter being mailed':

Path 1: 'the seal is sealed less securely' decreases 'envelope integrity' → 'envelope integrity' decreases 'successful delivery'

Path 2: 'the seal is sealed less securely' increases 'letter damage risk' → 'letter damage risk' decreases 'the letter being mailed'

Summary: The causal chain contains 1 positive relation(s) and 3 negative relation(s).
Overall, 'the seal is sealed less securely' tends to DECREASE 'the letter being mailed'.
```

**无因果链时：**
```
No direct causal path was found from 'more tank trucks added' to 'stamps used'.

However, during exploration, we discovered 3 causal relation(s) starting from 'more tank trucks added':
1. 'more tank trucks added' increases 'gasoline transportation'
2. 'more tank trucks added' increases 'fuel availability'
3. 'more tank trucks added' decreases 'transportation cost'

These relations explored 5 intermediate node(s), but none directly connected to 'stamps used'.
This suggests that 'more tank trucks added' likely has NO EFFECT on 'stamps used'.
```

---

### 2. `reason_with_description()` - 基于描述的推理

使用描述性文字、问题和选项，让 LLM 进行推理判断。

#### 函数签名

```python
def reason_with_description(
    self,
    description: str,
    question: Optional[str] = None,
    choices: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]
```

#### 参数说明

- `description`: 因果链或 BFS 结果的描述性文字（通常来自 `causal_chain_to_text()`）
- `question`: 问题文本（可选，默认使用 `self.question`）
- `choices`: 选项字典（可选，默认使用 `self.datapoint.get('choices')`）

#### 返回值

返回一个包含以下字段的字典：

```python
{
    "predicted_answer": str,        # 预测的答案标签（如 "more", "less", "no_effect"）
    "predicted_choice": str,        # 预测的选项字母（如 "A", "B", "C"）
    "reasoning": str,               # LLM 的推理过程
    "confidence": str,              # 置信度（"high", "medium", "low"）
}
```

#### 使用示例

```python
# 先生成描述性文字
description = wiqa.causal_chain_to_text(chain_result, bfs)

# 使用描述进行推理
reasoning_result = wiqa.reason_with_description(description)

print(f"预测答案: {reasoning_result['predicted_answer']}")
print(f"预测选项: {reasoning_result['predicted_choice']}")
print(f"置信度: {reasoning_result['confidence']}")
print(f"推理过程: {reasoning_result['reasoning']}")

# 验证结果
if reasoning_result['predicted_answer'] == datapoint['answer_label']:
    print("✓ 预测正确!")
```

#### 输出示例

```python
{
    "predicted_answer": "more",
    "predicted_choice": "A",
    "reasoning": "The causal analysis shows that 'the seal is sealed less securely' has 3 negative relations and 1 positive relation, indicating it tends to DECREASE 'the letter being mailed'. Since the question asks about HURTING the letter, a decrease in successful mailing means MORE hurting.",
    "confidence": "high"
}
```

---

## 完整工作流程

### 推荐使用流程

```python
from WIQACausalBuilder import WIQACausalBuilder

# 1. 初始化
wiqa = WIQACausalBuilder(datapoint)

# 2. 提取起点和终点
info = wiqa.extract_start_entity()
start = info["cause_event"]
target = info["outcome_base"]

# 3. BFS 扩展因果图
bfs = wiqa.expand_toward_target(
    start_X=start,
    target_Y=target,
    max_depth=3,
    max_relations_per_node=3
)

# 4. 提取因果链
chain_result = wiqa.get_causal_chain(
    bfs["triples"],
    start_X=start,
    target_Y=target
)

# 5. 转化为描述性文字
description = wiqa.causal_chain_to_text(chain_result, bfs)
print("因果分析描述:")
print(description)

# 6. 使用描述进行推理
reasoning_result = wiqa.reason_with_description(description)

# 7. 获取预测结果
predicted_answer = reasoning_result['predicted_answer']
confidence = reasoning_result['confidence']

print(f"\n预测答案: {predicted_answer}")
print(f"置信度: {confidence}")
```

---

## 关键特性

### `causal_chain_to_text()` 的特性

1. **智能选择描述策略**：
   - 有因果链时，重点描述路径
   - 无因果链时，利用 BFS 结果说明为何无影响

2. **清晰的文字表达**：
   - 使用自然语言（"increases", "decreases"）
   - 提供路径摘要和统计信息
   - 给出明确的影响方向判断

3. **可控的输出长度**：
   - 最多显示 3 条路径（避免过长）
   - BFS 结果最多显示 5 个关系

### `reason_with_description()` 的特性

1. **基于因果分析的推理**：
   - 不直接回答问题，而是基于提供的因果分析
   - 理解方向词（MORE, LESS, HURTING 等）
   - 考虑正负关系的平衡

2. **结构化输出**：
   - JSON 格式返回，易于解析
   - 提供推理过程和置信度
   - 包含答案文本和选项字母

3. **错误处理**：
   - JSON 解析失败时自动降级
   - 尝试从文本中提取答案
   - 异常情况返回默认值

---

## 测试

运行测试脚本：

```bash
python test_new_functions.py
```

测试脚本会运行两个场景：
1. 有因果链的情况
2. 无因果链的情况（no_effect）

---

## 注意事项

1. **LLM 依赖**：这两个函数都依赖 LLM（通过 `_call_llm()` 方法），确保 Ollama 服务正在运行。

2. **BFS 结果的重要性**：即使没有找到因果链，BFS 结果仍然有用，可以解释为什么没有影响。

3. **方向词处理**：`reason_with_description()` 会特别注意问题中的方向词（MORE, LESS, HURTING 等），并相应调整推理逻辑。

4. **置信度**：LLM 提供的置信度可以帮助过滤低置信度的预测。

---

## 与原有功能的集成

这两个新功能可以完美集成到现有的测试流程中（如 `test_v2.ipynb`）：

```python
# 在测试循环中
for datapoint in test_samples:
    wiqa = WIQACausalBuilder(datapoint)

    # ... 提取起点终点、BFS 扩展、提取因果链 ...

    # 使用新功能进行推理
    description = wiqa.causal_chain_to_text(chain_result, bfs)
    reasoning_result = wiqa.reason_with_description(description)

    # 获取预测并验证
    predicted = reasoning_result['predicted_answer']
    is_correct = (predicted == datapoint['answer_label'])
```

这样可以实现更可解释、更可靠的推理流程。
