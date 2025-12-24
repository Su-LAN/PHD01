# WIQA 推理模块 (WIQA Reasoning Module)

这是一个用于处理 WIQA 数据集问题的推理模块，可以从问题中提取事件对，并生成完整的推理输入字典。

## 📦 文件说明

- **`qiwa_function.py`** - 核心推理模块
- **`wiqa_reasoning_usage.py`** - 使用示例和文档

## 🚀 快速开始

### 1. 处理单个问题

```python
from qiwa_function import process_question

# 输入一个 question_stem
question = "suppose less gasoline is loaded onto tank trucks happens, how will it affect LESS electricity being produced."

# 返回完整的推理字典
result = process_question(question, answer_label='no_effect')

print(result['question'])      # 问题文本
print(result['choices'])       # 选项 {'text': ['more', 'less', 'no effect'], 'label': ['A', 'B', 'C']}
print(result['answer'])        # 答案: 'no_effect'
print(result['event_pair'])    # 事件对
```

### 2. 批量处理数据集

```python
from qiwa_function import process_dataset

# 准备数据集（列表或 HuggingFace dataset）
dataset = [
    {
        'question_stem': 'suppose more pollution happens, how will it affect MORE disease.',
        'answer_label': 'more',
        'metadata_graph_id': '123',
        'question_para_step': ['Step 1', 'Step 2', 'Step 3']
    },
    # ... 更多样本
]

# 批量处理，返回推理字典列表
results = process_dataset(dataset)

for result in results:
    print(result['question'])
    print(result['event_pair'])
```

## 📋 返回的字典结构

每个处理后的问题返回以下结构：

```python
{
    'question': str,               # 原始问题文本
    'choices': {                   # 答案选项
        'text': ['more', 'less', 'no effect'],
        'label': ['A', 'B', 'C']
    },
    'influence_graph': dict/None,  # 影响图（如果提供了 graph_id）
    'answer': str,                 # 正确答案标签
    'event_pair': {                # 提取的事件对
        'cause': str,              # 扰动事件（原因）
        'outcome': str,            # 结果事件
        'direction': str,          # 方向: 'MORE', 'LESS', 'NORMAL'
        'answer_label': str        # 答案标签
    },
    'para_steps': list,            # 过程步骤
    'metadata': dict               # 元数据信息
}
```

## 💡 主要功能

### 🔹 提取事件对

从 WIQA 问题中自动提取：
- **Cause（扰动事件）**: `"less gasoline is loaded onto tank trucks"`
- **Outcome（结果事件）**: `"electricity being produced"`
- **Direction（方向）**: `"LESS"`（自动识别 MORE/LESS/NORMAL）

### 🔹 两种处理方式

#### 方式 1: 快捷函数

```python
# 处理单个问题
result = process_question(question_stem, answer_label)

# 批量处理
results = process_dataset(dataset, limit=100)
```

#### 方式 2: 使用类（更多控制）

```python
from qiwa_function import WIQAReasoningModule

module = WIQAReasoningModule()

# 处理单个问题（可以传更多参数）
result = module.process_single_question(
    question_stem=question,
    answer_label='more',
    graph_id='123',
    para_steps=['step1', 'step2'],
    metadata={'custom': 'info'}
)

# 处理数据集样本
result = module.process_dataset_item(dataset_item)

# 批量处理
results = module.process_dataset(dataset)
```

## 📚 使用示例

### 示例 1: 基本使用

```python
from qiwa_function import process_question

question = "suppose more pollution in the environment happens, how will it affect more cocoon hatches."
result = process_question(question, answer_label='less')

# 提取关键信息用于推理
print(f"Cause: {result['event_pair']['cause']}")
# Output: more pollution in the environment

print(f"Outcome: {result['event_pair']['outcome']}")
# Output: cocoon hatches

print(f"Direction: {result['event_pair']['direction']}")
# Output: MORE

print(f"Answer: {result['answer']}")
# Output: less
```

### 示例 2: 处理 WIQA 数据集

```python
from datasets import load_dataset
from qiwa_function import process_dataset

# 加载 WIQA 数据集
ds = load_dataset('allenai/wiqa', split='validation[:100]')

# 处理前 100 个样本
results = process_dataset(ds, limit=100)

print(f"处理了 {len(results)} 个问题")

# 访问每个结果
for result in results:
    cause = result['event_pair']['cause']
    outcome = result['event_pair']['outcome']
    answer = result['answer']
    print(f"{cause} -> {outcome}: {answer}")
```

### 示例 3: 提取推理输入

```python
from qiwa_function import process_question

question = "suppose there will be fewer new trees happens, how will it affect LESS forest formation."
result = process_question(question, answer_label='more')

# 构造推理模型的输入
reasoning_input = {
    'perturbation': result['event_pair']['cause'],
    'outcome': result['event_pair']['outcome'],
    'direction': result['event_pair']['direction'],
    'context': result['para_steps'],
    'question': result['question'],
}

# 将这个字典传给你的推理模型
# model.predict(reasoning_input)
```

## 🎯 适用场景

✅ **场景 1**: 从问题中提取事件对进行因果推理
✅ **场景 2**: 批量预处理 WIQA 数据集
✅ **场景 3**: 为推理模型准备标准化输入
✅ **场景 4**: 分析 WIQA 问题的结构和模式

## 📌 注意事项

1. **Direction 识别**: 自动识别 `MORE/GREATER/LARGER/STRONGER` 和 `LESS/SMALLER/FEWER/WEAKER`
2. **Influence Graph**: 需要提供 `graph_id` 才能加载影响图
3. **元数据**: 自动从数据集样本中提取 `metadata_*` 字段

## 🔧 API 参考

### process_question()

```python
def process_question(
    question_stem: str,
    answer_label: Optional[str] = None,
    graph_id: Optional[str] = None
) -> Dict
```

**参数:**
- `question_stem`: 问题文本
- `answer_label`: 答案标签 ('more', 'less', 'no_effect')
- `graph_id`: 影响图 ID（可选）

**返回:** 推理字典

### process_dataset()

```python
def process_dataset(
    dataset: Any,
    limit: Optional[int] = None
) -> List[Dict]
```

**参数:**
- `dataset`: WIQA 数据集（列表或 HuggingFace dataset）
- `limit`: 最大处理数量（None 表示全部）

**返回:** 推理字典列表

### WIQAReasoningModule 类

```python
class WIQAReasoningModule:
    def __init__(self, whatif_metadata=None)

    def process_single_question(
        self,
        question_stem: str,
        answer_label: Optional[str] = None,
        graph_id: Optional[str] = None,
        para_steps: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict

    def process_dataset_item(self, item: Dict) -> Dict

    def process_dataset(
        self,
        dataset: Any,
        limit: Optional[int] = None
    ) -> List[Dict]
```

## 📖 更多示例

运行 `wiqa_reasoning_usage.py` 查看完整的使用示例：

```bash
python wiqa_reasoning_usage.py
```

包含以下示例：
- 示例 1: 处理单个 question_stem
- 示例 2: 处理完整的数据集样本
- 示例 3: 批量处理多个问题
- 示例 4: 使用 HuggingFace WIQA 数据集
- 示例 5: 提取推理所需的关键信息
