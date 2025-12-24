# Choice Reasoning V2 - 实现计划

## 核心改进

### 1. 在ego_expansion_builder.py中已添加的新方法

#### ✅ `get_causal_chain_paths()`
```python
def get_causal_chain_paths(
    result: Dict,
    intervention: str = None,
    target: str = None,
    min_confidence: float = 0.0,
    max_length: int = 5,
    max_paths: int = 10
) -> List[List[Dict]]
```
**功能**: 从三元组构建完整的因果链路径（edge序列）

**返回示例**:
```python
[
    [  # Chain 1
        {'head': 'stomach bug', 'relation': 'increases', 'tail': 'gastric acid', 'confidence': 0.85},
        {'head': 'gastric acid', 'relation': 'decreases', 'tail': 'pH', 'confidence': 0.75}
    ],
    [  # Chain 2
        {'head': 'stomach bug', 'relation': 'causes', 'tail': 'inflammation', 'confidence': 0.80}
    ]
]
```

#### ✅ `get_formatted_causal_chains()`
```python
def get_formatted_causal_chains(
    result: Dict,
    intervention: str = None,
    target: str = None,
    min_confidence: float = 0.3,
    max_paths: int = 10
) -> str
```
**功能**: 返回格式化的因果链文本（直接用于LLM prompt）

**返回示例**:
```
Chain 1: stomach bug --[increases]--> gastric acid (0.85) → gastric acid --[decreases]--> pH (0.75)
Chain 2: stomach bug --[causes]--> inflammation (0.80) → inflammation --[affects]--> digestion (0.70)
```

### 2. 在notebook中需要实现的新流程

#### Step 1: 识别观察对象
```python
observable_info = identify_observable_outcome(
    question=question,
    question_structure=question_structure,
    model=MODEL
)
# 返回: {'observable_outcome': 'eating more vegetables', 'reasoning': '...'}
```

#### Step 2: 获取格式化的因果链
```python
causal_chains_text = BUILDER.get_formatted_causal_chains(
    result=builder_result,
    intervention=question_structure.get('intervention'),
    target=question_structure.get('target_entity'),
    min_confidence=CONFIDENCE_THRESHOLD
)
```

#### Step 3: 对每个choice进行两次独立的LLM调用

**3.1 正向推理**
```python
forward_result = forward_reasoning(
    question=question,
    choice=choice,
    observable_outcome=observable_info['observable_outcome'],
    causal_chains_text=causal_chains_text,  # 完整的链！
    question_structure=question_structure,
    model=MODEL
)
# 返回: {
#     'confidence': 0.7,
#     'supporting_evidence': [...],
#     'causal_path': '...',
#     'path_strength': 'moderate'
# }
```

**3.2 反事实推理**（独立调用，不共享上下文）
```python
counterfactual_result = counterfactual_reasoning(
    question=question,
    choice=choice,
    observable_outcome=observable_info['observable_outcome'],
    causal_chains_text=causal_chains_text,  # 同样的链
    question_structure=question_structure,
    model=MODEL
)
# 返回: {
#     'test_result': 'pass',
#     'counterfactual_scenario': '...',
#     'confidence_adjustment': +0.2
# }
```

#### Step 4: 综合决策
```python
final_decision = make_final_decision(
    question=question,
    observable_outcome=observable_info['observable_outcome'],
    choice_evaluations=[
        {
            'choice': 'more',
            'forward_reasoning': forward_more,
            'counterfactual_reasoning': cf_more
        },
        {
            'choice': 'less',
            'forward_reasoning': forward_less,
            'counterfactual_reasoning': cf_less
        },
        {
            'choice': 'no_effect',
            'forward_reasoning': forward_no_effect,
            'counterfactual_reasoning': cf_no_effect
        }
    ],
    model=MODEL
)
```

## 主推理函数结构

```python
def predict_with_choice_reasoning_v2(
    question: str,
    parser,
    builder,
    model: str,
    confidence_threshold: float = 0.3,
    print_details: bool = True
) -> Dict[str, Any]:

    # 1. 解析问题结构
    question_structure = parser.parse_question_structure(question)

    # 2. 构建因果链
    builder_result = builder.build_causal_chain(question)

    # 3. **新增**: 识别观察对象
    observable_info = identify_observable_outcome(
        question, question_structure, model
    )

    # 4. **新增**: 获取格式化的因果链（不是散列的三元组！）
    causal_chains_text = builder.get_formatted_causal_chains(
        builder_result,
        intervention=question_structure.get('intervention'),
        target=question_structure.get('target_entity'),
        min_confidence=confidence_threshold
    )

    # 5. 对每个choice进行评估
    choice_evaluations = []
    for choice in ['more', 'less', 'no_effect']:
        # 5.1 正向推理（独立LLM调用）
        forward = forward_reasoning(
            question, choice, observable_info['observable_outcome'],
            causal_chains_text, question_structure, model
        )

        # 5.2 反事实推理（独立LLM调用）
        counterfactual = counterfactual_reasoning(
            question, choice, observable_info['observable_outcome'],
            causal_chains_text, question_structure, model
        )

        choice_evaluations.append({
            'choice': choice,
            'forward_reasoning': forward,
            'counterfactual_reasoning': counterfactual
        })

    # 6. 最终决策
    final_decision = make_final_decision(
        question, observable_info['observable_outcome'],
        choice_evaluations, model
    )

    return {
        'observable_outcome': observable_info,
        'causal_chains_text': causal_chains_text,
        'choice_evaluations': choice_evaluations,
        'final_decision': final_decision,
        'final_answer': final_decision['final_answer']
    }
```

## 关键优势

### 1. 提前识别观察对象
- **问题**: "MORE vegetables" 是什么意思？
- **解决**: 在推理开始前就明确（吃/种/买）
- **好处**: 所有choice基于同一理解进行评估

### 2. 使用完整的因果链
- **之前**: 散列的三元组列表
  ```
  - stomach bug → gastric acid
  - gastric acid → pH
  - pH → digestion
  ```
- **现在**: 完整的因果路径
  ```
  Chain 1: stomach bug --[increases]--> gastric acid (0.85) → gastric acid --[decreases]--> pH (0.75) → pH --[affects]--> digestion (0.70)
  ```
- **好处**: LLM能看到完整的因果机制，不用自己拼接

### 3. 两次独立的LLM调用
- **正向**: "假设这个选项是对的，找证据"
- **反事实**: "如果干预没发生，会怎样？"
- **好处**: 避免第一次推理影响第二次，更客观

### 4. 更严格的no_effect评估
- 反事实测试专门验证因果必要性
- 如果通过测试→有很强的no_effect证据
- 不会被弱的推测性证据误导

## 下一步

需要在notebook中实现：
1. ✅ `identify_observable_outcome()` 函数（已在cell-9）
2. ⚠️ 删除notebook中的 `build_causal_chains()` 和 `format_causal_chains()` - 使用builder的方法
3. ✅ `forward_reasoning()` 函数（已在cell-13）
4. ✅ `counterfactual_reasoning()` 函数（已在cell-13）
5. ✅ `make_final_decision()` 函数（已在cell-15）
6. ⚠️ 更新主推理函数使用新流程

