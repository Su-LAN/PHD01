# 如何使用改进后的Choice Reasoning系统

## 核心改进总结

您的建议完全正确！现在系统按以下方式工作：

### 1. ✅ 在ego_expansion_builder.py中已添加
```python
# 获取格式化的因果链（完整路径，不是散列三元组）
causal_chains_text = BUILDER.get_formatted_causal_chains(
    result=builder_result,
    intervention="stomach bug",
    target="vegetables",
    min_confidence=0.3
)
```

**输出**:
```
Chain 1: stomach bug --[increases]--> gastric acid (0.85) → gastric acid --[decreases]--> pH (0.75)
Chain 2: stomach bug --[causes]--> inflammation (0.80)
```

### 2. ✅ 提前识别观察对象
在choice评估之前，先明确"MORE vegetables"指什么：
```python
observable_info = identify_observable_outcome(
    question="suppose Having a stomach bug happens, how will it affect MORE vegetables",
    question_structure=parsed_structure,
    model=MODEL
)
# → {'observable_outcome': 'eating more vegetables', 'reasoning': '...'}
```

### 3. ✅ 每个choice两次独立LLM调用

**第一次：正向推理**
```python
forward_result = forward_reasoning(
    question=question,
    choice="more",  # 或 "less" 或 "no_effect"
    observable_outcome="eating more vegetables",
    causal_chains_text=causal_chains_text,  # 完整的链！
    question_structure=question_structure,
    model=MODEL
)
```
- 假设这个choice是正确的
- 在causal chains中寻找支持证据
- 返回confidence和支持理由

**第二次：反事实推理**（独立调用）
```python
counterfactual_result = counterfactual_reasoning(
    question=question,
    choice="more",
    observable_outcome="eating more vegetables",
    causal_chains_text=causal_chains_text,  # 同样的链
    question_structure=question_structure,
    model=MODEL
)
```
- 测试：如果没有stomach bug，会怎样？
- 验证因果关系的必要性
- 返回test_result (pass/fail/unclear)和confidence_adjustment

## 在Notebook中的使用示例

```python
# Step 1: 解析问题
question_structure = PARSER.parse_question_structure(question)

# Step 2: 构建因果链
builder_result = BUILDER.build_causal_chain(question)

# Step 3: 识别观察对象（新增！）
observable_info = identify_observable_outcome(
    question, question_structure, MODEL
)

# Step 4: 获取格式化的因果链（使用builder方法！）
causal_chains_text = BUILDER.get_formatted_causal_chains(
    builder_result,
    intervention=question_structure.get('intervention'),
    target=question_structure.get('target_entity'),
    min_confidence=CONFIDENCE_THRESHOLD
)

# Step 5: 对每个choice评估
for choice in ['more', 'less', 'no_effect']:
    # 5.1 正向推理
    forward = forward_reasoning(
        question, choice, observable_info['observable_outcome'],
        causal_chains_text, question_structure, MODEL
    )

    # 5.2 反事实推理（独立调用！）
    counterfactual = counterfactual_reasoning(
        question, choice, observable_info['observable_outcome'],
        causal_chains_text, question_structure, MODEL
    )

    # 合并结果
    choice_evaluations.append({
        'choice': choice,
        'forward': forward,
        'counterfactual': counterfactual
    })

# Step 6: 最终决策
final = make_final_decision(
    question, observable_info['observable_outcome'],
    choice_evaluations, MODEL
)
```

## 关键点

### ✅ 观察对象识别
- **时机**: 在seed extraction和因果链构建之后，choice评估之前
- **共享**: 所有三个choice使用同一个观察对象解释
- **好处**: 避免"MORE vegetables"的歧义

### ✅ 因果链格式
- **不是**: 散列的三元组
  ```
  - A → B
  - B → C
  - X → Y
  ```
- **而是**: 完整的路径
  ```
  Chain 1: A --[rel1]--> B (0.8) → B --[rel2]--> C (0.7)
  Chain 2: X --[rel3]--> Y (0.6)
  ```

### ✅ 独立的LLM调用
- forward和counterfactual不共享对话上下文
- 避免第一次推理影响第二次
- 更客观、更可靠

## Notebook中需要删除/修改的部分

**删除**cell-11中的：
- `build_causal_chains()` 函数
- `format_causal_chains()` 函数

**改用**:
```python
# 直接使用builder的方法
causal_chains_text = BUILDER.get_formatted_causal_chains(...)
```

**主推理函数**需要更新为使用新流程（见上面的示例）

