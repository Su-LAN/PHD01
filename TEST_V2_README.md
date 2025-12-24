# Test V2 - 更新版测试说明

本文档说明如何使用更新后的 `WIQACausalBuilder` 进行测试。

## 更新内容

### 新增功能

1. **`causal_chain_to_text()`** - 将因果链转为描述性文字
   - 有因果链时：描述路径结构和关系统计
   - 无因果链时：利用 BFS 结果生成说明
   - 使用 LLM 生成自然语言描述

2. **`reason_with_description()`** - 基于描述的 LLM 推理
   - 接收因果分析描述
   - 结合问题和选项进行推理
   - 返回预测答案、置信度和推理过程

3. **`attach_outcome_via_close()`** - 利用 close_hits 桥接
   - 当节点与目标"语义上接近"时创建桥接边
   - 增加找到因果链的可能性

4. **`expand_toward_target()` 增强** - 返回 close_hits
   - BFS 过程中记录与目标 close 的节点
   - 支持后续桥接处理

## 文件说明

### 1. `test_v2.ipynb` - Jupyter Notebook 测试

完整的交互式测试环境，包含：

**核心测试函数：**
- `test_causal_builder_v2()`: 使用新版 WIQACausalBuilder 进行测试

**统计分析：**
- 按真实标签统计准确率
- 按置信度统计准确率
- 混淆矩阵
- 有/无因果链的对比分析

**案例分析：**
- 错误案例详细分析（包含推理过程）
- 成功案例分析
- 因果分析描述展示

**结果保存：**
- JSON 格式详细结果
- TXT 格式简要报告

### 2. `test_v2_simple.py` - 简化测试脚本

命令行测试工具，支持：

**测试模式：**
1. 单样本详细测试 - 显示完整的推理过程
2. 多样本快速测试 (10 个) - 快速验证
3. 多样本快速测试 (100 个) - 全面评估

**使用方法：**
```bash
python test_v2_simple.py
```

然后根据提示选择测试模式。

## 测试流程

### 完整测试流程

```python
# 1. 初始化
wiqa = WIQACausalBuilder(datapoint)

# 2. 提取起点和终点
info = wiqa.extract_start_entity()
start = info["cause_event"]
target = info["outcome_base"]

# 3. BFS 扩展
bfs = wiqa.expand_toward_target(
    start_X=start,
    target_Y=target,
    max_depth=3,
    max_relations_per_node=3
)

# 4. 处理 close_hits（如果有）
close_hits = bfs.get("close_hits", [])
if close_hits:
    triples = wiqa.attach_outcome_via_close(
        bfs["triples"],
        close_hits,
        target,
        max_bridge_nodes=3
    )
else:
    triples = bfs["triples"]

# 5. 提取因果链
chain_result = wiqa.get_causal_chain(triples, start_X=start, target_Y=target)

# 6. 生成描述
description = wiqa.causal_chain_to_text(chain_result, bfs)

# 7. LLM 推理
reasoning_result = wiqa.reason_with_description(description)

# 8. 获取结果
predicted = reasoning_result['predicted_answer']
confidence = reasoning_result['confidence']
reasoning = reasoning_result['reasoning']
```

## 测试结果分析

### 输出指标

1. **基础指标**
   - 总样本数
   - 正确/错误/错误处理数
   - 准确率
   - 平均耗时

2. **扩展指标**
   - 未找到路径的样本数
   - 有 close_hits 的样本数
   - 按标签的准确率分布
   - 按置信度的准确率分布

3. **对比分析**
   - 找到因果链 vs 未找到因果链的准确率
   - 有 close_hits vs 无 close_hits 的准确率

### 结果文件

测试完成后会生成两个文件：

1. **`causal_builder_results_<timestamp>.json`**
   - 详细的测试结果
   - 每个样本的完整信息
   - 因果分析描述和推理过程

2. **`causal_builder_summary_<timestamp>.txt`**
   - 简要统计报告
   - 按标签统计
   - 易于阅读的文本格式

## 关键改进

### 相比旧版本的改进

1. **更智能的推理**
   - 使用 LLM 理解因果描述
   - 考虑问题的语义和上下文
   - 提供推理过程的可解释性

2. **更高的覆盖率**
   - close_hits 机制增加找到路径的机会
   - 即使没有直接路径也能基于 BFS 结果推理

3. **更好的可解释性**
   - 自然语言描述因果关系
   - 展示推理过程和置信度
   - 便于错误分析和调试

4. **更全面的分析**
   - 置信度统计
   - close_hits 效果分析
   - 详细的案例研究

## 性能对比

### 与基础版本对比

| 指标 | 基础版本 | 新版本 V2 |
|------|---------|-----------|
| 推理方法 | 规则基础 | LLM 推理 |
| 可解释性 | 低 | 高 |
| 处理 close 节点 | 否 | 是 |
| 提供置信度 | 否 | 是 |
| 推理过程 | 不可见 | 完整展示 |
| 平均耗时 | 快 | 较慢（需调用 LLM） |

## 使用建议

### 何时使用新版本

1. **需要高可解释性** - 查看完整的因果分析和推理过程
2. **复杂问题** - LLM 能更好地处理复杂的语义关系
3. **研究和调试** - 详细的输出便于分析错误原因
4. **置信度重要** - 需要知道预测的可靠程度

### 何时使用基础版本

1. **大规模测试** - 需要快速处理大量样本
2. **资源受限** - LLM 调用需要更多计算资源
3. **规则明确** - 简单的因果关系用规则即可

### 参数调优建议

1. **`max_depth`**: 3-5
   - 太小可能找不到路径
   - 太大会增加计算时间和噪声

2. **`max_relations_per_node`**: 3-5
   - 平衡覆盖率和计算成本

3. **`max_bridge_nodes`**: 1-3
   - 控制通过 close_hits 桥接的数量
   - 避免引入太多不确定性

## 常见问题

### Q1: 测试速度慢？
A: 新版本需要多次调用 LLM（提取起点、BFS 扩展、生成描述、推理），比基础版本慢。可以：
- 减少测试样本数
- 使用更快的 LLM 模型
- 先用基础版本快速筛选，再用新版本详细分析

### Q2: 准确率没有提高？
A: 准确率取决于多个因素：
- LLM 模型的质量
- 因果图构建的准确性
- prompt 的设计
- 可以尝试调整 prompt 或使用更强的模型

### Q3: close_hits 有帮助吗？
A: 查看对比分析单元格的统计：
- 如果"有 close_hits 的样本"准确率更高，说明有帮助
- 可以调整 `max_bridge_nodes` 参数优化

### Q4: 如何改进？
A: 几个方向：
- 优化 `causal_chain_to_text` 的描述质量
- 改进 `reason_with_description` 的 prompt
- 调整 `is_same_variable` 的判断逻辑
- 实验不同的 LLM 模型

## 示例输出

### 成功案例示例

```
问题: suppose less oxygen is inhaled happens, how will it affect MORE oxygen enters the blood.
正确答案: less

起点: less oxygen is inhaled
终点: oxygen enters the blood
找到 2 条因果链

因果分析描述:
From 'less oxygen is inhaled' to 'oxygen enters the blood', the system found 2 causal paths.
Path 1 shows that reduced oxygen intake directly decreases the amount reaching the lungs,
which subsequently reduces the oxygen entering the bloodstream...

预测答案: less
置信度: high
推理过程: The causal analysis clearly shows that less oxygen inhaled leads to
decreased oxygen in the blood through a direct causal chain...

✓ 预测正确!
```

### 错误案例示例

```
问题: suppose more tank trucks added happens, how will it affect fewer stamps used.
正确答案: no_effect

起点: more tank trucks added
终点: stamps used
找到 0 条因果链

因果分析描述:
No complete causal path from 'more tank trucks added' to 'stamps used' was found.
During exploration, we discovered several relations about fuel transportation,
but none connected to mail or stamps...

预测答案: no_effect
置信度: high

✓ 预测正确!
```

## 总结

新版测试系统通过以下方式提升了因果推理能力：

1. ✅ **LLM 增强推理** - 更好地理解语义关系
2. ✅ **close_hits 桥接** - 提高因果链发现率
3. ✅ **可解释性** - 完整展示推理过程
4. ✅ **置信度评估** - 量化预测可靠性
5. ✅ **全面分析** - 多维度统计和案例研究

建议根据具体需求选择合适的测试版本，必要时可以结合使用。
