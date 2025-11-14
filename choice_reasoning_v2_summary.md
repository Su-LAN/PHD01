# Choice Reasoning V2 - 设计总结

## 改进后的流程

### 整体流程：
1. **解析问题结构** (parse question structure)
2. **识别观察对象** (identify observable outcome) - 新增！
3. **构建因果链** (build causal chains) - 改进！使用完整的链路径
4. **对每个choice进行双重推理** (for each choice) - 重大改进！
   - 4.1 **正向推理** (forward reasoning) - 独立LLM调用
   - 4.2 **反事实推理** (counterfactual reasoning) - 独立LLM调用
5. **综合决策** (final decision)

## 关键改进点

### 1. 提前识别观察对象
**位置**：在获取seed和构建因果链之后，在choice评估之前

**作用**：
- 明确"MORE vegetables"到底指什么（吃/种/买）
- 基于干预上下文选择最合理的解释
- 所有choice共享同一个观察对象解释

### 2. 提供完整的因果链
**改进前**：只给三元组列表
```
- stomach bug → gastric acid (increases)
- gastric acid → pH (decreases)
- pH → digestion (affects)
```

**改进后**：给完整的因果路径
```
Chain 1: stomach bug --[increases]--> gastric acid (0.85) → gastric acid --[decreases]--> pH (0.75) → pH --[affects]--> digestion (0.60)
Chain 2: stomach bug --[causes]--> intestinal inflammation (0.80) → ...
```

### 3. 每个choice两次独立的LLM调用

**第一次：正向推理**
- 假设这个choice是正确的
- 寻找支持证据
- 评估因果路径强度
- 给出初始置信度

**第二次：反事实推理**
- 独立的LLM调用（不共享上下文）
- 测试：如果干预没发生，结果会如何
- 验证因果关系的必要性
- 调整置信度（+/- 0.3）

### 4. 综合决策考虑多个因素
- 反事实测试结果（pass/fail/unclear）
- 调整后的最终置信度
- 因果路径强度
- 特殊规则（no_effect的处理）

## 示例

**问题**: suppose Having a stomach bug happens, how will it affect MORE vegetables.

**Step 1**: 识别观察对象
→ "eating more vegetables" (因为胃病与饮食相关)

**Step 2**: 构建因果链
→ Chain 1: stomach bug → gastric acid ↑ → pH ↓ → digestion difficulty
→ Chain 2: stomach bug → appetite ↓ → food intake ↓

**Step 3**: 对"more"选项
- Forward: 找不到支持证据，confidence=0.2
- Counterfactual: 没有胃病→也不会多吃蔬菜，test_result=fail

**Step 4**: 对"no_effect"选项
- Forward: 胃病与吃蔬菜数量无直接因果，confidence=0.7
- Counterfactual: 没有胃病→吃蔬菜数量不变，test_result=pass

**Final**: no_effect (confidence=0.8)

## 优势

1. **更准确的观察对象识别**：避免歧义
2. **更完整的因果证据**：链而非碎片
3. **更严格的因果验证**：反事实测试
4. **更独立的推理过程**：两次调用不互相影响
5. **更合理的no_effect处理**：给予公平机会

