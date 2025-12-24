# -*- coding: utf-8 -*-
"""
WIQA Reasoning Module - Usage Examples
演示如何使用推理模块处理单个问题和整个数据集
"""

from WIQACausalBuilder import WIQAReasoningModule, process_question, process_dataset

print("=" * 80)
print("WIQA 推理模块使用示例")
print("=" * 80)

# ============================================================================
# 示例 1: 处理单个 question_stem
# ============================================================================
print("\n[示例 1] 处理单个 question_stem")
print("-" * 80)

# 只需要提供 question_stem，返回完整的推理字典
question_stem = "suppose less gasoline is loaded onto tank trucks happens, how will it affect LESS electricity being produced."

# 使用便捷函数
result = process_question(question_stem, answer_label='no_effect')

print(f"问题 (question): {result['question']}")
print(f"\n选项 (choices): {result['choices']}")
print(f"\n答案 (answer): {result['answer']}")
print(f"\n事件对 (event_pair):")
print(f"  - 扰动事件 (cause): {result['event_pair']['cause']}")
print(f"  - 结果事件 (outcome): {result['event_pair']['outcome']}")
print(f"  - 方向 (direction): {result['event_pair']['direction']}")
print(f"\n影响图 (influence_graph): {result['influence_graph']}")
print(f"过程步骤 (para_steps): {result['para_steps']}")
print(f"元数据 (metadata): {result['metadata']}")


# ============================================================================
# 示例 2: 处理单个数据集样本 (带完整信息)
# ============================================================================
print("\n\n[示例 2] 处理单个数据集样本")
print("-" * 80)

# 创建推理模块实例
module = WIQAReasoningModule()

# 模拟一个数据集样本
dataset_item = {
    'question_stem': 'suppose there will be fewer new trees happens, how will it affect LESS forest formation.',
    'answer_label': 'more',
    'metadata_graph_id': '144',
    'metadata_para_id': '1217',
    'metadata_question_id': 'influence_graph:1217:144:106#0',
    'metadata_question_type': 'INPARA_EFFECT',
    'metadata_path_len': 2,
    'question_para_step': [
        'A tree produces seeds',
        'The seeds are dispersed by wind, animals, etc',
        'The seeds reach the ground',
        'Grow into new trees',
        'These new trees produce seeds',
        'The process repeats itself over and over'
    ],
    'choices': {'text': ['more', 'less', 'no effect'], 'label': ['A', 'B', 'C']}
}

# 处理数据集样本
result = module.process_dataset_item(dataset_item)

print(f"问题: {result['question']}")
print(f"\n答案: {result['answer']}")
print(f"\n事件对:")
print(f"  Cause: {result['event_pair']['cause']}")
print(f"  Outcome: {result['event_pair']['outcome']}")
print(f"  Direction: {result['event_pair']['direction']}")
print(f"\n过程步骤 ({len(result['para_steps'])} 步):")
for i, step in enumerate(result['para_steps'], 1):
    print(f"  {i}. {step}")
print(f"\n元数据:")
print(f"  Question ID: {result['metadata']['question_id']}")
print(f"  Graph ID: {result['metadata']['graph_id']}")
print(f"  Question Type: {result['metadata']['question_type']}")


# ============================================================================
# 示例 3: 批量处理多个问题
# ============================================================================
print("\n\n[示例 3] 批量处理多个问题")
print("-" * 80)

# 准备多个数据集样本
dataset = [
    {
        'question_stem': 'suppose more pollution in the environment happens, how will it affect more cocoon hatches.',
        'answer_label': 'less',
        'metadata_graph_id': '1484',
        'metadata_question_type': 'EXOGENOUS_EFFECT',
        'question_para_step': ['A larva is born', 'The caterpillar eats constantly', 'The caterpillar spins a cocoon'],
    },
    {
        'question_stem': 'suppose there is more hydrogen in the star happens, how will it affect a LARGER white dwarf star.',
        'answer_label': 'more',
        'metadata_graph_id': '801',
        'metadata_question_type': 'INPARA_EFFECT',
        'question_para_step': ['A star burns in space', 'Eventually it burns all its hydrogen', 'Becomes a red giant'],
    },
    {
        'question_stem': 'suppose less water is added to the tray happens, how will it affect LESS ice cubes being made.',
        'answer_label': 'less',
        'metadata_graph_id': '2064',
        'metadata_question_type': 'EXOGENOUS_EFFECT',
        'question_para_step': ['You take a tray', 'Add water into the tray', 'Insert into freezer'],
    }
]

# 批量处理
results = process_dataset(dataset)

print(f"成功处理 {len(results)} 个问题\n")

for i, result in enumerate(results, 1):
    print(f"[问题 {i}]")
    print(f"  Question: {result['question'][:60]}...")
    print(f"  Answer: {result['answer']}")
    print(f"  Event Pair: {result['event_pair']['cause']} -> {result['event_pair']['outcome']}")
    print(f"  Direction: {result['event_pair']['direction']}")
    print(f"  Steps: {len(result['para_steps'])}")
    print()


# ============================================================================
# 示例 4: 使用 HuggingFace WIQA 数据集
# ============================================================================
print("\n[示例 4] 使用 HuggingFace WIQA 数据集")
print("-" * 80)

try:
    from datasets import load_dataset

    # 加载 WIQA 验证集的前 5 个样本
    print("正在加载 WIQA 数据集...")
    ds = load_dataset('allenai/wiqa', split='validation[:5]')

    # 处理数据集
    results = process_dataset(ds)

    print(f"成功处理 {len(results)} 个样本\n")

    for i, result in enumerate(results, 1):
        print(f"样本 {i}:")
        print(f"  Question: {result['question'][:70]}...")
        print(f"  Answer: {result['answer']}")

        if result['event_pair']:
            print(f"  Cause: {result['event_pair']['cause'][:50]}...")
            print(f"  Outcome: {result['event_pair']['outcome'][:50]}...")
            print(f"  Direction: {result['event_pair']['direction']}")

        print(f"  Question Type: {result['metadata'].get('question_type', 'N/A')}")
        print()

except ImportError:
    print("HuggingFace datasets 未安装，跳过此示例")
    print("安装命令: pip install datasets")
except Exception as e:
    print(f"加载数据集时出错: {e}")


# ============================================================================
# 示例 5: 获取特定字段用于推理
# ============================================================================
print("\n[示例 5] 提取推理所需的关键信息")
print("-" * 80)

question = "suppose more sunny days happens, how will it affect less sugar and oxygen produced."
result = process_question(question, answer_label='less')

# 提取推理所需的关键信息
reasoning_input = {
    'cause': result['event_pair']['cause'],
    'outcome': result['event_pair']['outcome'],
    'direction': result['event_pair']['direction'],
    'question': result['question'],
    'choices': result['choices'],
}

print("推理输入:")
print(f"  扰动: {reasoning_input['cause']}")
print(f"  结果: {reasoning_input['outcome']} ({reasoning_input['direction']})")
print(f"  问题: {reasoning_input['question']}")
print(f"  选项: {reasoning_input['choices']}")

# 这个字典可以直接传给你的推理模型
print("\n可以用于模型推理:")
print(reasoning_input)


print("\n" + "=" * 80)
print("所有示例完成!")
print("=" * 80)


# ============================================================================
# 总结：主要 API
# ============================================================================
print("\n\n主要 API 总结:")
print("=" * 80)
print("""
1. process_question(question_stem, answer_label=None, graph_id=None)
   - 快速处理单个问题
   - 返回: 推理字典

2. process_dataset(dataset, limit=None)
   - 快速处理整个数据集
   - 返回: 推理字典列表

3. WIQAReasoningModule 类:
   - process_single_question(): 处理单个问题（带更多选项）
   - process_dataset_item(): 处理数据集样本
   - process_dataset(): 处理整个数据集

返回的字典结构:
{
    'question': str,              # 问题文本
    'choices': dict,              # 选项 {'text': [...], 'label': [...]}
    'influence_graph': dict/None, # 影响图（如果有 graph_id）
    'answer': str,                # 答案标签
    'event_pair': dict,           # 事件对 {cause, outcome, direction, answer_label}
    'para_steps': list,           # 过程步骤
    'metadata': dict              # 元数据
}
""")
