"""
优化版本的 100 样本测试脚本
- 减少控制台输出（verbose=False）
- 添加进度条
- 自动保存结果到 JSON 文件
"""

import os
import json
import random
import pandas as pd
from datetime import datetime
from collections import defaultdict, Counter
import time
from WIQACausalBuilder import WIQACausalBuilder
from datasets import load_dataset
from tqdm import tqdm

# 设置随机种子
random.seed(42)

# 加载数据集
print("加载数据集...")
ds = load_dataset('allenai/wiqa', split="validation", trust_remote_code=True)
print(f"数据集大小: {len(ds)}")

def test_single_sample(datapoint, sample_idx, verbose=False):
    """测试单个样本（简化版，减少输出）"""
    result = {
        'sample_idx': sample_idx,
        'question': datapoint.get('question_stem', ''),
        'true_label': datapoint.get('answer_label', ''),
        'status': 'unknown',
        'predicted_label': None,
        'num_paths': 0,
        'shortest_path_length': None,
        'error_message': ''
    }

    try:
        wiqa = WIQACausalBuilder(datapoint)

        # 步骤 1: 提取起点和终点
        info = wiqa.extract_start_entity()
        if not info:
            result['status'] = 'extraction_error'
            result['error_message'] = '无法提取起点/终点信息'
            return result

        start = info["cause_event"]
        target = info["outcome_base"]

        # 步骤 2: BFS 因果图扩展
        bfs = wiqa.expand_toward_target(
            start_X=start,
            target_Y=target,
            max_depth=5,
            max_relations_per_node=5
        )

        result['num_triples'] = len(bfs['triples'])
        result['num_close_hits'] = len(bfs['close_hits'])

        # 步骤 3: 桥接 + 提取因果链
        if bfs["close_hits"]:
            triples_with_bridges = wiqa.bridge_close_hits(
                triples=bfs["triples"],
                close_hits=bfs["close_hits"],
                Y=target,
                max_bridge_nodes=3,
            )
        else:
            triples_with_bridges = bfs["triples"]

        chain_result = wiqa.get_causal_chain(triples_with_bridges, start_X=start, target_Y=target)
        result['num_paths'] = chain_result['num_paths']
        result['shortest_path_length'] = chain_result['shortest_path_length']

        # 步骤 4: 生成描述
        description = wiqa.causal_chain_to_text(chain_result, bfs)

        # 步骤 5: 推理
        reasoning_result = wiqa.reason_with_description(description, chain_result=chain_result)

        result['predicted_label'] = reasoning_result['predicted_answer']
        result['effect_on_base'] = reasoning_result.get('effect_on_base', '')
        result['reasoning'] = reasoning_result['reasoning']

        # 验证结果
        if reasoning_result['predicted_answer'] == datapoint['answer_label']:
            result['status'] = 'correct'
        else:
            result['status'] = 'wrong'

    except Exception as e:
        result['status'] = 'error'
        result['error_message'] = str(e)

    return result


# ============================================================================
# 主测试流程
# ============================================================================

print("\n" + "="*80)
print("WIQACausalBuilder 测试 - 100个样本（优化版）")
print("="*80)
print(f"测试样本数: 100")
print(f"最大深度: 5")
print(f"每节点最大关系数: 5")
print()

# 随机选择100个样本
sample_indices = random.sample(range(len(ds)), 100)

results = {
    'test_info': {
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'num_samples': 100,
        'dataset_size': len(ds),
        'max_depth': 5,
        'max_relations_per_node': 5,
        'random_seed': 42,
        'modifications': 'Using ONLY shortest paths with weighted voting (^1.5 penalty) + Counterfactual substitution check'
    },
    'summary': {
        'total': 0,
        'correct': 0,
        'wrong': 0,
        'error': 0,
        'extraction_error': 0,
        'no_path': 0,
        'with_close_hits': 0,
        'accuracy': 0.0
    },
    'detailed_results': []
}

start_time = time.time()

# 使用进度条
print("\n开始测试...")
for sample_idx in tqdm(sample_indices, desc="测试进度", ncols=100):
    datapoint = ds[sample_idx]

    # verbose=False 减少输出
    result = test_single_sample(datapoint, sample_idx, verbose=False)
    results['detailed_results'].append(result)

    # 更新统计
    results['summary']['total'] += 1

    if result['status'] == 'correct':
        results['summary']['correct'] += 1
    elif result['status'] == 'wrong':
        results['summary']['wrong'] += 1
    elif result['status'] == 'error':
        results['summary']['error'] += 1
    elif result['status'] == 'extraction_error':
        results['summary']['extraction_error'] += 1

    if result['num_paths'] == 0:
        results['summary']['no_path'] += 1

    if result.get('num_close_hits', 0) > 0:
        results['summary']['with_close_hits'] += 1

end_time = time.time()
total_time = end_time - start_time

# 计算准确率
if results['summary']['total'] > 0:
    results['summary']['accuracy'] = results['summary']['correct'] / results['summary']['total']

results['test_info']['total_time'] = total_time
results['test_info']['avg_time_per_sample'] = total_time / results['summary']['total']

# ============================================================================
# 保存结果到文件
# ============================================================================

timestamp = results['test_info']['timestamp']
output_file = f"causal_builder_results_{timestamp}.json"

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✓ 结果已保存到: {output_file}")

# ============================================================================
# 打印总结
# ============================================================================

print("\n" + "="*80)
print("测试完成!")
print("="*80)
print(f"\n总耗时: {total_time:.2f} 秒")
print(f"平均每样本: {total_time / results['summary']['total']:.2f} 秒")
print(f"\n总样本数: {results['summary']['total']}")
print(f"正确预测: {results['summary']['correct']}")
print(f"错误预测: {results['summary']['wrong']}")
print(f"处理错误: {results['summary']['error']}")
print(f"提取错误: {results['summary']['extraction_error']}")
print(f"未找到路径: {results['summary']['no_path']}")
print(f"有 close_hits 的样本: {results['summary']['with_close_hits']}")
print(f"\n准确率: {results['summary']['accuracy']:.2%}")

# ============================================================================
# 详细统计分析
# ============================================================================

print("\n" + "="*80)
print("详细统计分析")
print("="*80)

# 按标签统计
print("\n" + "-"*80)
print("按真实标签统计:")
print("-"*80)

label_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'wrong': 0})

for result in results['detailed_results']:
    if result['status'] in ['correct', 'wrong']:
        true_label = result['true_label']
        label_stats[true_label]['total'] += 1
        if result['status'] == 'correct':
            label_stats[true_label]['correct'] += 1
        else:
            label_stats[true_label]['wrong'] += 1

for label in ['more', 'less', 'no_effect']:
    if label in label_stats:
        stats = label_stats[label]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"\n{label.upper()}:")
        print(f"  总数: {stats['total']}")
        print(f"  正确: {stats['correct']}")
        print(f"  错误: {stats['wrong']}")
        print(f"  准确率: {acc:.2%}")

# 混淆矩阵
print("\n" + "-"*80)
print("混淆矩阵:")
print("-"*80)

confusion = defaultdict(Counter)

for result in results['detailed_results']:
    if result['status'] in ['correct', 'wrong']:
        true_label = result['true_label']
        pred_label = result['predicted_label']
        confusion[true_label][pred_label] += 1

header = "真实\\预测"
print(f"\n{header:<15} {'more':<10} {'less':<10} {'no_effect':<10}")
print("-" * 50)

for true_label in ['more', 'less', 'no_effect']:
    if true_label in confusion:
        counts = confusion[true_label]
        print(f"{true_label:<15} {counts['more']:<10} {counts['less']:<10} {counts['no_effect']:<10}")

# 路径长度分析
print("\n" + "-"*80)
print("路径长度分析:")
print("-"*80)

path_length_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

for result in results['detailed_results']:
    if result['status'] in ['correct', 'wrong'] and result['shortest_path_length'] is not None:
        length = result['shortest_path_length']
        path_length_stats[length]['total'] += 1
        if result['status'] == 'correct':
            path_length_stats[length]['correct'] += 1

for length in sorted(path_length_stats.keys()):
    stats = path_length_stats[length]
    acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
    print(f"\n长度 {length}:")
    print(f"  总数: {stats['total']}")
    print(f"  正确: {stats['correct']}")
    print(f"  准确率: {acc:.2%}")

# 保存统计摘要
summary_file = f"causal_builder_summary_{timestamp}.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("WIQACausalBuilder 测试结果摘要\n")
    f.write("="*80 + "\n\n")
    f.write(f"时间戳: {timestamp}\n")
    f.write(f"总样本数: {results['summary']['total']}\n")
    f.write(f"准确率: {results['summary']['accuracy']:.2%}\n")
    f.write(f"总耗时: {total_time:.2f} 秒\n\n")
    f.write("-"*80 + "\n")
    f.write("按标签统计:\n")
    f.write("-"*80 + "\n")
    for label in ['more', 'less', 'no_effect']:
        if label in label_stats:
            stats = label_stats[label]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            f.write(f"\n{label.upper()}:\n")
            f.write(f"  总数: {stats['total']}\n")
            f.write(f"  正确: {stats['correct']}\n")
            f.write(f"  准确率: {acc:.2%}\n")

print(f"\n✓ 摘要已保存到: {summary_file}")
print("\n测试完成！")
