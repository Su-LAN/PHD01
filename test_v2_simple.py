# -*- coding: utf-8 -*-
"""
简化版测试脚本 - 使用更新后的 WIQACausalBuilder
演示新功能的使用
"""

from WIQACausalBuilder import WIQACausalBuilder
from datasets import load_dataset
import random

def test_single_sample(datapoint, verbose=True):
    """测试单个样本"""
    if verbose:
        print("="*80)
        print("问题:", datapoint['question_stem'])
        print("正确答案:", datapoint['answer_label'])
        print("="*80)

    # 1. 初始化
    wiqa = WIQACausalBuilder(datapoint)

    # 2. 提取起点和终点
    info = wiqa.extract_start_entity()
    if not info:
        print("❌ 无法提取起点/终点信息")
        return None

    start = info["cause_event"]
    target = info["outcome_base"]

    if verbose:
        print(f"\n起点: {start}")
        print(f"终点: {target}")

    # 3. BFS 扩展
    bfs = wiqa.expand_toward_target(
        start_X=start,
        target_Y=target,
        max_depth=5,
        max_relations_per_node=5
    )

    if verbose:
        print(f"\nBFS 扩展结果:")
        print(f"  - 发现三元组数量: {len(bfs.get('triples', []))}")
        print(f"  - 访问节点数量: {len(bfs.get('visited', []))}")
        print(f"  - 是否找到目标: {bfs.get('found_target')}")
        print(f"  - 搜索深度: {bfs.get('depth_reached')}")

    # 4. 基于 close_hits 的桥接（使用 bridge_close_hits，而不是直接强行接 RESULTS_IN Y）
    close_hits = bfs.get("close_hits", [])
    if close_hits:
        if verbose:
            print(f"\n发现 {len(close_hits)} 个 close_hits 节点")
        triples_with_bridge = wiqa.bridge_close_hits(
            triples=bfs["triples"],
            close_hits=close_hits,
            Y=target,
            max_bridge_nodes=3,
        )
    else:
        triples_with_bridge = bfs["triples"]

    # 5. 提取因果链
    chain_result = wiqa.get_causal_chain(
        triples_with_bridge,
        start_X=start,
        target_Y=target
    )

    if verbose:
        print(f"\n找到 {chain_result['num_paths']} 条因果链")

    # 6. 转化为描述性文字
    description = wiqa.causal_chain_to_text(chain_result, bfs)

    if verbose:
        print("\n" + "-"*80)
        print("因果分析描述:")
        print("-"*80)
        print(description)

    # 7. 使用 Python + 描述进行推理（effect_on_base 由路径符号决定）
    reasoning_result = wiqa.reason_with_description(description, chain_result=chain_result)

    predicted = reasoning_result['predicted_answer']
    confidence = reasoning_result['confidence']
    reasoning = reasoning_result['reasoning']

    if verbose:
        print("\n" + "-"*80)
        print("推理结果:")
        print("-"*80)
        print(f"预测答案: {predicted}")
        print(f"置信度: {confidence}")
        print(f"\n推理过程:\n{reasoning}")

    # 8. 验证
    is_correct = predicted == datapoint['answer_label']

    if verbose:
        print("\n" + "="*80)
        if is_correct:
            print("✓ 预测正确!")
        else:
            print(f"✗ 预测错误! (预测: {predicted}, 实际: {datapoint['answer_label']})")
        print("="*80)

    return {
        'correct': is_correct,
        'predicted': predicted,
        'actual': datapoint['answer_label'],
        'confidence': confidence,
        'num_paths': chain_result['num_paths'],
        'num_close_hits': len(close_hits)
    }


def test_multiple_samples(dataset, num_samples=10):
    """测试多个样本"""
    print(f"\n{'='*80}")
    print(f"测试 {num_samples} 个样本")
    print(f"{'='*80}\n")

    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    results = []

    for i, idx in enumerate(sample_indices):
        print(f"\n样本 {i+1}/{num_samples}")
        result = test_single_sample(dataset[idx], verbose=False)
        if result:
            results.append(result)
            status = "✓" if result['correct'] else "✗"
            print(f"{status} 预测: {result['predicted']}, 实际: {result['actual']}, 置信度: {result['confidence']}")

    # 统计
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / len(results) if results else 0

    print(f"\n{'='*80}")
    print("测试结果:")
    print(f"{'='*80}")
    print(f"总样本数: {len(results)}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.2%}")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    print("\n加载数据集...")
    ds = load_dataset('allenai/wiqa', split="validation", trust_remote_code=True)
    print(f"数据集大小: {len(ds)}")

    # 设置随机种子
    random.seed(42)

    # 选择测试模式
    print("\n选择测试模式:")
    print("1. 单样本详细测试")
    print("2. 多样本快速测试 (10 个)")
    print("3. 多样本快速测试 (100 个)")

    choice = input("\n请输入选项 (1/2/3，默认 1): ").strip() or "1"

    if choice == "1":
        # 单样本测试
        idx = random.randint(0, len(ds) - 1)
        print(f"\n随机选择样本索引: {idx}")
        test_single_sample(ds[idx], verbose=True)

    elif choice == "2":
        # 10 个样本测试
        test_multiple_samples(ds, num_samples=10)

    elif choice == "3":
        # 100 个样本测试
        test_multiple_samples(ds, num_samples=100)

    else:
        print("无效选项！")
