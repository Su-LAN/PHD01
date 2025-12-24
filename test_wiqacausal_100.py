# -*- coding: utf-8 -*-
"""
测试 WIQACausalBuilder - 100个样本
按照 WIQACausalBuilder.main() 的格式进行详细测试
"""

import json
import random
import time
from datetime import datetime
from collections import defaultdict, Counter
from datasets import load_dataset
from WIQACausalBuilder import WIQACausalBuilder

# 设置随机种子以保证可重复性
random.seed(42)


def test_single_sample(datapoint, sample_idx, verbose=True):
    """
    测试单个样本，遵循 WIQACausalBuilder.main() 的格式

    参数:
        datapoint: WIQA 数据点
        sample_idx: 样本索引
        verbose: 是否打印详细信息

    返回:
        result: 包含详细测试结果的字典
    """
    result = {
        'sample_idx': sample_idx,
        'question': datapoint.get('question_stem', ''),
        'true_label': datapoint.get('answer_label', ''),
        'true_choice': datapoint.get('answer_label_as_choice', ''),
        'status': 'unknown',
        'predicted_label': None,
        'predicted_choice': None,
        'reasoning': '',
        'confidence': '',
        'num_paths': 0,
        'num_triples': 0,
        'num_visited': 0,
        'num_close_hits': 0,
        'depth_reached': 0,
        'found_target': False,
        'start': '',
        'target': '',
        'outcome_direction': '',
        'outcome_is_negated': False,
        'description': '',
        'error_message': ''
    }

    try:
        if verbose:
            print("\n" + "="*80)
            print(f"样本 {sample_idx}")
            print("="*80)
            print(f"问题: {datapoint['question_stem']}")
            print(f"正确答案: {datapoint['answer_label']} ({datapoint.get('answer_label_as_choice', '')})")

        # ========== 步骤 1: 提取起点和终点 ==========
        if verbose:
            print("\n" + "-"*80)
            print("步骤 1: 提取起点和终点")
            print("-"*80)

        wiqa = WIQACausalBuilder(datapoint)
        info = wiqa.extract_start_entity()

        if not info:
            result['status'] = 'extraction_error'
            result['error_message'] = '无法提取起点/终点信息'
            if verbose:
                print("❌ 无法提取起点/终点信息")
            return result

        start = info["cause_event"]
        target = info["outcome_base"]

        result['start'] = start
        result['target'] = target
        result['outcome_direction'] = info["outcome_direction_in_question"]
        result['outcome_is_negated'] = info.get("outcome_is_negated", False)

        if verbose:
            print(f"  - 原因事件 (cause_event): '{start}'")
            print(f"  - 结果事件 (outcome_event): '{info['outcome_event']}'")
            print(f"  - 结果基底 (outcome_base): '{target}'")
            print(f"  - 方向词 (outcome_direction_in_question): {info['outcome_direction_in_question']}")
            print(f"  - 是否否定 (outcome_is_negated): {info.get('outcome_is_negated', False)}")

        # ========== 步骤 2: BFS 因果图扩展 ==========
        if verbose:
            print("\n" + "-"*80)
            print("步骤 2: BFS 因果图扩展")
            print("-"*80)

        bfs = wiqa.expand_toward_target(
            start_X=start,
            target_Y=target,
            max_depth=5,
            max_relations_per_node=5
        )

        result['num_triples'] = len(bfs['triples'])
        result['num_visited'] = len(bfs['visited'])
        result['found_target'] = bfs['found_target']
        result['depth_reached'] = bfs['depth_reached']
        result['num_close_hits'] = len(bfs['close_hits'])

        if verbose:
            print(f"  - 发现三元组数量: {len(bfs['triples'])}")
            print(f"  - 访问节点数量: {len(bfs['visited'])}")
            print(f"  - 是否找到目标: {bfs['found_target']}")
            print(f"  - 搜索深度: {bfs['depth_reached']}")
            print(f"  - Close hits 数量: {len(bfs['close_hits'])}")

            if bfs['close_hits'] and verbose:
                print(f"\n  Close Hits 详情:")
                for i, hit in enumerate(bfs['close_hits'][:3], 1):  # 只显示前3个
                    print(f"    {i}. 节点: '{hit['node']}' (深度: {hit['depth']})")

        # ========== 步骤 3: 基于 close hits 的桥接 + 提取因果链 ==========
        if verbose:
            print("\n" + "-"*80)
            print("步骤 3: 基于 close hits 的桥接 + 提取因果链")
            print("-"*80)

        if bfs["close_hits"]:
            triples_with_bridges = wiqa.bridge_close_hits(
                triples=bfs["triples"],
                close_hits=bfs["close_hits"],
                Y=target,
                max_bridge_nodes=3,
            )
            added_bridges = len(triples_with_bridges) - len(bfs["triples"])
            if verbose:
                print(f"  通过 bridge_close_hits 添加了 {added_bridges} 条桥接边。")
        else:
            triples_with_bridges = bfs["triples"]
            if verbose:
                print("  无 close hits，可用于桥接的节点为空。")

        chain_result = wiqa.get_causal_chain(triples_with_bridges, start_X=start, target_Y=target)

        result['num_paths'] = chain_result['num_paths']

        if verbose:
            print(f"\n因果链提取结果:")
            print(f"  - 找到路径数量: {chain_result['num_paths']}")
            if chain_result['shortest_path_length']:
                print(f"  - 最短路径长度: {chain_result['shortest_path_length']}")
            print(f"  - 涉及节点数: {len(chain_result['all_nodes_in_paths'])}")

            # 打印所有路径（前3条）
            if chain_result['num_paths'] > 0:
                print(f"\n  因果路径（前3条）:")
                for i, path in enumerate(chain_result["paths"][:3], 1):
                    print(f"\n  路径 {i} (长度 {len(path)}):")
                    for e in path:
                        print(f"    {e['head']} --[{e['relation']}]--> {e['tail']}")

        # ========== 步骤 4: 生成文字描述 ==========
        if verbose:
            print("\n" + "-"*80)
            print("步骤 4: 生成因果分析描述")
            print("-"*80)

        description = wiqa.causal_chain_to_text(chain_result, bfs)
        result['description'] = description

        if verbose:
            print(f"\n自然语言描述（前300字符）:")
            print("-" * 80)
            print(description[:300])
            if len(description) > 300:
                print("...")
            print("-" * 80)

        # ========== 步骤 5: LLM 推理 ==========
        if verbose:
            print("\n" + "-"*80)
            print("步骤 5: 基于描述的 LLM 推理")
            print("-"*80)

        reasoning_result = wiqa.reason_with_description(description, chain_result=chain_result)

        result['predicted_label'] = reasoning_result['predicted_answer']
        result['predicted_choice'] = reasoning_result['predicted_choice']
        result['reasoning'] = reasoning_result['reasoning']
        result['confidence'] = reasoning_result['confidence']
        result['effect_on_base'] = reasoning_result.get('effect_on_base', '')

        if verbose:
            print(f"  - 对基础变量的影响 (effect_on_base): {reasoning_result.get('effect_on_base', 'N/A')}")
            print(f"  - 最终预测答案 (predicted_answer): {reasoning_result['predicted_answer']}")
            print(f"  - 预测选项 (predicted_choice): {reasoning_result['predicted_choice']}")
            print(f"  - 置信度 (confidence): {reasoning_result['confidence']}")

            print(f"\n推理过程（前200字符）:")
            print("-" * 80)
            print(reasoning_result['reasoning'][:200])
            if len(reasoning_result['reasoning']) > 200:
                print("...")
            print("-" * 80)

        # ========== 验证结果 ==========
        if reasoning_result['predicted_answer'] == datapoint['answer_label']:
            result['status'] = 'correct'
            if verbose:
                print("\n✓ 预测正确!")
        else:
            result['status'] = 'wrong'
            if verbose:
                print(f"\n✗ 预测错误!")
                print(f"  期望: {datapoint['answer_label']}")
                print(f"  得到: {reasoning_result['predicted_answer']}")

    except Exception as e:
        result['status'] = 'error'
        result['error_message'] = str(e)
        if verbose:
            print(f"\n❌ 处理样本时出错: {e}")
            import traceback
            traceback.print_exc()

    return result


def test_wiqa_dataset(num_samples=100, verbose=True):
    """
    测试 WIQA 数据集的多个样本

    参数:
        num_samples: 要测试的样本数量
        verbose: 是否打印详细信息

    返回:
        results: 包含所有测试结果的字典
    """
    print("="*80)
    print("WIQACausalBuilder 完整流程测试 - 100个样本")
    print("="*80)

    # 加载数据集
    print("\n加载 WIQA 数据集...")
    ds = load_dataset('allenai/wiqa', split="validation", trust_remote_code=True)
    print(f"数据集大小: {len(ds)}")
    print(f"计划测试样本数: {num_samples}")

    # 随机选择样本
    sample_indices = random.sample(range(len(ds)), min(num_samples, len(ds)))

    results = {
        'test_info': {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'num_samples': num_samples,
            'dataset_size': len(ds),
            'max_depth': 5,
            'max_relations_per_node': 5,
            'random_seed': 42
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

    for idx, sample_idx in enumerate(sample_indices):
        datapoint = ds[sample_idx]

        if verbose:
            print(f"\n\n{'#'*80}")
            print(f"进度: {idx+1}/{num_samples}")
            print(f"{'#'*80}")

        result = test_single_sample(datapoint, sample_idx, verbose=verbose)
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

        if result['num_close_hits'] > 0:
            results['summary']['with_close_hits'] += 1

    end_time = time.time()
    total_time = end_time - start_time

    # 计算准确率
    if results['summary']['total'] > 0:
        results['summary']['accuracy'] = results['summary']['correct'] / results['summary']['total']

    results['test_info']['total_time'] = total_time
    results['test_info']['avg_time_per_sample'] = total_time / results['summary']['total']

    # 打印最终总结
    print("\n\n" + "="*80)
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

    # 按标签统计
    print("\n" + "="*80)
    print("按真实标签统计:")
    print("="*80)

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

    # 按置信度统计
    print("\n" + "="*80)
    print("按置信度统计:")
    print("="*80)

    confidence_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'wrong': 0})

    for result in results['detailed_results']:
        if result['status'] in ['correct', 'wrong']:
            conf = result.get('confidence', 'unknown')
            confidence_stats[conf]['total'] += 1
            if result['status'] == 'correct':
                confidence_stats[conf]['correct'] += 1
            else:
                confidence_stats[conf]['wrong'] += 1

    for conf in ['high', 'medium', 'low', 'unknown']:
        if conf in confidence_stats:
            stats = confidence_stats[conf]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"\n{conf.upper()}:")
            print(f"  总数: {stats['total']}")
            print(f"  正确: {stats['correct']}")
            print(f"  错误: {stats['wrong']}")
            print(f"  准确率: {acc:.2%}")

    # 混淆矩阵
    print("\n" + "="*80)
    print("混淆矩阵:")
    print("="*80)

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

    return results


def save_results(results):
    """保存测试结果到文件"""
    timestamp = results['test_info']['timestamp']

    # 保存详细结果 JSON
    results_file = f"causal_builder_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {results_file}")

    # 保存简要报告
    summary_file = f"causal_builder_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WIQACausalBuilder 测试报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"测试时间: {timestamp}\n")
        f.write(f"总耗时: {results['test_info']['total_time']:.2f} 秒\n")
        f.write(f"平均每个样本: {results['test_info']['avg_time_per_sample']:.2f} 秒\n\n")
        f.write(f"总样本数: {results['summary']['total']}\n")
        f.write(f"正确预测: {results['summary']['correct']}\n")
        f.write(f"错误预测: {results['summary']['wrong']}\n")
        f.write(f"处理错误: {results['summary']['error']}\n")
        f.write(f"提取错误: {results['summary']['extraction_error']}\n")
        f.write(f"未找到路径: {results['summary']['no_path']}\n")
        f.write(f"有 close_hits 的样本: {results['summary']['with_close_hits']}\n\n")
        f.write(f"准确率: {results['summary']['accuracy']:.2%}\n\n")

        # 按标签统计
        f.write("="*80 + "\n")
        f.write("按标签统计:\n")
        f.write("="*80 + "\n")

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
                f.write(f"\n{label.upper()}:\n")
                f.write(f"  总数: {stats['total']}\n")
                f.write(f"  正确: {stats['correct']}\n")
                f.write(f"  错误: {stats['wrong']}\n")
                f.write(f"  准确率: {acc:.2%}\n")

        # 置信度统计
        f.write("\n" + "="*80 + "\n")
        f.write("按置信度统计:\n")
        f.write("="*80 + "\n")

        confidence_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'wrong': 0})

        for result in results['detailed_results']:
            if result['status'] in ['correct', 'wrong']:
                conf = result.get('confidence', 'unknown')
                confidence_stats[conf]['total'] += 1
                if result['status'] == 'correct':
                    confidence_stats[conf]['correct'] += 1
                else:
                    confidence_stats[conf]['wrong'] += 1

        for conf in ['high', 'medium', 'low', 'unknown']:
            if conf in confidence_stats:
                stats = confidence_stats[conf]
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                f.write(f"\n{conf.upper()}:\n")
                f.write(f"  总数: {stats['total']}\n")
                f.write(f"  正确: {stats['correct']}\n")
                f.write(f"  错误: {stats['wrong']}\n")
                f.write(f"  准确率: {acc:.2%}\n")

    print(f"简要报告已保存到: {summary_file}")


def main():
    """主函数"""
    # 测试100个样本（可以设置 verbose=False 来减少输出）
    results = test_wiqa_dataset(num_samples=100, verbose=True)

    # 保存结果
    save_results(results)

    print("\n" + "="*80)
    print("流程演示完成")
    print("="*80)


if __name__ == "__main__":
    main()
