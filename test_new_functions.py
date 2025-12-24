# -*- coding: utf-8 -*-
"""
测试 WIQACausalBuilder 的新功能：
1. causal_chain_to_text: 将因果链转化为描述性文字
2. reason_with_description: 使用描述性文字进行推理判断
"""

from WIQACausalBuilder import WIQACausalBuilder

def test_with_causal_chain():
    """测试有因果链的情况"""
    print("="*80)
    print("测试场景 1: 有因果链的情况")
    print("="*80)

    datapoint = {
        'question_stem': 'suppose the seal is sealed less securely happens, how will it affect HURTING the letter to be mailed.',
        'question_para_step': 's',
        'answer_label': 'more',
        'answer_label_as_choice': 'A',
        'choices': {'text': ['more', 'less', 'no effect'], 'label': ['A', 'B', 'C']},
        'metadata_question_id': 'influence_graph:485:1733:77#1',
        'metadata_graph_id': '1733',
        'metadata_para_id': '485',
        'metadata_question_type': 'INPARA_EFFECT',
        'metadata_path_len': 3
    }

    wiqa = WIQACausalBuilder(datapoint)

    # 提取起点和终点
    info = wiqa.extract_start_entity()
    start = info["cause_event"]
    target = info["outcome_base"]

    print(f"\n问题: {datapoint['question_stem']}")
    print(f"起点: {start}")
    print(f"终点: {target}")
    print(f"正确答案: {datapoint['answer_label']}")

    # BFS 扩展
    bfs = wiqa.expand_toward_target(start_X=start, target_Y=target, max_depth=3, max_relations_per_node=3)

    # 提取因果链
    chain_result = wiqa.get_causal_chain(bfs["triples"], start_X=start, target_Y=target)

    print(f"\n找到 {chain_result['num_paths']} 条因果链")

    # 转化为描述性文字
    description = wiqa.causal_chain_to_text(chain_result, bfs)
    print("\n" + "-"*80)
    print("因果分析描述:")
    print("-"*80)
    print(description)

    # 使用描述 + 结构化路径进行推理（effect_on_base 由 Python 计算）
    reasoning_result = wiqa.reason_with_description(description, chain_result=chain_result)
    print("\n" + "-"*80)
    print("推理结果:")
    print("-"*80)
    print(f"预测答案: {reasoning_result['predicted_answer']}")
    print(f"预测选项: {reasoning_result['predicted_choice']}")
    print(f"置信度: {reasoning_result['confidence']}")
    print(f"\n推理过程:\n{reasoning_result['reasoning']}")

    # 验证
    is_correct = reasoning_result['predicted_answer'] == datapoint['answer_label']
    print("\n" + "="*80)
    if is_correct:
        print("✓ 预测正确!")
    else:
        print(f"✗ 预测错误 (预测: {reasoning_result['predicted_answer']}, 实际: {datapoint['answer_label']})")
    print("="*80)

    return is_correct


def test_without_causal_chain():
    """测试没有因果链的情况（no_effect）"""
    print("\n\n" + "="*80)
    print("测试场景 2: 没有因果链的情况")
    print("="*80)

    datapoint = {
        'question_stem': 'suppose more tank trucks have been added to the pick up gasoline list happens, how will it affect fewer stamps are used.',
        'question_para_step': ['You gather a writing utensil and a piece of paper',
                               'You use the writing utensil to compose your letter on the paper',
                               'You fold the paper',
                               'Place it into an envelope',
                               'You write the address of the recipient on the outside of the envelope',
                               'You seal the envelope with moisture or tape',
                               'You apply a stamp to the outside of the envelope',
                               'You put the envelope in a mailbox for a letter carrier to collect',
                               'The letter carrier takes the envelope to the correct address.'],
        'answer_label': 'no_effect',
        'answer_label_as_choice': 'C',
        'choices': {'text': ['more', 'less', 'no effect'], 'label': ['A', 'B', 'C']},
        'metadata_question_id': 'out_of_para:706:1530:122',
        'metadata_graph_id': '1530',
        'metadata_para_id': '706',
        'metadata_question_type': 'OUTOFPARA_DISTRACTOR',
        'metadata_path_len': 0
    }

    wiqa = WIQACausalBuilder(datapoint)

    # 提取起点和终点
    info = wiqa.extract_start_entity()
    start = info["cause_event"]
    target = info["outcome_base"]

    print(f"\n问题: {datapoint['question_stem']}")
    print(f"起点: {start}")
    print(f"终点: {target}")
    print(f"正确答案: {datapoint['answer_label']}")

    # BFS 扩展
    bfs = wiqa.expand_toward_target(start_X=start, target_Y=target, max_depth=3, max_relations_per_node=3)

    # 提取因果链
    chain_result = wiqa.get_causal_chain(bfs["triples"], start_X=start, target_Y=target)

    print(f"\n找到 {chain_result['num_paths']} 条因果链")

    # 转化为描述性文字（应该使用 BFS 结果）
    description = wiqa.causal_chain_to_text(chain_result, bfs)
    print("\n" + "-"*80)
    print("因果分析描述:")
    print("-"*80)
    print(description)

    # 使用描述 + 结构化路径进行推理（effect_on_base 由 Python 计算）
    reasoning_result = wiqa.reason_with_description(description, chain_result=chain_result)
    print("\n" + "-"*80)
    print("推理结果:")
    print("-"*80)
    print(f"预测答案: {reasoning_result['predicted_answer']}")
    print(f"预测选项: {reasoning_result['predicted_choice']}")
    print(f"置信度: {reasoning_result['confidence']}")
    print(f"\n推理过程:\n{reasoning_result['reasoning']}")

    # 验证
    is_correct = reasoning_result['predicted_answer'] == datapoint['answer_label']
    print("\n" + "="*80)
    if is_correct:
        print("✓ 预测正确!")
    else:
        print(f"✗ 预测错误 (预测: {reasoning_result['predicted_answer']}, 实际: {datapoint['answer_label']})")
    print("="*80)

    return is_correct


if __name__ == "__main__":
    print("\n" + "="*80)
    print("WIQACausalBuilder 新功能测试")
    print("="*80)

    results = []

    # 测试场景 1
    results.append(test_with_causal_chain())

    # 测试场景 2
    results.append(test_without_causal_chain())

    # 总结
    print("\n\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"总测试数: {len(results)}")
    print(f"通过数: {sum(results)}")
    print(f"失败数: {len(results) - sum(results)}")
    print(f"准确率: {sum(results)/len(results)*100:.1f}%")
    print("="*80)
