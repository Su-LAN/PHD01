# -*- coding: utf-8 -*-
"""
WIQA Reasoning Module - Test Script
"""

from WIQACausalBuilder import process_question, process_dataset, WIQAReasoningModule

def test_all():
    print("="*70)
    print("WIQA Reasoning Module Test Suite")
    print("="*70)

    # Test 1: Single question
    print("\n[Test 1] Single Question Processing")
    print("-"*70)

    question = "suppose less gasoline is loaded onto tank trucks happens, how will it affect LESS electricity being produced."
    result = process_question(question, answer_label='no_effect')

    assert result['answer'] == 'no_effect'
    assert result['event_pair']['cause'] == 'less gasoline is loaded onto tank trucks'
    assert result['event_pair']['outcome'] == 'electricity being produced'
    assert result['event_pair']['direction'] == 'LESS'

    print("PASS: Question processed")
    print(f"  Cause: {result['event_pair']['cause']}")
    print(f"  Outcome: {result['event_pair']['outcome']}")
    print(f"  Direction: {result['event_pair']['direction']}")
    print(f"  Answer: {result['answer']}")

    # Test 2: Batch processing
    print("\n[Test 2] Batch Processing")
    print("-"*70)

    dataset = [
        {'question_stem': 'suppose more pollution happens, how will it affect MORE disease.', 'answer_label': 'more'},
        {'question_stem': 'suppose fewer trees exist, how will it affect LESS forest.', 'answer_label': 'more'},
    ]

    results = process_dataset(dataset)
    assert len(results) == 2

    print(f"PASS: Processed {len(results)} questions")
    for i, r in enumerate(results, 1):
        if r['event_pair']:
            print(f"  Q{i}: {r['event_pair']['cause'][:30]}... -> {r['answer']}")
        else:
            print(f"  Q{i}: No event pair extracted -> {r['answer']}")

    # Test 3: Direction extraction
    print("\n[Test 3] Direction Extraction")
    print("-"*70)

    test_cases = [
        ("suppose A happens, how will it affect MORE output.", "MORE"),
        ("suppose B happens, how will it affect LESS production.", "LESS"),
        ("suppose C happens, how will it affect normal operation.", "NORMAL"),
    ]

    for q, expected in test_cases:
        r = process_question(q)
        actual = r['event_pair']['direction']
        assert actual == expected
        print(f"PASS: {q[:40]}... -> {actual}")

    # Test 4: Dataset item processing
    print("\n[Test 4] Dataset Item Processing")
    print("-"*70)

    module = WIQAReasoningModule()
    item = {
        'question_stem': 'suppose fewer trees happens, how will it affect LESS forest formation.',
        'answer_label': 'more',
        'metadata_graph_id': '144',
        'metadata_question_type': 'INPARA_EFFECT',
        'question_para_step': ['Step 1', 'Step 2', 'Step 3'],
    }

    result = module.process_dataset_item(item)
    assert result['answer'] == 'more'
    assert len(result['para_steps']) == 3
    assert result['metadata']['question_type'] == 'INPARA_EFFECT'

    print("PASS: Dataset item processed")
    print(f"  Question Type: {result['metadata']['question_type']}")
    print(f"  Para Steps: {len(result['para_steps'])}")

    # Test 5: Output structure
    print("\n[Test 5] Output Structure")
    print("-"*70)

    result = process_question("suppose more rain, how will it affect LESS drought.", 'less')

    required = ['question', 'choices', 'influence_graph', 'answer', 'event_pair', 'para_steps', 'metadata']
    for key in required:
        assert key in result
        print(f"PASS: {key} exists ({type(result[key]).__name__})")

    # Summary
    print("\n" + "="*70)
    print("All Tests Passed!")
    print("="*70)
    return True

if __name__ == "__main__":
    try:
        test_all()
        print("\nStatus: SUCCESS")
        exit(0)
    except Exception as e:
        print(f"\nStatus: FAILED - {e}")
        import traceback
        traceback.print_exc()
        exit(1)
