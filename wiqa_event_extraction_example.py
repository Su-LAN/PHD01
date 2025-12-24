# -*- coding: utf-8 -*-
"""
WIQA Event Pair Extraction - Usage Examples
"""

from WIQACausalBuilder import WIQAEventExtractor, EventPair


# Example 1: Extract from a single question string
print("=" * 80)
print("Example 1: Extract from a single question")
print("=" * 80)

question = "suppose less gasoline is loaded onto tank trucks happens, how will it affect LESS electricity being produced."
pair = WIQAEventExtractor.extract_event_pair(question, answer_label='no_effect')

print(f"Question: {question}")
print(f"\nExtracted Event Pair:")
print(f"  Cause:     {pair.cause}")
print(f"  Outcome:   {pair.outcome}")
print(f"  Direction: {pair.direction}")
print(f"  Answer:    {pair.answer_label}")


# Example 2: Extract from WIQA dataset item
print("\n" + "=" * 80)
print("Example 2: Extract from dataset item")
print("=" * 80)

dataset_item = {
    'question_stem': 'suppose more pollution in the environment happens, how will it affect more cocoon hatches.',
    'answer_label': 'less'
}

pair = WIQAEventExtractor.extract_from_dataset_item(dataset_item)
print(f"Dataset item: {dataset_item}")
print(f"\nExtracted: {pair}")
print(f"As dict: {pair.to_dict()}")


# Example 3: Batch extraction
print("\n" + "=" * 80)
print("Example 3: Batch extraction from multiple items")
print("=" * 80)

items = [
    {'question_stem': 'suppose there will be fewer new trees happens, how will it affect LESS forest formation.', 'answer_label': 'more'},
    {'question_stem': 'suppose more sunny days happens, how will it affect less sugar and oxygen produced.', 'answer_label': 'less'},
    {'question_stem': 'suppose prices for aluminum rise happens, how will it affect less aluminum pollutes waterways.', 'answer_label': 'no_effect'},
]

pairs = WIQAEventExtractor.batch_extract(items)

print(f"Extracted {len(pairs)} event pairs:\n")
for i, pair in enumerate(pairs, 1):
    print(f"{i}. Cause: {pair.cause}")
    print(f"   Outcome: {pair.outcome} ({pair.direction})")
    print(f"   Answer: {pair.answer_label}\n")


# Example 4: Using with HuggingFace datasets
print("=" * 80)
print("Example 4: Integration with HuggingFace WIQA dataset")
print("=" * 80)

try:
    from datasets import load_dataset

    # Load a small sample
    ds = load_dataset('allenai/wiqa', split='validation[:5]')

    print(f"Loaded {len(ds)} samples from WIQA validation set\n")

    # Extract event pairs
    for i, item in enumerate(ds):
        pair = WIQAEventExtractor.extract_from_dataset_item(item)
        if pair:
            print(f"Sample {i+1}:")
            print(f"  Question: {item['question_stem'][:80]}...")
            print(f"  Cause -> Outcome: {pair.cause} -> {pair.outcome}")
            print(f"  Direction: {pair.direction}, Answer: {pair.answer_label}\n")

except ImportError:
    print("HuggingFace datasets not installed. Skip this example.")
except Exception as e:
    print(f"Error loading dataset: {e}")


print("=" * 80)
print("All examples completed!")
print("=" * 80)
