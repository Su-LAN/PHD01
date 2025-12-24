import os
import json
import pandas as pd
from WIQACausalBuilder import WIQACausalBuilder
from tqdm import tqdm

# Load the CSV file
csv_path = r'E:\PHD\01\other_code\CDCR-SFT\data\wiqa_test.csv'
df = pd.read_csv(csv_path)

print(f"Total datapoints in CSV: {len(df)}")
print(f"Question types distribution:")
print(df['question_type'].value_counts())
print("\n" + "="*80 + "\n")

# Store results
results = []

# Process each datapoint
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing WIQA test"):
    # Convert CSV row to the format expected by WIQACausalBuilder
    datapoint = {
        'question_stem': row['question_stem'],
        'para_steps': row['para_steps'],
        'answer_label': row['answer_label'],
        'answer_label_as_choice': row['answer_label_as_choice'],
        'question_type': row['question_type'],
        'improved_question': row['improved_question']
    }

    print(f"\n{'='*80}")
    print(f"Processing {idx+1}/{len(df)} - ID: {row['id']}")
    print(f"Type: {row['question_type']}")
    print(f"{'='*80}")
    print(f"Question: {datapoint['question_stem'][:100]}...")
    print(f"Gold answer: {datapoint['answer_label']} ({datapoint['answer_label_as_choice']})")
    print()

    try:
        # Initialize builder
        wiqa = WIQACausalBuilder(datapoint)

        # Step 1: Extract start and target entities
        print("Step 1: Extracting entities...")
        info = wiqa.extract_start_entity()
        start = info["cause_event"]
        target = info["outcome_base"]
        print(f"  Cause: '{start}'")
        print(f"  Outcome base: '{target}'")
        print()

        # Step 2: BFS expansion
        print("Step 2: BFS expansion...")
        bfs = wiqa.expand_toward_target(
            start_X=start,
            target_Y=target,
            max_depth=5,
            max_relations_per_node=5
        )
        print(f"  Triples found: {len(bfs['triples'])}")
        print(f"  Nodes visited: {len(bfs['visited'])}")
        print(f"  Close hits: {len(bfs['close_hits'])}")
        print()

        # Step 3: Bridge close hits and extract causal chain
        print("Step 3: Bridging and chain extraction...")
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
        print(f"  Paths found: {chain_result['num_paths']}")
        print()

        # Step 4: Generate description
        print("Step 4: Generating description...")
        description = wiqa.causal_chain_to_text(chain_result, bfs)
        print()

        # Step 5: LLM reasoning
        print("Step 5: Final reasoning...")
        reasoning_result = wiqa.reason_with_description(description, chain_result=chain_result)

        # Check result
        gold_label = datapoint['answer_label']
        pred_label = reasoning_result['predicted_answer']
        gold_norm = gold_label.strip().lower().replace(" ", "_")
        pred_norm = pred_label.strip().lower().replace(" ", "_")
        is_correct = (pred_norm == gold_norm)

        print(f"\nPrediction: {pred_label}")
        print(f"Gold: {gold_label}")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")

        # Store result
        result_entry = {
            'csv_id': row['id'],
            'question_type': row['question_type'],
            'question': datapoint['question_stem'],
            'gold_answer': gold_label,
            'gold_choice': datapoint['answer_label_as_choice'],
            'predicted_answer': pred_label,
            'predicted_choice': reasoning_result['predicted_choice'],
            'is_correct': is_correct,
            'confidence': reasoning_result['confidence'],
            'effect_on_base': reasoning_result.get('effect_on_base', 'N/A'),
            'reasoning': reasoning_result['reasoning'],
            'num_paths': chain_result['num_paths'],
            'num_triples': len(bfs['triples']),
        }
        results.append(result_entry)

    except Exception as e:
        print(f"\n✗ ERROR processing datapoint {idx+1}: {str(e)}")
        import traceback
        traceback.print_exc()

        # Store error result
        result_entry = {
            'csv_id': row['id'],
            'question_type': row['question_type'],
            'question': datapoint['question_stem'],
            'gold_answer': datapoint['answer_label'],
            'gold_choice': datapoint['answer_label_as_choice'],
            'predicted_answer': 'ERROR',
            'predicted_choice': 'ERROR',
            'is_correct': False,
            'confidence': 'N/A',
            'effect_on_base': 'N/A',
            'reasoning': f'Error: {str(e)}',
            'num_paths': 0,
            'num_triples': 0,
        }
        results.append(result_entry)

    print()

# Save all results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save to JSON
output_json = 'wiqa_test_results_by_type.json'
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Full results saved to: {output_json}")

# Save to CSV
results_df = pd.DataFrame(results)
output_csv = 'wiqa_test_results_by_type.csv'
results_df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"CSV results saved to: {output_csv}")

# Calculate statistics by question type
print("\n" + "="*80)
print("STATISTICS BY QUESTION TYPE")
print("="*80)

# Overall statistics
total_count = len(results)
correct_count = sum(1 for r in results if r['is_correct'])
error_count = sum(1 for r in results if r['predicted_answer'] == 'ERROR')
accuracy = correct_count / total_count if total_count > 0 else 0

print(f"\nOVERALL:")
print(f"  Total processed: {total_count}")
print(f"  Correct: {correct_count}")
print(f"  Wrong: {total_count - correct_count - error_count}")
print(f"  Errors: {error_count}")
print(f"  Accuracy: {accuracy:.2%}")

# Statistics by question type
print(f"\nBY QUESTION TYPE:")
for qtype in ['EXOGENOUS_EFFECT', 'INPARA_EFFECT']:
    type_results = [r for r in results if r['question_type'] == qtype]
    if type_results:
        type_total = len(type_results)
        type_correct = sum(1 for r in type_results if r['is_correct'])
        type_errors = sum(1 for r in type_results if r['predicted_answer'] == 'ERROR')
        type_accuracy = type_correct / type_total if type_total > 0 else 0

        print(f"\n  {qtype}:")
        print(f"    Total: {type_total}")
        print(f"    Correct: {type_correct}")
        print(f"    Wrong: {type_total - type_correct - type_errors}")
        print(f"    Errors: {type_errors}")
        print(f"    Accuracy: {type_accuracy:.2%}")

# Detailed results table
print("\n" + "="*80)
print("DETAILED RESULTS")
print("="*80)
print(f"{'Status':<8} {'ID':<5} {'Type':<20} {'Gold':<12} {'Pred':<12} {'Conf':<8}")
print("-" * 80)
for r in results:
    status = "✓" if r['is_correct'] else ("✗" if r['predicted_answer'] != 'ERROR' else "E")
    print(f"{status:<8} {r['csv_id']:<5} {r['question_type']:<20} {r['gold_answer']:<12} {r['predicted_answer']:<12} {r['confidence']}")
print("-" * 80)

print("\nDone!")
