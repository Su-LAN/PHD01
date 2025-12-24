import json
import requests
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict
import time

class WIQAOllamaEvaluator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.ollama_url = ollama_url
        self.model = model

    def load_wiqa_data(self, file_path: str, limit: int = None) -> List[Dict]:
        """Load WIQA dataset from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data.append(json.loads(line.strip()))
        return data

    def format_question_prompt(self, sample: Dict) -> str:
        """Format WIQA sample into a prompt for Ollama"""
        # Extract process steps
        steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(sample['question_para_step']) if step.strip()])

        # Extract question and choices
        question = sample['question_stem']
        choices_text = sample['choices']['text']
        choices_labels = sample['choices']['label']

        choices_str = "\n".join([f"{label}. {text}" for label, text in zip(choices_labels, choices_text)])

        prompt = f"""Given the following process:
{steps}

Question: {question}

Choices:
{choices_str}

Please answer with only the letter (A, B, or C) that best answers the question. Do not provide any explanation, just the letter."""

        return prompt

    def query_ollama(self, prompt: str) -> str:
        """Query Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.1  # Lower temperature for more deterministic responses
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result['response'].strip()
            else:
                return f"ERROR: {response.status_code}"
        except Exception as e:
            return f"ERROR: {str(e)}"

    def extract_answer(self, response: str) -> str:
        """Extract answer choice from Ollama response"""
        response = response.strip().upper()

        # Look for A, B, or C at the start or in the response
        for choice in ['A', 'B', 'C']:
            if response.startswith(choice):
                return choice
            if f"ANSWER: {choice}" in response or f"ANSWER IS {choice}" in response:
                return choice

        # Look for the first occurrence of A, B, or C
        for char in response:
            if char in ['A', 'B', 'C']:
                return char

        return "UNKNOWN"

    def evaluate_sample(self, sample: Dict, verbose: bool = False) -> Dict:
        """Evaluate a single WIQA sample"""
        prompt = self.format_question_prompt(sample)
        if verbose:
            print(f"Prompt:\n{prompt}\n")
        response = self.query_ollama(prompt)
        if verbose:
            print(f"Response: {response}\n")
        predicted_answer = self.extract_answer(response)
        correct_answer = sample['answer_label_as_choice']

        return {
            'question_id': sample['metadata_question_id'],
            'question': sample['question_stem'],
            'correct_answer': correct_answer,
            'correct_answer_text': sample['answer_label'],
            'predicted_answer': predicted_answer,
            'full_response': response,
            'is_correct': predicted_answer == correct_answer,
            'question_type': sample['metadata_question_type'],
            'path_len': sample['metadata_path_len']
        }

    def evaluate_dataset(self, file_path: str, num_samples: int = 100) -> Tuple[List[Dict], Dict]:
        """Evaluate multiple samples from WIQA dataset"""
        print(f"Loading WIQA dataset from {file_path}...")
        data = self.load_wiqa_data(file_path, limit=num_samples)
        print(f"Loaded {len(data)} samples")

        results = []
        print(f"\nEvaluating with Ollama model: {self.model}")
        print("="*80)

        for i, sample in enumerate(data):
            print(f"\n[{i+1}/{len(data)}] Processing question: {sample['metadata_question_id']}")
            result = self.evaluate_sample(sample, verbose=(i < 2))  # Show first 2 in detail
            results.append(result)

            status = "CORRECT" if result['is_correct'] else "WRONG"
            print(f"[{status}] Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")

            # Add small delay to avoid overwhelming the API
            time.sleep(0.1)

        # Calculate statistics
        stats = self.calculate_statistics(results)

        return results, stats

    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation statistics"""
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        accuracy = (correct / total * 100) if total > 0 else 0

        # Statistics by question type
        by_type = defaultdict(lambda: {'total': 0, 'correct': 0})
        for r in results:
            q_type = r['question_type']
            by_type[q_type]['total'] += 1
            if r['is_correct']:
                by_type[q_type]['correct'] += 1

        type_stats = {
            q_type: {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            }
            for q_type, stats in by_type.items()
        }

        # Statistics by path length
        by_path_len = defaultdict(lambda: {'total': 0, 'correct': 0})
        for r in results:
            path_len = r['path_len']
            by_path_len[path_len]['total'] += 1
            if r['is_correct']:
                by_path_len[path_len]['correct'] += 1

        path_len_stats = {
            path_len: {
                'total': stats['total'],
                'correct': stats['correct'],
                'accuracy': (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            }
            for path_len, stats in by_path_len.items()
        }

        return {
            'total': total,
            'correct': correct,
            'incorrect': total - correct,
            'accuracy': accuracy,
            'by_question_type': type_stats,
            'by_path_length': path_len_stats
        }

    def save_results(self, results: List[Dict], stats: Dict, output_file: str):
        """Save evaluation results to JSON file"""
        output = {
            'metadata': {
                'model': self.model,
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(results)
            },
            'statistics': stats,
            'results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {output_file}")

    def print_summary(self, stats: Dict):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Total samples: {stats['total']}")
        print(f"Correct: {stats['correct']}")
        print(f"Incorrect: {stats['incorrect']}")
        print(f"Accuracy: {stats['accuracy']:.2f}%")

        print("\n" + "-"*80)
        print("ACCURACY BY QUESTION TYPE:")
        print("-"*80)
        for q_type, type_stats in sorted(stats['by_question_type'].items()):
            print(f"{q_type:30s}: {type_stats['correct']:3d}/{type_stats['total']:3d} ({type_stats['accuracy']:5.2f}%)")

        print("\n" + "-"*80)
        print("ACCURACY BY PATH LENGTH:")
        print("-"*80)
        for path_len, len_stats in sorted(stats['by_path_length'].items()):
            print(f"Path length {path_len:2d}: {len_stats['correct']:3d}/{len_stats['total']:3d} ({len_stats['accuracy']:5.2f}%)")

        print("="*80)


def main():
    # Configuration
    WIQA_DATA_FILE = "wiqa_train_data.json"
    NUM_SAMPLES = 20  # Number of samples to evaluate (starting with 20 for faster testing)
    OLLAMA_MODEL = "llama3.1:8b"  # Using available llama3.1:8b model
    OUTPUT_FILE = f"wiqa_ollama_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Initialize evaluator
    evaluator = WIQAOllamaEvaluator(model=OLLAMA_MODEL)

    # Run evaluation
    results, stats = evaluator.evaluate_dataset(WIQA_DATA_FILE, num_samples=NUM_SAMPLES)

    # Print summary
    evaluator.print_summary(stats)

    # Save results
    evaluator.save_results(results, stats, OUTPUT_FILE)

    # Print some example errors for analysis
    print("\n" + "="*80)
    print("EXAMPLE ERRORS (First 5):")
    print("="*80)
    errors = [r for r in results if not r['is_correct']][:5]
    for i, error in enumerate(errors, 1):
        print(f"\n[Error {i}]")
        print(f"Question: {error['question']}")
        print(f"Correct: {error['correct_answer']} ({error['correct_answer_text']})")
        print(f"Predicted: {error['predicted_answer']}")
        print(f"Response: {error['full_response'][:200]}...")


if __name__ == "__main__":
    main()
