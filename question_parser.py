"""
Question Parser for Meta-Level Causal Reasoning
================================================
Handles questions that ask about effects on pre-specified outcomes.

Example:
    "suppose X happens, how will it affect LESS Y"
    - This asks how X affects "less Y" (the outcome itself)
    - If X causes less Y, the answer is "more" (more of the "less Y" phenomenon)
"""

import re
import ollama
from typing import Dict, Optional, Tuple


class QuestionParser:
    """
    解析问题结构，识别是否为meta-level的因果推理问题
    """

    def __init__(self, model_name: str = "gemma2:27b", verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose

    def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['response'].strip()

    def parse_question_structure(self, question: str) -> Dict:
        """
        解析问题结构，识别是否包含预设结果

        Args:
            question: 原始问题

        Returns:
            包含以下字段的字典：
            - is_meta_level: bool，是否为meta-level问题
            - intervention: str，干预/假设条件（suppose X happens）
            - target_phrase: str，目标短语（如"LESS rabbits"）
            - target_direction: str，目标方向（more/less/no_effect）
            - target_entity: str，目标实体（如"rabbits"）
            - question_type: str，问题类型（direct/meta）
        """
        # 构建LLM提示
        prompt = f"""Analyze the structure of this causal reasoning question.

Question: {question}

Identify:
1. Is this asking about a DIRECT effect (e.g., "how will X affect Y?")
   OR a META-LEVEL effect (e.g., "how will X affect LESS Y?" or "how will X affect MORE Y?")?

2. If META-LEVEL:
   - Extract the intervention/condition (the "suppose X happens" part)
   - Extract the target phrase exactly as written (e.g., "LESS rabbits", "MORE production")
   - Identify the direction word in the target (MORE/LESS/INCREASE/DECREASE etc.)
   - Extract the target entity (e.g., "rabbits", "production")

3. If DIRECT:
   - Extract the intervention/condition
   - Extract the target entity

Return ONLY in this JSON format:
{{
  "question_type": "meta" or "direct",
  "intervention": "the intervention/condition text",
  "target_phrase": "exact phrase like 'LESS rabbits'" or null if direct,
  "target_direction": "more" or "less" or null if direct,
  "target_entity": "the entity being affected"
}}

Examples:
Question: "suppose the female is sterile happens, how will it affect LESS rabbits."
Output: {{"question_type": "meta", "intervention": "female is sterile", "target_phrase": "LESS rabbits", "target_direction": "less", "target_entity": "rabbits"}}

Question: "suppose there is more rain, how will it affect plant growth?"
Output: {{"question_type": "direct", "intervention": "more rain", "target_phrase": null, "target_direction": null, "target_entity": "plant growth"}}

Now analyze:"""

        try:
            response = self._call_llm(prompt)

            # 尝试解析JSON
            import json
            # 查找JSON部分
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))

                # 标准化字段
                is_meta = result.get('question_type', 'direct') == 'meta'

                return {
                    'is_meta_level': is_meta,
                    'intervention': result.get('intervention', ''),
                    'target_phrase': result.get('target_phrase'),
                    'target_direction': self._normalize_direction(result.get('target_direction')),
                    'target_entity': result.get('target_entity', ''),
                    'question_type': result.get('question_type', 'direct')
                }
        except Exception as e:
            if self.verbose:
                print(f"LLM parsing failed: {e}, using heuristic parsing")

        # Fallback: 使用启发式规则解析
        return self._heuristic_parse(question)

    def _heuristic_parse(self, question: str) -> Dict:
        """
        启发式解析（备用方案）
        """
        question_lower = question.lower()

        # 检查是否包含方向性词汇 + 实体的模式
        # 如 "affect LESS rabbits", "affect MORE production"
        patterns = [
            r'affect\s+(more|less|increase|decrease|higher|lower|greater|fewer)\s+(\w+)',
            r'impact\s+(more|less|increase|decrease|higher|lower|greater|fewer)\s+(\w+)',
            r'influence\s+(more|less|increase|decrease|higher|lower|greater|fewer)\s+(\w+)',
        ]

        is_meta = False
        target_phrase = None
        target_direction = None
        target_entity = None

        for pattern in patterns:
            match = re.search(pattern, question_lower)
            if match:
                is_meta = True
                direction_word = match.group(1)
                entity = match.group(2)

                target_direction = self._normalize_direction(direction_word)
                target_entity = entity
                target_phrase = f"{direction_word.upper()} {entity}"
                break

        # 提取干预条件（suppose ... happens部分）
        intervention = ""
        suppose_match = re.search(r'suppose\s+(.+?)\s+happens', question_lower)
        if suppose_match:
            intervention = suppose_match.group(1)

        return {
            'is_meta_level': is_meta,
            'intervention': intervention,
            'target_phrase': target_phrase,
            'target_direction': target_direction,
            'target_entity': target_entity,
            'question_type': 'meta' if is_meta else 'direct'
        }

    def _normalize_direction(self, direction: Optional[str]) -> Optional[str]:
        """
        标准化方向词
        """
        if not direction:
            return None

        direction = direction.lower().strip()

        # 映射到标准的more/less
        positive = ['more', 'increase', 'higher', 'greater', 'increased']
        negative = ['less', 'decrease', 'lower', 'fewer', 'reduced', 'decreased']

        if direction in positive:
            return 'more'
        elif direction in negative:
            return 'less'
        else:
            return direction

    def should_invert_answer(
        self,
        question_structure: Dict,
        causal_decision: str
    ) -> Tuple[bool, str]:
        """Decide the final answer for meta-level questions."""

        if not question_structure.get('is_meta_level'):
            return False, causal_decision

        target_direction = question_structure.get('target_direction')

        if causal_decision == 'no_effect':
            return False, 'no_effect'

        if target_direction == 'less':
            if causal_decision == 'less':
                return True, 'more'
            if causal_decision == 'more':
                return False, 'less'
            return False, 'no_effect'

        if target_direction == 'more':
            if causal_decision in ('more', 'less'):
                return False, causal_decision
            return False, 'no_effect'

        return False, causal_decision

    def _invert_direction(self, direction: str) -> str:
        """反转方向"""
        if direction == 'more':
            return 'less'
        elif direction == 'less':
            return 'more'
        else:
            return direction

    def explain_reasoning(
        self,
        question_structure: Dict,
        causal_decision: str,
        final_answer: str
    ) -> str:
        """
        生成推理解释
        """
        if not question_structure['is_meta_level']:
            return f"Direct causal reasoning: {causal_decision}"

        explanation = f"""Meta-level causal reasoning detected:

Question asks: "How will {question_structure['intervention']} affect {question_structure['target_phrase']}?"

Causal chain analysis: {question_structure['intervention']} → {causal_decision} {question_structure['target_entity']}

Interpretation:
- The question asks about effects on the outcome "{question_structure['target_phrase']}"
- The causal chain shows: {causal_decision} {question_structure['target_entity']}
- Since the intervention leads to {causal_decision} {question_structure['target_entity']},
  this means {final_answer} of the "{question_structure['target_phrase']}" phenomenon

Final answer: {final_answer}
"""
        return explanation


def test_parser():
    """测试函数"""
    parser = QuestionParser()

    test_cases = [
        "suppose the female is sterile happens, how will it affect LESS rabbits.",
        "suppose there is more rain, how will it affect plant growth?",
        "if temperature increases, how will it affect MORE ice melting?",
    ]

    for question in test_cases:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        structure = parser.parse_question_structure(question)
        print(f"Structure: {structure}")

        # 模拟不同的因果决策
        for causal_decision in ['more', 'less', 'no_effect']:
            should_invert, final_answer = parser.should_invert_answer(
                structure, causal_decision
            )
            print(f"\nIf causal chain says '{causal_decision}':")
            print(f"  Should invert: {should_invert}")
            print(f"  Final answer: {final_answer}")


if __name__ == "__main__":
    test_parser()
