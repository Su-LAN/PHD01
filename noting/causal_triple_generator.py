"""Generate causal triples with confidence filtering via local Ollama."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import ollama


PROMPT_TEMPLATE = """You are a scientific reasoning assistant.

### TASK
Given a natural language question about a scientific or technical scenario,
your goal is to extract the key entities and infer possible *causal or functional*
relationships among them.

### RULES
1. You may include **implicit intermediate scientific variables** (e.g., temperature, resistance, energy, pressure, conductivity, metabolism, etc.) if they logically connect the cause and effect.
2. Only output a relation if a clear, evidence-based scientific mechanism exists.
3. If no known or plausible mechanism exists between two entities, output "no_relation".
4. Do NOT hallucinate connections—if uncertain, use "no_relation".
5. Each triple must include a brief scientific explanation ("description") and a numeric "confidence" value between 0.0–1.0.
6. Focus on *causal* or *functional* relations (not associations or co-occurrences).

### OUTPUT FORMAT
Return a JSON object with:
{{
  "entities": ["entity1", "entity2", ...],
  "triples": [
    {{
      "head": "...",
      "relation": "...",
      "tail": "...",
      "description": "...",
      "confidence": 0.xx
    }}
  ]
}}

Allowed relation types:
["causes", "produces", "increases", "reduces", "enables", "depends on", "no_relation"]

### EXAMPLES
Example 1:
Question: "Suppose more carbon dioxide in the air happens, how will it affect more current is produced?"
Output:
{{
  "entities": ["carbon dioxide", "temperature", "resistance", "current"],
  "triples": [
    {{"head": "carbon dioxide", "relation": "increases", "tail": "temperature", "description": "CO2 traps heat, raising air temperature.", "confidence": 0.9}},
    {{"head": "temperature", "relation": "increases", "tail": "resistance", "description": "Higher temperature raises electrical resistance.", "confidence": 0.85}},
    {{"head": "resistance", "relation": "reduces", "tail": "current", "description": "Higher resistance lowers current under constant voltage.", "confidence": 0.95}}
  ]
}}

Example 2:
Question: "If the salinity of water increases, how does it affect the buoyancy of an object?"
Output:
{{
  "entities": ["salinity", "water density", "buoyancy"],
  "triples": [
    {{"head": "salinity", "relation": "increases", "tail": "water density", "description": "Higher salinity increases the density of water.", "confidence": 0.92}},
    {{"head": "water density", "relation": "increases", "tail": "buoyancy", "description": "Greater water density increases buoyant force.", "confidence": 0.9}}
  ]
}}

Now, analyze the following question carefully and produce the output in the same JSON structure.

Question: {question}
"""


def generate_causal_triples(
    question: str,
    model: str = "gemma2:27b",
    confidence_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Call Ollama locally to generate causal triples and filter by confidence."""

    prompt = PROMPT_TEMPLATE.format(question=question.strip())
    response = ollama.generate(model=model, prompt=prompt)
    raw_text = response.get("response", "").strip()

    parsed = _parse_json_payload(raw_text)
    entities = parsed.get("entities") if isinstance(parsed, dict) else None
    triples = parsed.get("triples") if isinstance(parsed, dict) else None

    if not isinstance(entities, list):
        entities = []
    if not isinstance(triples, list):
        triples = []

    filtered_triples: List[Dict[str, Any]] = []
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        confidence = _safe_float(triple.get("confidence"))
        if confidence is None:
            continue
        if confidence < confidence_threshold:
            continue
        triple = dict(triple)
        triple["confidence"] = confidence
        filtered_triples.append(triple)

    return {
        "question": question,
        "entities": entities,
        "triples": filtered_triples,
        "raw_response": raw_text,
        "model": model,
        "confidence_threshold": confidence_threshold,
    }


def _parse_json_payload(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = "\n".join(line for line in cleaned.splitlines()[1:-1])

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result > 1.0:
        result = result / 100.0
    if 0.0 <= result <= 1.0:
        return result
    return None


__all__ = ["generate_causal_triples"]


if __name__ == "__main__":
    sample_question = "Suppose more carbon dioxide in the air happens, how will it affect more current is produced?"
    result = generate_causal_triples(sample_question, confidence_threshold=0.7)
    print(json.dumps(result, ensure_ascii=False, indent=2))
