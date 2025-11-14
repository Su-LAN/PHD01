# Observable Outcome Integration - Implementation Complete

## Summary

The observable outcome identification has been successfully integrated into the `ego_expansion_builder.py` at the **seed extraction** and **causal triple generation** stages. This is a fundamental architectural change that makes the entire causal graph building process context-aware.

## What Changed

### 1. New Method: `identify_observable_outcome()`

**Location**: Lines 248-362 in `ego_expansion_builder.py`

**Purpose**: Identifies the specific, measurable outcome being asked about BEFORE any seed extraction or causal graph building begins.

**Parameters**:
- `question`: The input question
- `intervention`: Optional intervention entity (helps with context)
- `target`: Optional target entity (helps with context)

**Returns**:
```python
{
    'observable_outcome': 'eating more vegetables',  # Specific interpretation
    'reasoning': 'Given stomach bug affects digestion...'  # Why this interpretation
}
```

**Example**:
```python
# Question: "suppose Having a stomach bug happens, how will it affect MORE vegetables"
result = builder.identify_observable_outcome(
    question=question,
    intervention="stomach bug",
    target="vegetables"
)
# Returns: {'observable_outcome': 'eating more vegetables', 'reasoning': '...'}
```

### 2. Modified: `extract_seeds()`

**Location**: Lines 364-420 in `ego_expansion_builder.py`

**Changes**:
- Added `observable_outcome` parameter
- When provided, the prompt includes context about what outcome we're observing
- Guides LLM to extract entities relevant to the specific outcome

**Example**:
```python
# OLD: Generic seed extraction
seeds = builder.extract_seeds(question)
# Returns: {stomach bug, vegetables, digestion, ...}

# NEW: Context-aware seed extraction
seeds = builder.extract_seeds(
    question=question,
    observable_outcome="eating more vegetables"
)
# Returns: {stomach bug, digestion, appetite, dietary fiber, nutrients, ...}
#          (focuses on eating-related entities, not farming/market)
```

**Key Improvement**:
- If observing "eating vegetables" → prioritizes: digestion, nutrients, dietary fiber, appetite
- If observing "growing vegetables" → prioritizes: soil, farmers, land, crops, agriculture
- If observing "buying vegetables" → prioritizes: market, prices, consumers, supply

### 3. Modified: `expand_causal_relations()`

**Location**: Lines 422-590 in `ego_expansion_builder.py`

**Changes**:
- Added `observable_outcome` parameter
- When generating causal relations, the LLM is guided to focus on relations relevant to the observable outcome
- Avoids generating irrelevant relations

**Example**:
```python
# OLD: Generic causal relation generation
relations = builder.expand_causal_relations(
    seed="stomach bug",
    entity_context=all_entities,
    existing_relations=all_relations
)
# Might return relations about: farming, agriculture, market prices (irrelevant)

# NEW: Context-aware causal relation generation
relations = builder.expand_causal_relations(
    seed="stomach bug",
    entity_context=all_entities,
    existing_relations=all_relations,
    observable_outcome="eating more vegetables"
)
# Returns only eating-relevant relations: stomach bug → appetite, stomach bug → nausea, etc.
```

**Key Improvement**:
The prompt explicitly guides the LLM:
- "If outcome is 'eating vegetables' and seed is 'stomach bug': Focus on digestion, appetite, nausea"
- "Avoid unrelated agricultural or market concepts"

### 4. Modified: `build_causal_chain()`

**Location**: Lines 770-882 in `ego_expansion_builder.py`

**Changes**:
- Added `intervention` and `target` parameters
- **Step 0 (NEW)**: Identifies observable outcome FIRST
- **Step 1**: Passes observable outcome to `extract_seeds()`
- **Step 2**: Passes observable outcome to `expand_causal_relations()`
- **Return value**: Includes `observable_outcome` in result dictionary

**New Flow**:
```python
def build_causal_chain(question, intervention=None, target=None):
    # STEP 0 (NEW): Identify what we're observing
    observable_info = identify_observable_outcome(question, intervention, target)
    observable_outcome = observable_info['observable_outcome']

    # STEP 1: Extract seeds (context-aware)
    seeds = extract_seeds(question, observable_outcome)

    # STEP 2: Expand relations (context-aware)
    for entity in seeds:
        relations = expand_causal_relations(
            entity, all_entities, all_relations,
            observable_outcome=observable_outcome  # Pass context!
        )

    # STEP 3: Build chains and return
    return {
        'seeds': seeds,
        'entities': all_entities,
        'edges': all_edges,
        'chains': chains,
        'observable_outcome': observable_info  # Include in result!
    }
```

## Usage in Notebook

### Before (Old Way):
```python
# In choice_reasoning.ipynb
builder_result = BUILDER.build_causal_chain(question)
# Observable outcome identified later, during choice evaluation
```

### After (New Way):
```python
# In choice_reasoning.ipynb
question_structure = PARSER.parse_question_structure(question)

# Build causal chain with context
builder_result = BUILDER.build_causal_chain(
    question=question,
    intervention=question_structure.get('intervention'),
    target=question_structure.get('target_entity')
)

# Observable outcome is now available immediately
observable_info = builder_result['observable_outcome']
observable_outcome = observable_info['observable_outcome']

# Use this for all choice evaluations
for choice in ['more', 'less', 'no_effect']:
    forward_result = forward_reasoning(
        question, choice, observable_outcome, ...
    )
    counterfactual_result = counterfactual_reasoning(
        question, choice, observable_outcome, ...
    )
```

## Benefits

### 1. **Earlier Context Understanding**
- Observable outcome identified at the very beginning (before seed extraction)
- All subsequent steps use this context

### 2. **More Relevant Seed Entities**
- Seeds are focused on the specific outcome domain
- Example: For "eating vegetables" → gets digestion/nutrition entities, not farming entities

### 3. **More Relevant Causal Relations**
- Causal relations directly support the observable outcome
- Reduces noise and irrelevant information

### 4. **Consistent Interpretation**
- The SAME observable outcome is used throughout:
  - Seed extraction
  - Causal relation generation
  - Choice evaluation (forward + counterfactual)
  - Final decision

### 5. **Better Performance**
- Focused knowledge graph (smaller, more relevant)
- Higher quality causal chains (more directly related to the question)
- More accurate reasoning (less confused by irrelevant relations)

## Example Comparison

### Question: "suppose Having a stomach bug happens, how will it affect MORE vegetables"

#### OLD System (without integration):
1. Extract seeds: {stomach bug, vegetables, soil, farmers, market, digestion, ...}
2. Generate relations: stomach bug → crop yield, vegetables → market price, ...
3. Identify observable: "eating vegetables" (too late!)
4. Try to reason with mixed/irrelevant relations

#### NEW System (with integration):
1. **Identify observable**: "eating more vegetables" (eating, not growing/buying)
2. Extract seeds (focused): {stomach bug, vegetables, digestion, appetite, nutrients, fiber}
3. Generate relations (focused): stomach bug → appetite↓, stomach bug → nausea, ...
4. Reason with relevant relations only

## Backward Compatibility

The changes are mostly backward compatible:

- `build_causal_chain(question)` still works (intervention/target are optional)
- If no intervention/target provided, observable outcome identification uses only the question
- Old code will continue to work, just without the context-aware benefits

## Next Steps

The notebook (`choice_reasoning.ipynb`) should be updated to:

1. Parse question structure first to get intervention and target
2. Pass these to `build_causal_chain()`
3. Use the returned `observable_outcome` for all choice evaluations
4. Remove the separate `identify_observable_outcome()` function from the notebook (use builder's version)

## Files Modified

1. **`ego_expansion_builder.py`**:
   - Added `identify_observable_outcome()` method
   - Modified `extract_seeds()` to accept `observable_outcome` parameter
   - Modified `expand_causal_relations()` to accept `observable_outcome` parameter
   - Modified `build_causal_chain()` to orchestrate the new flow

2. **Documentation created**:
   - This file: `OBSERVABLE_OUTCOME_INTEGRATION.md`

## Verification

To verify the changes work correctly, test with:

```python
from ego_expansion_builder import EgoExpansionCausalBuilder

builder = EgoExpansionCausalBuilder(model_name="gemma2:27b")

question = "suppose Having a stomach bug happens, how will it affect MORE vegetables"

result = builder.build_causal_chain(
    question=question,
    intervention="stomach bug",
    target="vegetables"
)

print("Observable Outcome:", result['observable_outcome'])
print("Seeds:", result['seeds'])
print("Number of edges:", len(result['edges']))

# Check that entities are relevant to eating, not farming
for entity in result['entities']:
    print(f"  - {entity}")
```

Expected: Entities and relations should be focused on eating/digestion domain, not agriculture/market domain.
