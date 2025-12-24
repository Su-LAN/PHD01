# WIQACausalBuilder 流程可视化

## 高层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     WIQA Question                               │
│  "suppose X happens, how will it affect Y?"                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 1: Extract Start & Target                     │
│                  (extract_start_entity)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LLM Prompt: Extract cause_event & outcome_base         │   │
│  └─────────────────────────────────────────────────────────┘   │
│  Output: start="X", target="Y", direction="MORE/LESS"          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            Step 2: BFS Causal Graph Expansion                   │
│                 (expand_toward_target)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Depth 0: frontier = [X]                                 │  │
│  │    ↓                                                     │  │
│  │  Find one-hop relations from X                          │  │
│  │    ├→ (X, RESULTS_IN, A)                                │  │
│  │    ├→ (X, NOT_RESULTS_IN, B)                            │  │
│  │    └→ (X, RESULTS_IN, C)                                │  │
│  │    ↓                                                     │  │
│  │  Check each tail node:                                  │  │
│  │    ├─ Exact match with Y? → FOUND!                      │  │
│  │    ├─ is_same_variable(tail, Y) = "same"? → Bridge & FOUND│ │
│  │    ├─ is_same_variable(tail, Y) = "close"? → Record hit│  │
│  │    └─ Otherwise: add to next frontier                   │  │
│  │    ↓                                                     │  │
│  │  Depth 1: frontier = [A, B, C]                          │  │
│  │    ↓ (repeat...)                                        │  │
│  │  Depth 2, 3, ... (until max_depth or found)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│  Output: triples=[(h,r,t),...], close_hits=[...], found=T/F   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│          Step 2.5: Attach Close Hits (if needed)                │
│              (attach_outcome_via_close)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ If close_hits exist:                                     │  │
│  │   For each close node:                                   │  │
│  │     Add bridge edge: (close_node, RESULTS_IN, Y)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  Output: extended_triples                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 3: Extract Causal Chains                      │
│                  (get_causal_chain)                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Build graph from triples                                 │  │
│  │   ↓                                                       │  │
│  │ DFS search all paths from X to Y                         │  │
│  │   Path 1: X → A → B → Y                                  │  │
│  │   Path 2: X → C → Y                                      │  │
│  │   ...                                                     │  │
│  │   ↓                                                       │  │
│  │ Compute statistics                                       │  │
│  │   - num_paths                                            │  │
│  │   - shortest_path_length                                │  │
│  │   - all_nodes_in_paths                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│  Output: paths=[...], num_paths=N, ...                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│         Step 4: Generate Natural Language Description           │
│                 (causal_chain_to_text)                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Phase 1: Structured Summary                              │  │
│  │   "Found N paths from X to Y"                            │  │
│  │   "Path 1: X --[R1]--> A --[R2]--> Y"                    │  │
│  │   "M RESULTS_IN edges, K NOT_RESULTS_IN edges"          │  │
│  │   ↓                                                       │  │
│  │ Phase 2: LLM Polishing                                   │  │
│  │   LLM Prompt: Convert summary to natural English        │  │
│  │   Constraint: Don't predict answer, just describe        │  │
│  └──────────────────────────────────────────────────────────┘  │
│  Output: natural_description (text)                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Step 5: LLM Reasoning & Prediction                 │
│                (reason_with_description)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ LLM Prompt:                                              │  │
│  │   QUESTION: [original question]                         │  │
│  │   CAUSAL ANALYSIS: [natural description]                │  │
│  │   CHOICES: A. more / B. less / C. no effect             │  │
│  │   ↓                                                       │  │
│  │   Task: Select answer based on causal analysis          │  │
│  │   ↓                                                       │  │
│  │ LLM Output (JSON):                                       │  │
│  │   {                                                       │  │
│  │     "predicted_answer": "more/less/no_effect",          │  │
│  │     "predicted_choice": "A/B/C",                        │  │
│  │     "reasoning": "...",                                  │  │
│  │     "confidence": "high/medium/low"                     │  │
│  │   }                                                       │  │
│  │   ↓                                                       │  │
│  │ Answer Canonicalization                                 │  │
│  │   (normalize to: more/less/no_effect)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  Output: predicted_answer, reasoning, confidence                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Final Output                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • Predicted answer: more/less/no_effect                  │  │
│  │ • Reasoning process: [detailed explanation]             │  │
│  │ • Confidence: high/medium/low                           │  │
│  │ • Causal chains: [all discovered paths]                │  │
│  │ • Natural description: [readable explanation]           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## BFS 扩展详细流程

```
expand_toward_target(X, Y, max_depth=3)

Initialize:
  frontier = [X]
  visited = {X}
  triples = []
  close_hits = []
  found = False

┌─────────────────────────────────────────────────┐
│  Depth 0                                        │
│  ┌──────────────────────────────────────────┐  │
│  │ Current: X                               │  │
│  │   ↓                                      │  │
│  │ find_causal_relations(X, target=Y)       │  │
│  │   └→ LLM generates:                      │  │
│  │      (X, RESULTS_IN, A)                  │  │
│  │      (X, NOT_RESULTS_IN, B)              │  │
│  │      (X, RESULTS_IN, C)                  │  │
│  │   ↓                                      │  │
│  │ For each new node (A, B, C):             │  │
│  │   ├─ A: is_same_variable(A, Y)?          │  │
│  │   │   └→ "different" → add to frontier   │  │
│  │   ├─ B: is_same_variable(B, Y)?          │  │
│  │   │   └→ "different" → add to frontier   │  │
│  │   └─ C: is_same_variable(C, Y)?          │  │
│  │       └→ "close" → add to close_hits     │  │
│  │                    + add to frontier     │  │
│  └──────────────────────────────────────────┘  │
│  Next frontier: [A, B, C]                       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Depth 1                                        │
│  ┌──────────────────────────────────────────┐  │
│  │ Current: A                               │  │
│  │   ↓                                      │  │
│  │ find_causal_relations(A, target=Y)       │  │
│  │   └→ (A, RESULTS_IN, D)                  │  │
│  │      (A, NOT_RESULTS_IN, E)              │  │
│  │   ↓                                      │  │
│  │ Check: is_same_variable(D, Y)?           │  │
│  │   └→ "different"                         │  │
│  │ Check: is_same_variable(E, Y)?           │  │
│  │   └→ "same"! ✓                          │  │
│  │   ↓                                      │  │
│  │ Bridge: (E, RESULTS_IN, Y)               │  │
│  │ found = True! → STOP                     │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                    ↓
Return:
  triples = [
    (X, RESULTS_IN, A),
    (X, NOT_RESULTS_IN, B),
    (X, RESULTS_IN, C),
    (A, RESULTS_IN, D),
    (A, NOT_RESULTS_IN, E),
    (E, RESULTS_IN, Y)  ← bridge edge
  ]
  close_hits = [{"node": C, "depth": 1}]
  found = True
```

---

## is_same_variable 决策树

```
is_same_variable(node, target)

LLM evaluates semantic relationship
              ↓
    ┌─────────┴─────────┐
    │  Are they about   │
    │  the same thing?  │
    └─────────┬─────────┘
              │
     ┌────────┴────────┐
    No                Yes
     │                 │
     ▼                 ▼
Different       ┌──────────────┐
(skip)          │ Same polarity?│
                └──────┬───────┘
                       │
              ┌────────┴────────┐
             Yes               No
              │                 │
              ▼                 ▼
     ┌──────────────┐   ┌──────────────┐
     │    "same"    │   │  "opposite"  │
     │              │   │              │
     │ Bridge with  │   │ Bridge with  │
     │ RESULTS_IN   │   │NOT_RESULTS_IN│
     └──────────────┘   └──────────────┘

Example:
  Target: "oxygen in blood"

  Node: "blood oxygenation"
    → Same thing? Yes
    → Same direction? Yes
    → Label: "same"
    → Action: (node, RESULTS_IN, target)

  Node: "oxygen depletion"
    → Same thing? Yes
    → Same direction? No (opposite)
    → Label: "opposite"
    → Action: (node, NOT_RESULTS_IN, target)

  Node: "hemoglobin level"
    → Same thing? Related but not quite
    → Label: "close"
    → Action: Add to close_hits

  Node: "water temperature"
    → Same thing? No
    → Label: "different"
    → Action: Continue search
```

---

## Path Extraction (DFS)

```
get_causal_chain(triples, X, Y)

Build adjacency graph:
  graph = {
    X: [edge1, edge2, ...],
    A: [edge3, ...],
    ...
  }

DFS Search:

dfs(current=X, path=[])
  │
  ├─ If current == Y and path not empty:
  │    save path ✓
  │    return
  │
  └─ For each outgoing edge from current:
       │
       ├─ If edge.tail in path:  ← cycle detection
       │    skip (avoid cycles)
       │
       └─ Else:
            path.append(edge)
            dfs(edge.tail, path)  ← recurse
            path.pop()            ← backtrack

Example:
  Graph:
    X → A (RESULTS_IN)
    X → B (NOT_RESULTS_IN)
    A → Y (RESULTS_IN)
    B → C (RESULTS_IN)
    C → Y (NOT_RESULTS_IN)

  DFS finds:
    Path 1: X → A → Y
    Path 2: X → B → C → Y

  Statistics:
    num_paths = 2
    shortest_path_length = 2
    all_nodes = {X, A, B, C, Y}
```

---

## Description Generation Pipeline

```
causal_chain_to_text(chain_result, bfs_result)

┌─────────────────────────────────────────────────┐
│  Phase 1: Structured Summary                    │
│  ┌───────────────────────────────────────────┐  │
│  │ IF num_paths > 0:                         │  │
│  │   "Found N paths from X to Y"             │  │
│  │   For each path (up to 3):                │  │
│  │     List edges with relations             │  │
│  │   Count positive/negative edges           │  │
│  │                                            │  │
│  │ ELSE (no paths):                          │  │
│  │   "No path found"                         │  │
│  │   List BFS one-hop relations (up to 8)   │  │
│  │   "Explored N nodes"                      │  │
│  └───────────────────────────────────────────┘  │
│  ↓                                              │
│  raw_summary = """                              │
│  From 'X' to 'Y', found 2 paths.               │
│  Path 1: X --[RESULTS_IN]--> A --[R...]--> Y  │
│  Path 2: X --[NOT_RESULTS_IN]--> B --> Y      │
│  2 positive, 1 negative edges.                │
│  """                                            │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Phase 2: LLM Polish                            │
│  ┌───────────────────────────────────────────┐  │
│  │ LLM Prompt:                               │  │
│  │   "Convert this structured summary        │  │
│  │    into natural English.                  │  │
│  │    Don't predict answer, just describe."  │  │
│  │                                            │  │
│  │ Input: raw_summary                        │  │
│  │                                            │  │
│  │ Output: polished natural language         │  │
│  └───────────────────────────────────────────┘  │
│  ↓                                              │
│  natural_description = """                      │
│  The causal analysis identified two pathways   │
│  connecting X to Y. The first pathway shows    │
│  X increasing A, which then affects Y. The     │
│  second demonstrates X decreasing B, leading   │
│  to changes in Y. The pathways contain         │
│  predominantly positive causal relationships.  │
│  """                                            │
└─────────────────────────────────────────────────┘
                    ↓
              Return description
```

---

## Final Reasoning Flow

```
reason_with_description(description, question, choices)

┌─────────────────────────────────────────────────┐
│  Construct LLM Prompt                           │
│  ┌───────────────────────────────────────────┐  │
│  │ QUESTION:                                 │  │
│  │   [original question]                     │  │
│  │                                            │  │
│  │ CAUSAL ANALYSIS:                          │  │
│  │   [natural language description]          │  │
│  │                                            │  │
│  │ ANSWER CHOICES:                           │  │
│  │   A. more                                 │  │
│  │   B. less                                 │  │
│  │   C. no effect                            │  │
│  │                                            │  │
│  │ Task:                                     │  │
│  │   Select answer based on analysis         │  │
│  │   Return JSON with reasoning              │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  LLM Generates Response                         │
│  {                                              │
│    "predicted_answer": "less",                 │
│    "predicted_choice": "B",                    │
│    "reasoning": "The causal chain shows...",   │
│    "confidence": "high"                        │
│  }                                              │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Answer Canonicalization                        │
│  ┌───────────────────────────────────────────┐  │
│  │ Normalize variations:                     │  │
│  │   "less" / "fewer" → "less"              │  │
│  │   "more" / "greater" → "more"            │  │
│  │   "no effect" / "none" → "no_effect"     │  │
│  │                                            │  │
│  │ Use choice letter to verify:             │  │
│  │   If choice="B" and choices[B]="less"    │  │
│  │   → confirmed: "less"                    │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
                    ↓
              Return result
```

---

## Complete Example Trace

### Input
```
Question: "suppose less oxygen is inhaled, how will it affect MORE oxygen enters blood"
```

### Trace

```
[Step 1: Extract]
  LLM extracts:
    cause_event = "less oxygen is inhaled"
    outcome_base = "oxygen enters blood"
    outcome_direction = "MORE"

[Step 2: BFS]
  Depth 0: frontier = ["less oxygen is inhaled"]
    → find_causal_relations("less oxygen is inhaled")
    → LLM returns:
        ("less O2 inhaled", NOT_RESULTS_IN, "oxygen in lungs")
        ("less O2 inhaled", NOT_RESULTS_IN, "air intake")

  Depth 1: frontier = ["oxygen in lungs", "air intake"]
    → find_causal_relations("oxygen in lungs")
    → LLM returns:
        ("oxygen in lungs", RESULTS_IN, "oxygen diffusion")
    → is_same_variable("oxygen diffusion", "oxygen enters blood")?
    → LLM: "same"
    → Add bridge: ("oxygen diffusion", RESULTS_IN, "oxygen enters blood")
    → found = True!

  Result:
    triples = [
      ("less O2 inhaled", NOT_RESULTS_IN, "oxygen in lungs"),
      ("less O2 inhaled", NOT_RESULTS_IN, "air intake"),
      ("oxygen in lungs", RESULTS_IN, "oxygen diffusion"),
      ("oxygen diffusion", RESULTS_IN, "oxygen enters blood")
    ]

[Step 3: Extract Paths]
  DFS finds 1 path:
    ["less O2 inhaled"] → [NOT_RESULTS_IN] → ["oxygen in lungs"]
                       → [RESULTS_IN] → ["oxygen diffusion"]
                       → [RESULTS_IN] → ["oxygen enters blood"]

  Statistics:
    num_paths = 1
    shortest = 3 edges
    nodes = 4

[Step 4: Description]
  Structured summary:
    "Found 1 path. Path has 1 negative edge and 2 positive edges."

  LLM polishes:
    "The analysis reveals one causal pathway. Less oxygen inhaled
     decreases oxygen in the lungs, which subsequently affects
     diffusion and ultimately the amount entering the bloodstream."

[Step 5: Reasoning]
  LLM prompt:
    QUESTION: ... affect MORE oxygen enters blood
    ANALYSIS: [description above]
    CHOICES: A/B/C

  LLM reasons:
    "Pathway starts with negative (NOT_RESULTS_IN). Less oxygen
     inhaled means less in lungs, less diffusion, less enters blood.
     Question asks about MORE oxygen, but effect is less.
     Answer: B. less"

  Output:
    {
      "predicted_answer": "less",
      "predicted_choice": "B",
      "confidence": "high",
      "reasoning": "..."
    }

[Verification]
  Predicted: "less"
  Actual: "less"
  ✓ Correct!
```

---

## Summary

这个系统的核心是**将符号推理（图搜索）与神经推理（LLM）结合**：

1. **符号部分**: BFS、路径提取、图结构
2. **神经部分**: 关系生成、语义匹配、最终推理
3. **桥梁**: 自然语言描述作为两者的接口

通过这种混合架构，系统既保持了结构化推理的可解释性，又具备了 LLM 的灵活性和语义理解能力。
