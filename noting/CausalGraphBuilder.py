import ollama
import re
import json
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import networkx as nx

class CausalGraphBuilder:
    """构建因果DAG的完整系统
    
    === 当前置信度计算（简化版）===
    
    1. A类三元组 (confidence = 1.0 固定)
       - 完全基于问题文本中的明确因果关系
       - 两端实体都在E_Q中
       - 有明确的文本证据
    
    2. B类三元组 (confidence = LLM评估 0.0-1.0)
       - 一端实体在E_Q中，另一端是LLM推荐的外部实体
       - 通过LLM评估合理性并返回置信度
    
    3. C类三元组 (confidence = 0.5 固定)
       - 一端是B类新实体，另一端是全新的桥接实体
    
    === 未来评分系统（待实现）===
    
    将来可集成更复杂的评分机制：
    - s_text: NLI模型验证证据强度（ENTAIL/NEU/CONTRA）
    - s_prior: LLM自一致性（多提示、多采样投票）
    - s_connect: 图算法计算与E_Q的连通性
    - s_iden: 可识别性助益（d-separation、背门/前门检测）
    - s_dir: 方向稳定度（因果触发词、时序、双向打分）
    
    通过 set_scorer() 方法可以注入外部评分器。
    
    === 约束说明 ===
    
    A类：head ∈ E_Q AND tail ∈ E_Q
    B类：(head ∈ E_Q AND tail ∉ E_Q) OR (head ∉ E_Q AND tail ∈ E_Q)
    C类：(head ∉ (E_Q ∪ B) AND tail ∈ B AND tail ∉ E_Q)
    """
    
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        # 英文因果触发词
        self.causal_triggers = [
            "because", "cause", "lead to", "result in", "due to", 
            "therefore", "thus", "consequently", "hence", "so",
            "trigger", "produce", "generate", "bring about", "give rise to",
            "contribute to", "lead", "result", "effect", "affect",
            "influence", "impact", "induce", "provoke", "stem from"
        ]
        
        # 预留：外部评分器（NLI、因果识别等）
        self.external_scorer = None
    
    def set_scorer(self, scorer):
        """设置外部评分器（为将来的NLI、因果识别等预留接口）
        
        评分器应该实现以下方法：
        - scorer.compute_score(triple_class, question, triple) -> float
        - scorer.score_evidence(question, triple) -> s_text
        - scorer.score_consistency(question, triple) -> s_prior  
        - scorer.score_direction(triple) -> s_dir
        - scorer.score_connectivity(graph, triple, eq_entities) -> s_connect
        - scorer.score_identifiability(graph, triple) -> s_iden
        
        Example:
            class ExternalScorer:
                def compute_score(self, triple_class, question, triple):
                    s_text = self.score_evidence(question, triple)
                    s_prior = self.score_consistency(question, triple)
                    s_dir = self.score_direction(triple)
                    # 综合评分
                    return weighted_average([s_text, s_prior, s_dir])
        """
        self.external_scorer = scorer
    
    def _compute_confidence(self, triple_class: str, question: str = None, 
                           triple: Dict = None, verification: Dict = None) -> float:
        """计算三元组的置信度
        
        当前版本：简化计算
        - A类: 1.0 (固定)
        - B类: LLM评估 (0.0-1.0) 或使用已有verification结果
        - C类: 0.5 (固定)
        
        未来版本：可集成外部评分器
        - 如果设置了 external_scorer，将调用其评分方法
        - 综合 s_text, s_prior, s_connect, s_iden, s_dir
        
        Args:
            triple_class: 'A', 'B', or 'C'
            question: 问题文本
            triple: 三元组字典
            verification: 已有的验证结果（用于B类，避免重复调用LLM）
        """
        # 如果有外部评分器，使用外部评分器
        if self.external_scorer and triple:
            return self.external_scorer.compute_score(triple_class, question, triple)
        
        # 否则使用简化版本
        if triple_class == 'A':
            return 1.0
        elif triple_class == 'C':
            return 0.5
        else:  # B类
            # 优先使用已有的verification结果（避免重复调用LLM）
            if verification and 'confidence' in verification:
                return verification['confidence']
            # 如果没有verification但有triple，重新验证
            elif triple and question:
                verification_result = self._verify_triple_evidence_relaxed(
                    question, triple['head'], triple['tail'], 'causes'
                )
                return verification_result['confidence']
            # 默认值
            return 0.6
        
    def extract_entities(self, question: str) -> Set[str]:
        """步骤1：从问题中抽取实体（E_Q）"""
        prompt = f"""Extract only entities from the question. 
An entity must be a noun or noun phrase that denotes an object, substance, place, organization, or named concept.
Do NOT include actions, events, properties, or adjectives.
Exclude gerunds/participles (words ending with “-ing”) unless they are part of a well-known multi-word noun (e.g., “machine learning”, “global warming”) or followed by a head noun (e.g., “freezing point”). 
If a word can be a verb or a noun (e.g., “freeze/freezing”), include it only when it functions as a noun with a head noun; otherwise exclude it.

Normalize to singular lemmas, remove duplicates, and keep multi-word noun phrases intact. 
Return ONLY the entity names, one per line, with no numbers, bullets, quotes, or explanations. 
If there are no entities, return nothing.

Question: {question}

Entities:"""
        
        response = ollama.generate(model=self.model_name, prompt=prompt)
        entities_text = response['response'].strip()
        
        # 解析实体
        entities = set()
        for line in entities_text.split('\n'):
            # 移除各种格式：编号、bullet、破折号等
            entity = line.strip()
            
            # 跳过空行
            if not entity or len(entity) < 2:
                continue
            
            # 跳过常见的提示词和标题
            if entity.lower() in ['entities:', 'entity list:', 'here is', 'here are', 'the entities are:', 'example output format:']:
                continue
            if entity.lower().startswith(('here', 'the following', 'entity', 'entities')):
                continue
            
            # 移除开头的编号 (1. 2. 3. 或 1) 2) 3))
            entity = re.sub(r'^\d+[\.)]\s*', '', entity)
            # 移除bullet points
            entity = entity.lstrip('-•*>').strip()
            # 移除引号
            entity = entity.strip('"\'')
            
            if entity and len(entity) > 1:
                entities.add(entity)
        
        return entities
    
    def extract_causal_triggers(self, text: str) -> List[Dict]:
        """抽取因果触发词及其位置"""
        triggers = []
        for trigger in self.causal_triggers:
            for match in re.finditer(trigger, text, re.IGNORECASE):
                triggers.append({
                    'trigger': trigger,
                    'start': match.start(),
                    'end': match.end()
                })
        return triggers
    
    def extract_class_a_triples(self, question: str, entities: Set[str]) -> List[Dict]:
        """步骤2：生成A类三元组（完全来自问题，且仅保留题内明确断言的因果事实）"""

        # ===== 基础准备 =====
        entities = set(entities)
        entity_lower_map = {e.lower(): e for e in entities}

        # 因果白名单 & 非因果黑名单
        ALLOWED_REL = {"causes", "increases", "decreases", "prevents", "enables", "needed_for"}
        FORBIDDEN_REL = {"contains", "part_of", "has", "is_a", "located_in", "example_of", "equals", "correlates_with"}

        # 真正“问句/设问”的触发词（不要把 suppose/assume 当问句）
        QUERY_PATTERNS = (
            "how will", "how would", "how does", "does ",  # 注意 `does ` 后有空格，避免误伤
            "what happens if", "有没有影响", "是否", "affect", "impact"
        )

        import re
        triple_line_re = re.compile(r'^\s*(?P<head>.+?)\s*->\s*(?P<rel>.+?)\s*->\s*(?P<tail>.+?)\s*\|\s*(?P<ev>.+?)\s*$')
        scoped_entity_re = re.compile(r'^(?P<base>[^{]+?)(?:\{(?P<scope>[^}]+)\})?$')

        def _parse_scoped_name(name: str):
            m = scoped_entity_re.match(name.strip())
            if not m:
                return name.strip(), None
            base = m.group("base").strip()
            scope = (m.group("scope") or "").strip() or None
            return base, scope

        def _is_query_evidence(ev_text: str) -> bool:
            ev_low = ev_text.lower()
            return any(pat in ev_low for pat in QUERY_PATTERNS)

        def _normalize_relation(rel: str) -> str:
            return rel.strip().lower().replace(" ", "_")

        # 宽松证据匹配（大小写不敏感、折叠空白、可去标点）
        def _normalize_txt(s: str) -> str:
            s = s.strip().lower()
            s = re.sub(r'\s+', ' ', s)
            s = re.sub(r'[^\w\s]', '', s)
            return s

        def _evidence_in_question(ev_text: str, q: str) -> bool:
            ev = ev_text.strip()
            if ev.startswith('"') and ev.endswith('"') and len(ev) >= 2:
                ev = ev[1:-1]
            return _normalize_txt(ev) in _normalize_txt(q)
        
        def _find_matching_entity(self, text: str, entities: Set[str], entity_lower_map: Dict[str, str]) -> Optional[str]:
            import re, difflib

            if not text:
                return None

            # 去掉可能的作用域标签： water{for=snow} -> water
            base = re.sub(r'\{[^}]*\}\s*$', '', text.strip()).strip()

            # 1) 精确匹配
            if base in entities:
                return base

            # 2) 小写匹配
            low = base.lower()
            if low in entity_lower_map:
                return entity_lower_map[low]

            # 3) 子串/包含匹配（宽松）
            for e in entities:
                el = e.lower()
                if low in el or el in low:
                    return e

            # 4) 近似匹配（避免轻微拼写差异）
            candidates = difflib.get_close_matches(base, list(entities), n=1, cutoff=0.88)
            if candidates:
                return candidates[0]

            return None

        # ===== 构造 Prompt =====
        entities_str = ", ".join(sorted(entities))  # 修正：不要直接打印 set
        prompt = f"""You are a causal graph extractor.

Goal
- From the question text, extract ONLY causal facts that are explicitly asserted or unambiguously entailed by in-sentence constructions (light linguistic entailment).
- Build a question-specific causal graph using ONLY the provided base entities.
- Do NOT infer an answer or invent facts. Do NOT output correlations or structural facts.

Entity scope & disambiguation
- If the same surface word refers to different contexts/roles, add a scope in braces after the base entity, e.g., water{{in=cube}}, water{{for=snow}}, water{{role=precursor}}.
- The base name MUST be one of the allowed entities; scopes are free-form annotations.
- Never merge nodes with different scopes unless the text explicitly states they are the same.

Use ONLY these base entities:
{entities_str}

Causal relations (normalize to this closed set ONLY)
- causes, increases, decreases, prevents, enables, needed_for

Linguistic patterns → relation mapping (must apply)
- "Y requires X", "Y needs X"                              → needed_for(X, Y)
- "for Y to happen, X", "for Y to VERB, X", "to VERB, X"   → needed_for(X, Y)
- "X leads to Y", "X results in Y", "X causes Y"           → causes
- "more X → more Y" / "less X → less Y"                    → increases / decreases
- "without X, Y cannot/does not happen"                    → prevents(X, Y) or needed_for(X, Y) (choose the closer wording)
- "X allows/enables/permits Y"                             → enables

Ignore NON-causal / structural patterns entirely
- has/have/contains, part_of, is_a, located_in, equals, example_of, correlates_with
- Structural links may be used only to craft scopes (e.g., water{{in=cube}}) but MUST NOT become causal triples.

Queries vs. facts
- Do NOT output triples for query forms or meta-questions:
  "how will X affect Y", "does X affect Y", "what happens if", "是否", "有没有影响", "影响吗", "affect", "impact".
- Words like "suppose", "assume", or "let's say" DO NOT by themselves invalidate asserted/entailed causal statements in the same sentence. Keep extracting if a causal pattern is present.

Evidence
- For every triple, provide a minimal exact substring from the question that contains the trigger phrase; wrap it in double quotes.

If NO causal facts satisfy the above, output exactly:
NONE

Question:
{question}

Return in this EXACT format (one per line), no numbering, no extra text:
head_entity -> relation -> tail_entity | "evidence_text"

Examples (follow EXACTLY; base entity must be from the list; scopes allowed):

Example 1
Question: "For plants to grow, water is needed."
Entities: water, plants
Triples:
water -> needed_for -> plants | "For plants to grow, water is needed."

Example 2
Question: "Less salt leads to lower blood pressure."
Entities: salt, blood pressure
Triples:
salt -> decreases -> blood pressure | "Less salt leads to lower blood pressure."

Example 3
Question: "Suppose the tray has water. For snow to form, water must freeze."
Entities: tray, water, snow
Triples:
water{{for=snow}} -> needed_for -> snow | "For snow to form, water must freeze."

Triples:
"""

        # ===== 生成与解析 =====
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            raw = response.get('response', '').strip()
            print(raw)
        except Exception as e:
            print(f"[extract_class_a_triples] LLM error: {e}")
            return []

        _idx = raw.lower().find("triples:")
        triples_text = raw[_idx + len("triples:"):] if _idx != -1 else raw
        triples_text = triples_text.strip()

        triples: List[Dict] = []
        if triples_text.lower() == "none":
            return triples

        for line in triples_text.splitlines():
            line = line.strip()
            if not line or "->" not in line or "|" not in line:
                continue

            m = triple_line_re.match(line)
            if not m:
                continue

            head_raw, rel_raw, tail_raw, ev_raw = m.group("head"), m.group("rel"), m.group("tail"), m.group("ev")

            # 证据中若是问句/设问触发，则忽略
            if _is_query_evidence(ev_raw):
                continue

            rel = _normalize_relation(rel_raw)
            if rel in FORBIDDEN_REL:
                continue
            if rel not in ALLOWED_REL:
                continue

            head_base, head_scope = _parse_scoped_name(head_raw)
            tail_base, tail_scope = _parse_scoped_name(tail_raw)

            head_match = self._find_matching_entity(head_base, entities, entity_lower_map)
            tail_match = self._find_matching_entity(tail_base, entities, entity_lower_map)
            if not head_match or not tail_match:
                continue

            if head_match == tail_match and (not head_scope and not tail_scope):
                continue

            if not _evidence_in_question(ev_raw, question):
                continue

            triple = {
                "head": head_match,
                "head_scope": head_scope,
                "relation": rel,
                "tail": tail_match,
                "tail_scope": tail_scope,
                "evidence": ev_raw.strip(),
                "class": "A",
            }

            conf = self._compute_confidence("A", question, triple)
            ev_low = ev_raw.lower()
            if triple["head_scope"] and any(k in ev_low for k in ["for", "in", "into", "from", "to"]):
                conf = min(1.0, conf + 0.05)
            if triple["tail_scope"] and any(k in ev_low for k in ["for", "in", "into", "from", "to"]):
                conf = min(1.0, conf + 0.05)
            triple["confidence"] = conf

            triples.append(triple)

        return triples

    def expand_class_b_triples(self, question: str, eq_entities: Set[str],
                               existing_triples: List[Dict], k: int = 3, debug: bool = False) -> List[Dict]:
        """步骤3：扩展B类三元组（问题实体 + 外部实体）
        约束：两端必须恰好一个在E_Q中；仅保留因果（causes）方向；自动修正LLM方向错误。
        """
        b_triples = []
        eq_entities_lower = {e.lower() for e in eq_entities}

        for entity in list(eq_entities)[:3]:
            prompt = f"""Task: Find NEW causal relationships for "{entity}"

Question: {question}

Already identified entities (DO NOT USE): {', '.join(eq_entities)}

Find NEW entities (NOT in the above list) in TWO categories:

━━━ CATEGORY 1: CAUSES ━━━
What NEW entities CAUSE "{entity}"?
Arrow direction: NEW_ENTITY -> causes -> {entity}

Format exactly:
X -> causes -> {entity}

━━━ CATEGORY 2: EFFECTS ━━━
What NEW entities are CAUSED BY "{entity}"?
Arrow direction: {entity} -> causes -> NEW_ENTITY

Format exactly:
{entity} -> causes -> Y

Return exactly {k} lines for CAUSES and {k} lines for EFFECTS (total {k*2}), no bullets, no numbering.
"""
            response = ollama.generate(model=self.model_name, prompt=prompt)
            candidates_text = response['response'].strip()

            if debug:
                print(f"\n[DEBUG] B类候选 for '{entity}':\n{candidates_text}")

            count = 0
            for raw_line in candidates_text.split('\n'):
                if count >= k * 2:
                    break
                line = raw_line.strip().strip('*').strip('-').strip()
                if '->' not in line:
                    continue

                try:
                    parts = [p.strip() for p in line.split('->')]
                    if len(parts) < 3:
                        continue
                    left, rel_word, right = parts[0].strip('"\' '), parts[1].lower(), parts[2].strip('"\' ')
                    # 只接受 causes
                    if "cause" not in rel_word:
                        continue

                    left_in_eq = left.lower() in eq_entities_lower
                    right_in_eq = right.lower() in eq_entities_lower

                    # 判别实际方向（期望：一端是 E_Q，另一端是新实体）
                    if left_in_eq and not right_in_eq:
                        # EFFECT: entity -> causes -> NEW
                        head, tail = left, right
                        actual_direction = 'effect'
                    elif not left_in_eq and right_in_eq:
                        # CAUSE: NEW -> causes -> entity
                        head, tail = left, right
                        actual_direction = 'cause'
                    elif not left_in_eq and not right_in_eq:
                        # 两端都不在E_Q，尝试模糊靠拢 entity
                        if entity.lower() in right.lower():
                            head, tail = left, entity
                            actual_direction = 'cause'
                        elif entity.lower() in left.lower():
                            head, tail = entity, right
                            actual_direction = 'effect'
                        else:
                            continue
                    else:
                        # 两端都在E_Q，这是A类
                        continue

                    # 重新规范：我们最终统一 relation 为 "causes"
                    verification = self._verify_triple_evidence_relaxed(
                        question, head, tail, actual_direction, debug=debug
                    )
                    if debug:
                        print(f"    候选: {head} -> causes -> {tail} | verify={verification['is_valid']}, conf={verification['confidence']:.2f}")

                    if not verification['is_valid']:
                        continue

                    # B类约束：恰好一端在E_Q
                    head_in_eq = head.lower() in eq_entities_lower
                    tail_in_eq = tail.lower() in eq_entities_lower
                    if (head_in_eq ^ tail_in_eq) is False:
                        if debug:
                            print("    ❌ 不满足B类约束（两端都在或都不在 E_Q）")
                        continue

                    triple = {
                        'head': head,
                        'relation': 'causes',
                        'tail': tail,
                        'evidence': verification['evidence'],
                        'class': 'B',
                        'confidence': self._compute_confidence('B', question, None, verification=verification)
                    }
                    b_triples.append(triple)
                    count += 1

                except Exception as e:
                    if debug:
                        print(f"    解析错误: {e}")
                    continue

        # 去方向冲突
        b_triples = self._check_direction_consistency(existing_triples + b_triples)
        # 只保留 B 类
        b_triples = [t for t in b_triples if t.get('class') == 'B']
        return b_triples
    
    def expand_class_c_triples(self, question: str, eq_entities: Set[str],
                               b_entities: Set[str], existing_triples: List[Dict],
                               max_path_length: int = 2, debug: bool = False) -> List[Dict]:
        """步骤4：扩展C类三元组（桥接实体）
        
        约束：三元组的实体有一个在B类新增的实体中（不在A类/E_Q中），另一个是全新的桥接实体
        """
        c_triples = []
        
        if not b_entities:
            if debug:
                print("[DEBUG] 没有B类实体，跳过C类扩展")
            return c_triples
        
        # 🔧 清理所有实体名称中的markdown符号
        b_entities_clean = {e.strip('*').strip() for e in b_entities}
        
        # 收集已知实体
        eq_entities_lower = {e.lower() for e in eq_entities}
        all_known_entities = eq_entities.union(b_entities_clean)
        all_known_lower = {e.lower() for e in all_known_entities}
        
        if debug:
            print(f"\n[DEBUG] C类扩展:")
            print(f"  B类实体（清理后）: {b_entities_clean}")
            print(f"  已知实体总数: {len(all_known_entities)}")
        
        # 扩展前4个B类实体（增加数量）
        for idx, b_entity in enumerate(list(b_entities_clean)[:4], 1):
            # 【关键】确保b_entity确实是B类新增的（不在E_Q中）
            if b_entity.lower() in eq_entities_lower:
                if debug:
                    print(f"\n  [{idx}] 跳过 '{b_entity}': 在E_Q中")
                continue
            
            if debug:
                print(f"\n  [{idx}] 为B类实体 '{b_entity}' 寻找桥接概念...")
            
            prompt = f"""Based on the question context, propose 2 NEW bridging concepts that connect "{b_entity}" to the causal chain. These should be intermediate concepts that are NOT already mentioned.

Question: {question}

Target entity: "{b_entity}" (from extended reasoning)

Already known entities (DO NOT USE): {', '.join(all_known_entities)}

Propose NEW bridging concepts that:
1. Have a causal relationship with "{b_entity}"
2. Help connect the causal chain
3. Are NOT in the known entities list above

Format: new_bridge_concept -> relation -> "{b_entity}"

Example:
Industrial activity -> contributes to -> Greenhouse gas emissions
Environmental policy -> influences -> Deforestation

Return only triples with NEW concepts, one per line:"""
            
            response = ollama.generate(model=self.model_name, prompt=prompt)
            candidates_text = response['response'].strip()
            
            if debug:
                print(f"  LLM响应:\n{candidates_text}")
            
            found_any = False
            for line in candidates_text.split('\n')[:3]:  # 增加到3行
                line = line.strip()
                if '->' in line:
                    try:
                        parts = [p.strip() for p in line.split('->')]
                        if len(parts) >= 3:
                            # 🔧 清理markdown符号
                            bridge_entity = parts[0].strip('"\'*').strip()
                            relation = parts[1].strip('*').strip()
                            target = parts[2].strip('"\'*').strip()
                            
                            if debug:
                                print(f"\n    解析: {bridge_entity} -> {relation} -> {target}")
                            
                            # 【关键约束1】确保bridge_entity是全新的（不在E_Q和B类中）
                            if bridge_entity.lower() in all_known_lower:
                                if debug:
                                    print(f"      ❌ 跳过：'{bridge_entity}' 已在已知实体中")
                                continue
                            
                            # 【关键约束2】确保target是B类实体（不在E_Q中）
                            target_in_b = target.lower() in {e.lower() for e in b_entities_clean}
                            target_in_eq = target.lower() in eq_entities_lower
                            
                            # 如果target解析不对，用当前的b_entity
                            if not target_in_b or target_in_eq:
                                if debug:
                                    print(f"      目标实体不匹配，使用当前B类实体: {b_entity}")
                                target = b_entity
                            
                            # 路径长度检查（暂时放宽）
                            # temp_graph = self._build_temp_graph(existing_triples + c_triples)
                            # path_ok = self._check_path_length(temp_graph, bridge_entity, 
                            #                           eq_entities, max_path_length)
                            path_ok = True  # 暂时跳过路径检查，先看能不能生成
                            
                            if debug:
                                print(f"      路径检查: {path_ok}")
                            
                            if path_ok:
                                # 【最终验证】确保C类约束：
                                # - head是新的桥接实体（不在E_Q和B中）
                                # - tail是B类实体（在B中但不在E_Q中）
                                head_is_new = bridge_entity.lower() not in all_known_lower
                                tail_is_b_class = (target.lower() in {e.lower() for e in b_entities_clean} and 
                                                  target.lower() not in eq_entities_lower)
                                
                                if debug:
                                    print(f"      验证C类约束:")
                                    print(f"        head_is_new ({bridge_entity}): {head_is_new}")
                                    print(f"        tail_is_b_class ({target}): {tail_is_b_class}")
                                
                                if head_is_new and tail_is_b_class:
                                    triple = {
                                        'head': bridge_entity,
                                        'relation': relation,
                                        'tail': target,
                                        'evidence': f"Bridging concept connecting to {target}",
                                        'class': 'C'
                                    }
                                    # 使用统一的置信度计算
                                    triple['confidence'] = self._compute_confidence('C', question, triple)
                                    c_triples.append(triple)
                                    found_any = True
                                    
                                    # 更新已知实体
                                    all_known_entities.add(bridge_entity)
                                    all_known_lower.add(bridge_entity.lower())
                                    
                                    if debug:
                                        print(f"      ✅ 添加C类三元组: {bridge_entity} -> {target}")
                                else:
                                    if debug:
                                        print(f"      ❌ 不满足C类约束")
                    except Exception as e:
                        if debug:
                            print(f"      ❌ 解析错误: {e}")
                        continue
            
            if debug and not found_any:
                print(f"    该B类实体未生成C类三元组")
        
        return c_triples
    
    def _verify_triple_evidence(self, question: str, entity1: str, 
                                entity2: str, direction: str) -> Dict:
        """验证三元组的证据（严格版本，用于A类）"""
        prompt = f"""Determine whether the following causal relationship is entailed or supported by the question text.

Question: {question}

Causal relationship: "{entity1}" {direction} "{entity2}"

Please answer:
1. Is it entailed (Yes/No)
2. Confidence level (0.0-1.0)
3. Supporting evidence (quote from the question)

Format:
Entailed: Yes/No
Confidence: 0.8
Evidence: ..."""
        
        response = ollama.generate(model=self.model_name, prompt=prompt)
        result_text = response['response'].strip()
        
        is_valid = "yes" in result_text.lower().split('\n')[0]
        
        # 解析置信度
        confidence = 0.5
        for line in result_text.split('\n'):
            if 'confidence' in line.lower():
                try:
                    confidence = float(re.findall(r'0\.\d+|1\.0', line)[0])
                except:
                    pass
        
        # 提取证据
        evidence = ""
        for line in result_text.split('\n'):
            if 'evidence' in line.lower():
                evidence = line.split(':', 1)[1].strip() if ':' in line else ""
                break
        
        return {
            'is_valid': is_valid and confidence > 0.5,
            'confidence': confidence,
            'evidence': evidence
        }
    
    def _verify_triple_evidence_relaxed(self, question: str, entity1: str, 
                                        entity2: str, direction: str, debug: bool = False) -> Dict:
        """验证三元组的合理性（宽松版本，用于B/C类 - 允许外部推理）"""
        prompt = f"""Based on the question context and common sense, determine if the following causal relationship is reasonable and relevant.

Question context: {question}

Proposed causal relationship: "{entity1}" {direction} "{entity2}"

The entity "{entity1}" may NOT appear directly in the question - that's OK. Judge based on:
1. Is this relationship logically reasonable given the question context?
2. Does it help explain or expand the causal chain in the question?

Please answer:
1. Is it reasonable (Yes/No)
2. Confidence level (0.0-1.0)
3. Brief reasoning

Format:
Reasonable: Yes/No
Confidence: 0.7
Reasoning: ..."""
        
        response = ollama.generate(model=self.model_name, prompt=prompt)
        result_text = response['response'].strip()
        
        if debug:
            print(f"\n[DEBUG] 验证 '{entity1}' {direction} '{entity2}':")
            print(f"LLM响应:\n{result_text}\n")
        
        # 宽松的判断：只要不是明确的"No"就接受
        first_line = result_text.lower().split('\n')[0] if result_text else ""
        is_valid = "yes" in first_line or "reasonable" in first_line
        
        # 如果第一行没有明确答案，检查整个响应
        if not is_valid:
            is_valid = "yes" in result_text.lower() and "no" not in first_line
        
        # 解析置信度（B类的默认置信度稍低）
        confidence = 0.6  # 默认值
        for line in result_text.split('\n'):
            if 'confidence' in line.lower():
                try:
                    # 尝试提取数字
                    numbers = re.findall(r'0\.\d+|1\.0|1', line)
                    if numbers:
                        confidence = float(numbers[0])
                except:
                    pass
        
        # 提取推理
        reasoning = ""
        for line in result_text.split('\n'):
            if 'reasoning' in line.lower():
                reasoning = line.split(':', 1)[1].strip() if ':' in line else ""
                break
        
        if not reasoning:
            # 如果没找到Reasoning行，使用整个响应的后半部分
            lines = result_text.split('\n')
            if len(lines) > 2:
                reasoning = ' '.join(lines[2:])
        
        # 🔧 降低阈值：从0.4降到0.3，更容易接受
        final_is_valid = is_valid and confidence > 0.3
        
        if debug:
            print(f"解析结果: is_valid={is_valid}, confidence={confidence}, final={final_is_valid}")
        
        return {
            'is_valid': final_is_valid,
            'confidence': confidence,
            'evidence': reasoning or f"External reasoning: {entity1} {direction} {entity2}"
        }
    
    def _check_direction_consistency(self, triples: List[Dict]) -> List[Dict]:
        """检查方向一致性，解决冲突"""
        # 构建实体对的方向映射
        direction_map = defaultdict(list)
        
        for triple in triples:
            key = tuple(sorted([triple['head'], triple['tail']]))
            direction_map[key].append(triple)
        
        # 检查冲突
        consistent_triples = []
        for key, candidates in direction_map.items():
            if len(candidates) == 1:
                consistent_triples.append(candidates[0])
            else:
                # 有冲突，保留置信度最高的
                best = max(candidates, key=lambda x: x.get('confidence', 0.5))
                consistent_triples.append(best)
        
        return consistent_triples
    
    def _build_temp_graph(self, triples: List[Dict]) -> nx.DiGraph:
        """构建临时图用于路径分析"""
        G = nx.DiGraph()
        for triple in triples:
            G.add_edge(triple['head'], triple['tail'], 
                      relation=triple['relation'])
        return G
    
    def _check_path_length(self, graph: nx.DiGraph, node: str, 
                          target_nodes: Set[str], max_length: int) -> bool:
        """检查节点到目标节点的最短路径"""
        if node in target_nodes:
            return True
        
        for target in target_nodes:
            if target in graph:
                try:
                    path_length = nx.shortest_path_length(
                        graph.to_undirected(), node, target
                    )
                    if path_length <= max_length:
                        return True
                except nx.NetworkXNoPath:
                    continue
        
        return False
    
    def build_dag(self, triples: List[Dict]) -> nx.DiGraph:
        """步骤6：构建DAG并处理环"""
        G = nx.DiGraph()
        
        # 按类别和置信度排序
        class_priority = {'A': 3, 'B': 2, 'C': 1}
        sorted_triples = sorted(
            triples, 
            key=lambda x: (class_priority[x['class']], x.get('confidence', 0.5)),
            reverse=True
        )
        
        for triple in sorted_triples:
            G.add_edge(
                triple['head'], 
                triple['tail'],
                relation=triple['relation'],
                evidence=triple['evidence'],
                class_type=triple['class'],
                confidence=triple.get('confidence', 0.5)
            )
            
            # 检查是否形成环
            if not nx.is_directed_acyclic_graph(G):
                # 找到环并删除最低分的边
                cycles = list(nx.simple_cycles(G))
                if cycles:
                    cycle = cycles[0]
                    # 删除环中置信度最低的边
                    min_edge = None
                    min_conf = float('inf')
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i+1) % len(cycle)]
                        if G.has_edge(u, v):
                            conf = G[u][v].get('confidence', 0.5)
                            if conf < min_conf:
                                min_conf = conf
                                min_edge = (u, v)
                    
                    if min_edge:
                        G.remove_edge(*min_edge)
        
        return G
    
    def visualize_graph(self, G: nx.DiGraph) -> str:
        """可视化图结构（文本形式）"""
        output = ["=== Causal Relationship Graph ===\n"]
        output.append(f"Nodes: {G.number_of_nodes()}")
        output.append(f"Edges: {G.number_of_edges()}\n")
        
        for u, v, data in G.edges(data=True):
            output.append(
                f"{u} --[{data.get('relation', 'relation')}]--> {v} "
                f"(Class: {data.get('class_type', 'N/A')}, "
                f"Confidence: {data.get('confidence', 0.5):.2f})"
            )
        
        return '\n'.join(output)
    
    def process_question(self, question: str, choices: List[str] = None, debug: bool = False) -> Dict:
        """完整处理流程"""
        print("Step 1: Extracting entities...")
        eq_entities = self.extract_entities(question)
        print(f"Found {len(eq_entities)} entities: {eq_entities}\n")
        
        print("Step 2: Extracting Class A triples...")
        a_triples = self.extract_class_a_triples(question, eq_entities)
        print(f"Found {len(a_triples)} Class A triples\n")
        
        print("Step 3: Expanding Class B triples...")
        b_triples = self.expand_class_b_triples(question, eq_entities, a_triples, k=2, debug=debug)
        
        # 收集B类新增的实体（不在E_Q中的实体）
        eq_entities_lower = {e.lower() for e in eq_entities}
        b_entities = set()
        for t in b_triples:
            # 检查head是否是新实体
            if t['head'].lower() not in eq_entities_lower:
                b_entities.add(t['head'])
            # 检查tail是否是新实体
            if t['tail'].lower() not in eq_entities_lower:
                b_entities.add(t['tail'])
        
        print(f"Found {len(b_triples)} Class B triples, added {len(b_entities)} new entities")
        if b_entities:
            print(f"New entities: {b_entities}\n")
        else:
            print()
        
        print("Step 4: Expanding Class C triples...")
        c_triples = self.expand_class_c_triples(
            question, eq_entities, b_entities, a_triples + b_triples, max_path_length=2, debug=debug
        )
        print(f"Found {len(c_triples)} Class C triples\n")
        
        print("Step 5: Building DAG...")
        all_triples = a_triples + b_triples + c_triples
        graph = self.build_dag(all_triples)
        
        return {
            'entities': eq_entities,
            'b_entities': b_entities,  # 添加B类新实体信息
            'triples': {
                'A': a_triples,
                'B': b_triples,
                'C': c_triples
            },
            'graph': graph,
            'visualization': self.visualize_graph(graph)
        }


# ===== 使用示例 =====
if __name__ == "__main__":
    # 示例问题（英文）
    question = """Climate change leads to increased extreme weather events. These extreme weather 
    events damage crop growth, thus affecting food production. Reduced food production leads to 
    price increases, ultimately impacting people's quality of life."""
    
    # 创建构建器
    builder = CausalGraphBuilder(model_name="gemma2:27b")
    
    # 处理问题
    result = builder.process_question(question)
    
    # 输出结果
    print("\n" + "="*60)
    print(result['visualization'])
    print("\n" + "="*60)
    print(f"\nTotal: {len(result['triples']['A'])} Class A, "
          f"{len(result['triples']['B'])} Class B, "
          f"{len(result['triples']['C'])} Class C triples")