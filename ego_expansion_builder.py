"""
Ego-Expansion 因果图构建器（改进版）
=============================================
核心理念：基于实体集合扩散，构建带关系描述的因果图

主要改进：
1. 使用实体集合引导扩散，而非原始问题
2. 生成具体的关系类型和描述
3. 构建更丰富的三元组（head, relation, tail, description）
"""

import ollama
import os
import unicodedata
import re
from typing import List, Dict, Set, Tuple, Optional, Union
from datetime import datetime
from collections import defaultdict, deque
import json


class EgoExpansionCausalBuilder:
    """
    基于实体集合的ego-expansion因果图构建器
    
    通过从种子实体出发，基于已有实体集合引导扩展，构建带描述的因果关系
    """
    
    def __init__(
        self,
        model_name: str = "gemma2:27b",
        max_neighbors_per_seed: int = 3,
        max_expansion_depth: int = 2,
        max_relations_per_entity: int = 3,
        ascii_logs: bool | None = None,
        verbose: bool = True,
    ):
        """
        初始化构建器

        Args:
            model_name: 使用的LLM模型名称
            max_neighbors_per_seed: 每个种子实体最多扩展的邻居数
            max_expansion_depth: 最大扩展深度
            max_relations_per_entity: 每个实体最多的关系数
            verbose: 是否打印日志到控制台
        """
        self.model_name = model_name
        self.max_neighbors = max_neighbors_per_seed
        self.max_depth = max_expansion_depth
        self.max_relations = max_relations_per_entity
        self.verbose = verbose
        self.log = []
        # 控制是否将日志强制转换为 ASCII，避免控制台/工具不支持 UTF-8 时出现乱码
        if ascii_logs is None:
            self.ascii_logs = os.environ.get("ASCII_LOGS", "0") == "1"
        else:
            self.ascii_logs = bool(ascii_logs)
        # 在最终回合补全潜在遗漏关系的数量上限
        self.max_backfill_relations = 10
    
    def _log(self, msg: str) -> None:
        """记录日志信息（可选 ASCII 降级以避免乱码）"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        raw = f"[{timestamp}] {msg}"
        if self.ascii_logs:
            line = msg.strip()
            # 如果整行是由非 ASCII 字符组成的分隔符，用 '=' 替代
            if line and all(ord(ch) > 127 for ch in line) and len(set(line)) == 1 and len(line) >= 5:
                raw = f"[{timestamp}] " + ("=" * 60)
            else:
                try:
                    raw = unicodedata.normalize('NFKD', raw).encode('ascii', 'ignore').decode('ascii')
                except Exception:
                    raw = ''.join(ch if ord(ch) < 128 else '?' for ch in raw)
        self.log.append(raw)
        if self.verbose:
            print(raw)
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM获取响应"""
        response = ollama.generate(model=self.model_name, prompt=prompt)
        return response['response'].strip()
    
    def _clean_entity(self, entity: str) -> str:
        """清理实体名称，去除多余的格式字符"""
        # 去除编号、标点等
        entity = re.sub(r'^\d+[\.)]\s*', '', entity)
        entity = re.sub(r'^[-•*>"\'\s]+', '', entity)
        entity = re.sub(r'[-•*>"\'\s]+$', '', entity)
        return entity.strip()
    
    def _is_valid_entity(self, entity: str, known_entities: Set[str]) -> bool:
        """
        检查实体是否有效
        
        Args:
            entity: 待检查的实体
            known_entities: 已知实体集合
            
        Returns:
            是否为有效实体
        """
        # 基本长度检查
        if not entity or len(entity) <= 2:
            return False
        
        # 检查是否已存在（忽略大小写）
        if entity.lower() in {e.lower() for e in known_entities}:
            return False
        
        # 过滤掉明显的非实体内容
        invalid_phrases = [
            'here are', 'causes:', 'none', 'no ', 'not ', 
            'there are', 'based on', 'since', 'if ', 'entity', 
            'entities', 'example', 'such as', 'including', 'the following',
            'relationship', 'relation', 'triple', 'description'
        ]
        
        entity_lower = entity.lower()
        for phrase in invalid_phrases:
            if phrase in entity_lower:
                return False
        
        return True
    
    def _parse_relation_response(self, response_text: str, seed: str, known_entities: Set[str]) -> List[Dict]:
        """
        解析LLM返回的关系三元组
        
        Args:
            response_text: LLM响应文本
            seed: 当前种子实体
            known_entities: 已知实体集合
            
        Returns:
            解析后的关系列表
        """
        relations = []
        
        # 尝试用JSON格式解析
        try:
            # 寻找JSON数组
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_relations = json.loads(json_str)
                
                for rel in parsed_relations:
                    if isinstance(rel, dict):
                        head = self._clean_entity(str(rel.get('head', '')))
                        tail = self._clean_entity(str(rel.get('tail', '')))
                        relation = rel.get('relation', 'causes')
                        description = rel.get('description', '')
                        # 可选置信度
                        conf = rel.get('confidence', None)
                        try:
                            if conf is not None:
                                conf = float(conf)
                                if conf > 1.0:
                                    conf = conf / 100.0
                                conf = max(0.0, min(1.0, conf))
                        except Exception:
                            conf = None
                        
                        # 验证实体
                        if head and tail and head != tail:
                            # 至少有一个实体应该是seed或已知实体
                            if head == seed or tail == seed or head in known_entities or tail in known_entities:
                                item = {
                                    'head': head,
                                    'tail': tail,
                                    'relation': relation,
                                    'description': description
                                }
                                if conf is not None:
                                    item['confidence'] = conf
                                relations.append(item)
        except:
            pass
        
        # 如果JSON解析失败，尝试文本解析
        if not relations:
            lines = response_text.split('\n')
            current_triple = {}
            
            for line in lines:
                line = line.strip()
                
                # 尝试匹配不同格式
                # 格式1: (head, relation, tail)
                triple_match = re.match(r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)', line)
                if triple_match:
                    head = self._clean_entity(triple_match.group(1))
                    relation = triple_match.group(2).strip()
                    tail = self._clean_entity(triple_match.group(3))
                    
                    if head and tail and head != tail:
                        current_triple = {
                            'head': head,
                            'tail': tail,
                            'relation': relation,
                            'description': ''
                        }
                        relations.append(current_triple)
                    continue
                
                # 格式2: head -> relation -> tail
                arrow_match = re.match(r'([^-]+)\s*->\s*([^-]+)\s*->\s*(.+)', line)
                if arrow_match:
                    head = self._clean_entity(arrow_match.group(1))
                    relation = arrow_match.group(2).strip()
                    tail = self._clean_entity(arrow_match.group(3))
                    
                    if head and tail and head != tail:
                        current_triple = {
                            'head': head,
                            'tail': tail,
                            'relation': relation,
                            'description': ''
                        }
                        relations.append(current_triple)
                    continue
                
                # 检查是否是描述行
                if 'description:' in line.lower() and current_triple:
                    desc = line.split(':', 1)[-1].strip()
                    if relations and not relations[-1].get('description'):
                        relations[-1]['description'] = desc
                if 'confidence:' in line.lower() and current_triple:
                    try:
                        val = line.split(':', 1)[-1].strip()
                        val_num = re.sub(r'[^0-9.]+', '', val)
                        conf = float(val_num) if val_num else None
                        if conf is not None:
                            if conf > 1.0:
                                conf = conf / 100.0
                            conf = max(0.0, min(1.0, conf))
                            if relations and relations[-1].get('confidence') is None:
                                relations[-1]['confidence'] = conf
                    except Exception:
                        pass
        
        return relations
    
    # ==================== 核心功能方法 ====================

    def identify_observable_outcome(
        self,
        question: str,
        intervention: Optional[str] = None,
        target: Optional[str] = None
    ) -> Dict[str, str]:
        """
        识别问题中的具体观察对象/可测量结果

        在提取种子之前调用，用于明确像"MORE vegetables"这样的模糊表述的具体含义。

        Args:
            question: 输入问题文本
            intervention: 干预实体（如果已知）
            target: 目标实体（如果已知）

        Returns:
            包含observable_outcome和reasoning的字典
            例如: {
                'observable_outcome': 'eating more vegetables',
                'reasoning': 'Given the intervention is stomach bug, which affects digestion...'
            }
        """
        self._log("\n" + "="*60)
        self._log("STEP 0: Identify Observable Outcome")
        self._log("="*60)

        # 构建提示词
        prompt = f"""You are a causal reasoning expert. Your task is to identify the SPECIFIC OBSERVABLE OUTCOME being asked about in this question.

QUESTION:
{question}
"""

        if intervention:
            prompt += f"\nKNOWN INTERVENTION: {intervention}"
        if target:
            prompt += f"\nKNOWN TARGET ENTITY: {target}"

        prompt += """

TASK:
Clarify what SPECIFIC, MEASURABLE outcome or behavior is being asked about.

Examples:
- "How will it affect MORE vegetables" could mean:
  * Eating more vegetables? (dietary behavior)
  * Growing more vegetables? (agricultural production)
  * Buying more vegetables? (purchasing behavior)

- "How will it affect LESS water" could mean:
  * Using less water? (consumption)
  * Having less water available? (supply/availability)
  * Producing less water? (generation)

Consider:
1. The context of the intervention (what domain does it relate to?)
2. The most direct and plausible interpretation
3. What would be observable or measurable

Return ONLY a JSON object with two fields:
{
  "observable_outcome": "the specific, concrete outcome (e.g., 'eating more vegetables')",
  "reasoning": "brief explanation of why this interpretation makes sense given the intervention context"
}"""

        # 获取LLM响应
        response_text = self._call_llm(prompt)
        self._log(f"\nLLM Response:\n{response_text}")

        # 解析JSON响应
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{[^{}]*"observable_outcome"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                observable_outcome = result.get('observable_outcome', '').strip()
                reasoning = result.get('reasoning', '').strip()
            else:
                # 降级：尝试从文本中提取
                observable_outcome = ''
                reasoning = response_text

                # 查找可能的结果描述
                for line in response_text.split('\n'):
                    if 'observable_outcome' in line.lower() or 'outcome:' in line.lower():
                        # 提取冒号后的内容
                        parts = re.split(r'[:：]', line, 1)
                        if len(parts) > 1:
                            observable_outcome = parts[1].strip().strip('"\'')
                            break

                if not observable_outcome:
                    # 最后降级：使用原始target
                    observable_outcome = target if target else "the target outcome"
                    reasoning = "Could not parse LLM response, using default"

            result_dict = {
                'observable_outcome': observable_outcome,
                'reasoning': reasoning
            }

            self._log(f"\n✅ Observable Outcome: {observable_outcome}")
            self._log(f"   Reasoning: {reasoning[:200]}...")

            return result_dict

        except Exception as e:
            self._log(f"⚠️ Error parsing observable outcome: {e}")
            # 返回默认值
            return {
                'observable_outcome': target if target else "the target outcome",
                'reasoning': f"Error parsing LLM response: {str(e)}"
            }

    def extract_seeds(
        self,
        question: str,
        observable_outcome: Optional[str] = None
    ) -> Set[str]:
        """
        步骤1：从问题中抽取种子实体

        Args:
            question: 输入问题文本
            observable_outcome: 可观察结果的具体描述（用于引导种子提取）

        Returns:
            种子实体集合
        """
        self._log("\n" + "="*60)
        self._log("STEP 1: Extract Seed Entities")
        self._log("="*60)

        # 构建提示词
        prompt = f"""Extract ONLY concrete entities (nouns/noun phrases) from this question.
Exclude: actions, events, properties, adjectives, gerunds, verbs.

Question: {question}
"""

        # 如果提供了observable_outcome，添加上下文信息
        if observable_outcome:
            prompt += f"""
OBSERVABLE OUTCOME CONTEXT:
The question is asking about: {observable_outcome}

When extracting entities, consider entities that are RELEVANT to this specific outcome.
For example:
- If observing "eating vegetables" → include entities like: digestion, nutrients, dietary fiber, appetite
- If observing "growing vegetables" → include entities like: soil, farmers, land, crops, agriculture
- If observing "buying vegetables" → include entities like: market, prices, consumers, supply

Focus on entities that could causally relate to: {observable_outcome}
"""

        prompt += """
Return entity names only, one per line. Be specific and concrete:"""
        
        # 获取LLM响应
        response_text = self._call_llm(prompt)
        self._log(f"\nLLM Response:\n{response_text}")
        
        # 解析实体
        seeds = set()
        for line in response_text.split('\n'):
            entity = self._clean_entity(line)
            if self._is_valid_entity(entity, seeds):
                seeds.add(entity)
        
        # 保留全部 LLM 提取到的种子（仅做清洗/去重），不做语义筛选
        return seeds
    
    def expand_causal_relations(
        self,
        seed: str,
        entity_context: Set[str],
        existing_relations: List[Dict],
        observable_outcome: Optional[str] = None
    ) -> List[Dict]:
        """
        步骤2：基于实体集合扩散因果关系

        Args:
            seed: 当前要扩展的实体
            entity_context: 当前已知的所有实体集合（用于引导扩散方向）
            existing_relations: 已存在的关系列表（用于避免重复）
            observable_outcome: 可观察结果（用于引导因果关系生成）

        Returns:
            新的因果关系三元组列表，每个包含：
            - head: 头实体
            - relation: 关系类型
            - tail: 尾实体
            - description: 关系描述
        """

        # 构建已存在关系的表示，用于去重
        existing_pairs = set()
        for rel in existing_relations:
            existing_pairs.add((rel['head'], rel['tail']))
            existing_pairs.add((rel['tail'], rel['head']))  # 双向检查

        # 构建提示词
        prompt = f"""You are building a causal knowledge graph. Given the entity "{seed}" and a set of related entities, generate DIRECT causal relationships.

Current entity: {seed}
Available entities in the graph: {', '.join(list(entity_context)[:20])}
"""

        # 如果提供了observable_outcome，添加上下文信息
        if observable_outcome:
            prompt += f"""
OBSERVABLE OUTCOME CONTEXT:
The ultimate goal is to understand causal relationships related to: {observable_outcome}

When generating causal relations for "{seed}", prioritize relationships that are RELEVANT to this outcome.

Examples:
- If outcome is "eating vegetables" and seed is "stomach bug":
  → Focus on: digestion, appetite, nausea, dietary fiber, nutrients
  → Avoid: unrelated agricultural or market concepts

- If outcome is "growing vegetables" and seed is "soil":
  → Focus on: nutrients, fertility, crops, farmers, land
  → Avoid: unrelated dietary or health concepts

Generate causal relations that help explain how entities connect to: {observable_outcome}
"""

        prompt += f"""
Generate up to {self.max_relations} causal relationship triples involving "{seed}".

CRITICAL Requirements:
1. Focus on IMMEDIATE and DIRECT causal relationships only:
   - Each relationship should represent a single causal step
   - DO NOT skip intermediate causes or effects
   - If A affects C through B, create (A→B) and (B→C), NOT (A→C)
   
2. Identify the PROXIMATE (nearest) causes and effects:
   - What DIRECTLY and IMMEDIATELY causes "{seed}"? (not distant root causes)
   - What does "{seed}" DIRECTLY and IMMEDIATELY cause? (not final outcomes)
   - Think of causal chains as step-by-step processes
   
3. Be SPECIFIC about causal mechanisms:
   - Each triple should represent ONE clear causal mechanism
   - Avoid vague or overly broad relationships
   - Focus on observable, concrete causal links
   
4. Types of valid DIRECT causal relationships:
   - causes (direct causation)
   - triggers (immediate activation)
   - produces (direct production)
   - generates (immediate generation)
   - increases/decreases (direct quantitative effect)
   - enables/prevents (direct facilitation/blocking)
   
5. For each triple, provide:
   - head: the direct cause/source
   - relation: specific causal verb
   - tail: the immediate effect/result
   - description: explain the DIRECT mechanism (not indirect pathways)
   - confidence: model-estimated confidence in [0.0, 1.0] (two decimals)

Important: Think step-by-step about causation. If there's an intermediate step, include it as a separate relationship rather than jumping from start to end.

Return as a JSON array:
[
  {{
    "head": "entity1",
    "relation": "causes",
    "tail": "entity2",
    "description": "How entity1 DIRECTLY causes entity2 (one step)",
    "confidence": 0.82
  }},
  ...
]

Generate precise, step-by-step causal relationships:"""

        # Encourage generation of novel intermediates that are not yet in the graph
        known_list = ', '.join(list(entity_context)[:20])
        min_novel = max(1, min(self.max_relations - 1, 2))
        prompt += f"""

NOVELTY REQUIREMENT:
- At least {min_novel} triples MUST introduce a previously unseen intermediate entity NOT in the known list: {known_list or '(none)'} (except "{seed}").
- Anchor each triple to "{seed}" (either head or tail must equal "{seed}").
- Prefer concrete scientific intermediates (e.g., temperature, pressure, rate, concentration, energy, voltage, resistance, density).
"""
        


        # 获取LLM响应（使用新提示）
        response_text = self._call_llm(prompt)
        self._log(f"LLM Response:\n{response_text[:500]}...")
        
        # 解析关系
        new_relations = self._parse_relation_response(response_text, seed, entity_context)
        
        # 过滤重复关系
        filtered_relations = []
        for rel in new_relations:
            # 跳过占位的无关系项，避免将其加入图
            if str(rel.get('relation', '')).strip().lower() == 'no_relation':
                continue
            pair = (rel['head'], rel['tail'])
            reverse_pair = (rel['tail'], rel['head'])
            
            if pair not in existing_pairs and reverse_pair not in existing_pairs:
                filtered_relations.append(rel)
                existing_pairs.add(pair)
        
        # 仅进行存在性去重与数量截断；不做语义打分/筛选
        # Prioritize relations that introduce novel entities; fallback to force novelty once if needed
        def _introduces_novel_entity(rel: Dict) -> bool:
            h = str(rel.get('head','')).strip()
            t = str(rel.get('tail','')).strip()
            return (h and h not in entity_context and h != seed) or (t and t not in entity_context and t != seed)

        prioritized = sorted(filtered_relations, key=lambda r: (not _introduces_novel_entity(r),))

        if not any(_introduces_novel_entity(r) for r in prioritized):
            known_list = ', '.join(list(entity_context)[:20])
            fallback_prompt = (
                f"Generate up to {self.max_relations} DIRECT causal triples that each include at least one NEW intermediate "
                f"entity not in: {known_list or '(none)'} (except '{seed}').\n"
                f"Anchor each triple to '{seed}' on either head or tail. Use JSON array with fields: "
                f"head, relation, tail, description, confidence (0.0-1.0)."
            )
            response_text2 = self._call_llm(fallback_prompt)
            self._log(f"Fallback LLM Response:\n{response_text2[:500]}...")
            fallback_rels = self._parse_relation_response(response_text2, seed, entity_context)
            for rel in fallback_rels:
                if str(rel.get('relation', '')).strip().lower() == 'no_relation':
                    continue
                pair = (rel.get('head'), rel.get('tail'))
                reverse_pair = (rel.get('tail'), rel.get('head'))
                if pair not in existing_pairs and reverse_pair not in existing_pairs:
                    prioritized.append(rel)
                    existing_pairs.add(pair)

        out = prioritized[: self.max_relations]
        novel_cnt = sum(1 for r in out if _introduces_novel_entity(r))
        self._log(f"Generated {len(out)} relations (novel_entities={novel_cnt})")
        return out

    def infer_missing_relations(
        self,
        entities: Set[str],
        existing_relations: List[Dict],
    ) -> List[Dict]:
        """
        额外步骤：在已知实体之间查找遗漏的直接因果关系（使用同一 LLM）。

        仅在给定实体集合内寻找关系，不引入新实体；返回的数量不超过
        self.max_backfill_relations，并保持与 expand 阶段相同的数据结构。
        """
        if not entities:
            return []

        # 已有有向对，用于避免重复
        existing_pairs = set()
        for rel in existing_relations:
            try:
                existing_pairs.add((rel['head'], rel['tail']))
            except Exception:
                continue

        # 构造提示：给出实体清单与已知关系，请求返回缺失的直接关系
        entities_list = ", ".join(sorted(list(entities))[:50])
        known_lines = []
        for rel in existing_relations[:100]:  # 控制提示长度
            head = rel.get('head', '')
            relation = rel.get('relation', '')
            tail = rel.get('tail', '')
            if head and tail:
                known_lines.append(f"- {head} -> {relation} -> {tail}")
        known_block = "\n".join(known_lines) if known_lines else "(none)"

        prompt = f"""You are completing a causal knowledge graph.

Entities (use ONLY these; do NOT invent new entities):
{entities_list}

Existing DIRECT relationships (avoid duplicates of these):
{known_block}

Task: Identify additional MISSING DIRECT causal relationships BETWEEN THE GIVEN ENTITIES.
Constraints:
1) Direct, single-step causal links only (no multi-hop or vague relations)
2) Head and tail must both be from the given entity list
3) Avoid duplicates of existing links
4) Return at most {self.max_backfill_relations} relations
5) Include a model-estimated confidence in [0.0, 1.0]

Return as a JSON array like:
[
  {{"head": "A", "relation": "causes", "tail": "B", "description": "...", "confidence": 0.78}},
  ...
]
"""

        try:
            response_text = self._call_llm(prompt)
        except Exception as e:
            self._log(f"Backfill LLM call failed: {e}")
            return []

        # 复用解析器；seed 无需限定，只要求属于实体集合
        new_relations = self._parse_relation_response(response_text, seed="", known_entities=entities)

        # 过滤：仅保留实体集内、且未出现过的有向边
        results: List[Dict] = []
        for rel in new_relations:
            head = rel.get('head')
            tail = rel.get('tail')
            if head in entities and tail in entities and (head, tail) not in existing_pairs and head != tail:
                results.append(rel)
                existing_pairs.add((head, tail))

        if self.max_backfill_relations is not None and self.max_backfill_relations > 0:
            results = results[: int(self.max_backfill_relations)]

        self._log(f"Backfilled {len(results)} missing relations among known entities")
        return results
    
    def _build_edges_from_relations(
        self, 
        relations: List[Dict], 
        depth: int
    ) -> Tuple[List[Dict], Set[str]]:
        """
        从关系列表构建边和新实体
        
        Args:
            relations: 关系列表
            depth: 当前深度
            
        Returns:
            (边列表, 新实体集合)
        """
        edges = []
        new_entities = set()
        
        for rel in relations:
            edge = {
                'head': rel['head'],
                'tail': rel['tail'],
                'relation': rel['relation'],
                'description': rel['description'],
                'confidence': rel.get('confidence'),
                'depth': depth
            }
            edges.append(edge)
            
            # 收集新实体
            new_entities.add(rel['head'])
            new_entities.add(rel['tail'])
        
        return edges, new_entities
    
    def _extract_chains(
        self, 
        edges: List[Dict], 
        seeds: Set[str]
    ) -> List[List[str]]:
        """
        从边集合中提取因果链
        
        Args:
            edges: 边的列表
            seeds: 种子实体集合
            
        Returns:
            因果链列表
        """
        # 构建邻接表
        graph = defaultdict(list)
        for edge in edges:
            graph[edge['head']].append(edge['tail'])
        
        # 使用BFS找因果链
        chains = []
        
        for seed in seeds:
            # BFS搜索
            queue = deque([(seed, [seed])])
            visited = {seed}
            
            while queue:
                node, path = queue.popleft()
                
                # 如果路径足够长，记录它
                if len(path) >= 2:
                    chains.append(path)
                
                # 限制链长度
                if len(path) >= 5:
                    continue
                
                # 扩展路径
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        # 去重并排序
        unique_chains = []
        seen = set()
        
        for chain in sorted(chains, key=len, reverse=True):
            chain_key = tuple(chain)
            if chain_key not in seen:
                seen.add(chain_key)
                unique_chains.append(chain)
        
        return unique_chains[:10]  # 返回最多10条链
    
    # ==================== 主要流程方法 ====================
    
    def build_causal_chain(
        self,
        question: str,
        intervention: Optional[str] = None,
        target: Optional[str] = None
    ) -> Dict:
        """
        主要构建流程：从问题构建完整的因果图

        Args:
            question: 输入问题文本
            intervention: 干预实体（可选，用于识别观察对象）
            target: 目标实体（可选，用于识别观察对象）

        Returns:
            包含实体、边、链、observable_outcome等信息的结果字典
        """
        # Step 0: 识别观察对象（在提取种子之前）
        observable_info = self.identify_observable_outcome(
            question=question,
            intervention=intervention,
            target=target
        )
        observable_outcome = observable_info['observable_outcome']

        # Step 1: 抽取种子实体（带有观察对象上下文）
        seeds = self.extract_seeds(question, observable_outcome=observable_outcome)
        if not seeds:
            self._log("⚠️ No seeds found!")
            return {
                'seeds': set(),
                'entities': set(),
                'edges': [],
                'chains': [],
                'observable_outcome': observable_info
            }

        all_entities = seeds.copy()
        all_edges = []
        all_relations = []  # 存储所有关系，用于去重

        # Step 2: 逐层扩展（带有观察对象上下文）
        current_layer = list(seeds)
        for depth in range(self.max_depth):
            next_layer = set()

            for entity in current_layer:
                # 基于实体集合扩散因果关系（带有观察对象上下文）
                new_relations = self.expand_causal_relations(
                    entity,
                    all_entities,
                    all_relations,
                    observable_outcome=observable_outcome
                )
                
                # 处理新关系
                for rel in new_relations:
                    all_relations.append(rel)
                    
                    # 构建边
                    edge = {
                        'head': rel['head'],
                        'tail': rel['tail'],
                        'relation': rel['relation'],
                        'description': rel['description'],
                        'confidence': rel.get('confidence'),
                        'depth': depth + 1
                    }
                    all_edges.append(edge)
                    
                    # 收集新实体
                    for e in [rel['head'], rel['tail']]:
                        if e not in all_entities:
                            all_entities.add(e)
                            next_layer.add(e)

 
            # 早停机制
            if not next_layer:

                break
            
            # 限制下一层的实体数量
            current_layer = list(next_layer)[:self.max_neighbors * 2]

        # Step 3: 构建因果链
        # Backfill missing direct relations among known entities
        backfilled = self.infer_missing_relations(all_entities, all_relations)
        if backfilled:
            for rel in backfilled:
                all_relations.append(rel)
                edge = {
                    'head': rel['head'],
                    'tail': rel['tail'],
                    'relation': rel['relation'],
                    'description': rel.get('description', ''),
                    'confidence': rel.get('confidence'),
                    'depth': self.max_depth + 1
                }
                all_edges.append(edge)
                
        chains = self._extract_chains(all_edges, seeds)

        # 输出统计
        self._print_statistics(seeds, all_entities, all_edges, chains)

        return {
            'seeds': seeds,
            'entities': all_entities,
            'edges': all_edges,
            'chains': chains,
            'observable_outcome': observable_info
        }
    
    # ==================== 三元组处理方法 ====================
    
    def get_all_triples(self, result: Dict, format: str = "structured") -> List:
        """
        获取所有的三元组

        Args:
            result: build_causal_chain的返回结果
            format: 输出格式
                - "structured": 返回完整的字典格式（包含description）
                - "simple": 返回简单的(head, relation, tail)元组格式
                - "detailed": 返回包含所有信息的详细格式

        Returns:
            三元组列表
        """
        edges = result.get('edges', [])

        if format == "simple":
            # 简单格式：(head, relation, tail)
            triples = []
            for edge in edges:
                triple = (edge['head'], edge['relation'], edge['tail'])
                triples.append(triple)
            return triples

        elif format == "structured":
            # 结构化格式：保留所有信息但简化展示
            triples = []
            for edge in edges:
                triple = {
                    'triple': (edge['head'], edge['relation'], edge['tail']),
                    'description': edge.get('description', ''),
                    'depth': edge['depth'],
                    'confidence': edge.get('confidence')
                }
                triples.append(triple)
            return triples

        elif format == "detailed":
            # 详细格式：返回所有边信息
            return edges

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'simple', 'structured', or 'detailed'")

    def get_causal_chain_paths(
        self,
        result: Dict,
        intervention: Optional[str] = None,
        target: Optional[str] = None,
        min_confidence: float = 0.0,
        max_length: int = 5,
        max_paths: int = 10
    ) -> List[List[Dict]]:
        """
        获取完整的因果链路径（从干预到目标的edge序列）

        Args:
            result: build_causal_chain的返回结果
            intervention: 干预实体（起点），如果为None则使用所有种子
            target: 目标实体（终点），如果为None则返回所有路径
            min_confidence: 最小置信度阈值
            max_length: 最大路径长度（edge数量）
            max_paths: 返回的最大路径数量

        Returns:
            路径列表，每条路径是edge字典的列表
            例如: [
                [
                    {'head': 'A', 'relation': 'causes', 'tail': 'B', 'confidence': 0.8},
                    {'head': 'B', 'relation': 'increases', 'tail': 'C', 'confidence': 0.7}
                ],
                ...
            ]
        """
        edges = result.get('edges', [])
        seeds = result.get('seeds', set())

        # 过滤低置信度的边
        filtered_edges = [e for e in edges if e.get('confidence', 1.0) >= min_confidence]

        # 构建图（head -> list of (tail, edge)）
        graph = defaultdict(list)
        for edge in filtered_edges:
            graph[edge['head'].lower()].append((edge['tail'].lower(), edge))

        # 确定起点
        if intervention:
            start_nodes = [intervention.lower()]
        else:
            start_nodes = [s.lower() for s in seeds]

        # 确定终点匹配函数
        def matches_target(node):
            if not target:
                return len(graph.get(node, [])) == 0  # 叶子节点
            target_lower = target.lower()
            return target_lower in node or node in target_lower

        # BFS搜索所有路径
        all_paths = []

        for start in start_nodes:
            queue = deque([(start, [])])
            visited_paths = set()

            while queue and len(all_paths) < max_paths:
                current, path = queue.popleft()

                # 避免重复路径
                path_key = tuple([e['head'] + '->' + e['tail'] for e in path])
                if path_key in visited_paths:
                    continue
                visited_paths.add(path_key)

                # 如果路径达到目标或达到最大长度，保存它
                if matches_target(current) and len(path) > 0:
                    all_paths.append(path)
                    if len(all_paths) >= max_paths:
                        break
                    continue

                # 如果路径过长，跳过
                if len(path) >= max_length:
                    continue

                # 扩展路径
                for next_node, edge in graph.get(current, []):
                    # 避免循环
                    if any(e['tail'].lower() == next_node for e in path):
                        continue
                    new_path = path + [edge]
                    queue.append((next_node, new_path))

        # 按路径长度和平均置信度排序
        def path_score(path):
            avg_conf = sum(e.get('confidence', 0.5) for e in path) / len(path) if path else 0
            return (-len(path), -avg_conf)  # 优先较长路径，然后是高置信度

        all_paths.sort(key=path_score)

        return all_paths[:max_paths]

    def format_causal_chain_path(self, path: List[Dict], include_confidence: bool = True) -> str:
        """
        将因果链路径格式化为可读字符串

        Args:
            path: get_causal_chain_paths返回的单条路径
            include_confidence: 是否包含置信度

        Returns:
            格式化的字符串
            例如: "A --[causes]--> B (0.8) → B --[increases]--> C (0.7)"
        """
        if not path:
            return "(empty path)"

        parts = []
        for edge in path:
            head = edge['head']
            rel = edge['relation']
            tail = edge['tail']
            conf = edge.get('confidence', 0)

            if include_confidence and conf > 0:
                parts.append(f"{head} --[{rel}]--> {tail} ({conf:.2f})")
            else:
                parts.append(f"{head} --[{rel}]--> {tail}")

        return " → ".join(parts)

    def get_formatted_causal_chains(
        self,
        result: Dict,
        intervention: Optional[str] = None,
        target: Optional[str] = None,
        min_confidence: float = 0.3,
        max_paths: int = 10
    ) -> str:
        """
        获取格式化的因果链文本（用于LLM prompt）

        Args:
            result: build_causal_chain的返回结果
            intervention: 干预实体
            target: 目标实体
            min_confidence: 最小置信度
            max_paths: 最大路径数

        Returns:
            格式化的因果链文本
        """
        paths = self.get_causal_chain_paths(
            result,
            intervention=intervention,
            target=target,
            min_confidence=min_confidence,
            max_paths=max_paths
        )

        if not paths:
            return "  (No causal chains found)"

        lines = []
        for i, path in enumerate(paths, 1):
            chain_str = self.format_causal_chain_path(path, include_confidence=True)
            lines.append(f"Chain {i}: {chain_str}")

        return "\n  ".join(lines)


    
    def triple_to_sentence(
        self, 
        triple: Union[Tuple, Dict],
        include_description: bool = False,
        use_llm: bool = True
    ) -> str:
        """
        将单个三元组转换为自然语言句子
        
        Args:
            triple: 三元组，可以是：
                - 元组格式: (head, relation, tail)
                - 字典格式: {'head': ..., 'relation': ..., 'tail': ..., 'description': ...}
                - 结构化格式: {'triple': (head, relation, tail), 'description': ...}
            include_description: 是否包含关系描述（如果有）
            use_llm: 是否使用LLM生成句子（True）或使用模板（False）
            
        Returns:
            自然语言句子
            
        Examples:
            >>> # 使用简单三元组
            >>> triple = ('Climate change', 'causes', 'extreme weather')
            >>> sentence = builder.triple_to_sentence(triple)
            >>> print(sentence)
            "Climate change causes extreme weather."
            
            >>> # 使用get_all_triples的结果
            >>> triples = builder.get_all_triples(result, format="simple")
            >>> sentence = builder.triple_to_sentence(triples[0])
            
            >>> # 使用带描述的格式
            >>> triple_with_desc = {
            ...     'triple': ('inflation', 'causes', 'reduced spending'),
            ...     'description': 'Higher prices reduce purchasing power'
            ... }
            >>> sentence = builder.triple_to_sentence(triple_with_desc, include_description=True)
            >>> print(sentence)
            "Inflation causes reduced spending. (Higher prices reduce purchasing power)"
        """
        # 解析不同格式的输入
        if isinstance(triple, tuple):
            # 元组格式: (head, relation, tail)
            if len(triple) != 3:
                raise ValueError(f"Triple tuple must have exactly 3 elements, got {len(triple)}")
            head, relation, tail = triple
            description = ""
            
        elif isinstance(triple, dict):
            # 检查是否是结构化格式
            if 'triple' in triple:
                # 格式: {'triple': (head, relation, tail), 'description': ...}
                triple_tuple = triple['triple']
                if isinstance(triple_tuple, tuple) and len(triple_tuple) == 3:
                    head, relation, tail = triple_tuple
                else:
                    raise ValueError("Invalid triple format in structured dict")
                description = triple.get('description', '')
                
            elif all(key in triple for key in ['head', 'relation', 'tail']):
                # 格式: {'head': ..., 'relation': ..., 'tail': ...}
                head = triple['head']
                relation = triple['relation']
                tail = triple['tail']
                description = triple.get('description', '')
                
            else:
                raise ValueError("Dictionary must contain either 'triple' key or 'head', 'relation', 'tail' keys")
                
        else:
            raise TypeError(f"Triple must be tuple or dict, got {type(triple)}")
        
        # 使用LLM生成句子
        if use_llm:
            try:
                # 构建提示词
                if include_description and description:
                    prompt = f"""Rewrite the causal relationship as a concise yet informative sentence.
                        Triple: ({head}, {relation}, {tail})
                        Relationship description (you may paraphrase but not invent facts): {description}

                        Requirements:
                        1) Output exactly ONE sentence, 12–24 words.
                        2) Avoid bland templates like "{head} {relation} {tail}."
                        3) Use active, natural phrasing.
                        4) Incorporate at least ONE concrete detail from the description (mechanism, condition, magnitude, population, or timeframe).
                        - If a mechanism appears, express it with "by/through ...".
                        - If a condition appears, start with "When/If ...".
                        5) Do NOT introduce any information not present in the triple or description.

                        Output: a single sentence only.
                        """
                else:
                    prompt = f"""Rewrite the causal relationship as a concise yet informative sentence.
                        Triple: ({head}, {relation}, {tail})
                        Requirements:
                        1) Output exactly ONE sentence, 12–24 words.
                        2) Avoid bland templates like "{head} {relation} {tail}."
                        3) Use active, natural phrasing.
                        4) Incorporate at least ONE concrete detail from the description (mechanism, condition, magnitude, population, or timeframe).
                        - If a mechanism appears, express it with "by/through ...".
                        - If a condition appears, start with "When/If ...".
                        5) Do NOT introduce any information not present in the triple or description.

                        Output: a single sentence only.
                        """ 
                
                # 调用LLM
                response = self._call_llm(prompt)
                
                # 清理响应
                sentence = response.strip()
                # 移除可能的引号
                sentence = sentence.strip('"\'')
                # 确保句子以句号结尾
                if sentence and not sentence[-1] in '.!?':
                    sentence += '.'
                # 确保首字母大写
                if sentence:
                    sentence = sentence[0].upper() + sentence[1:]
                
                return sentence
                
            except Exception as e:
                use_llm = False  # 降级到模板方法
        
        # 如果不使用LLM或LLM失败，使用模板方法
        if not use_llm:
            # 关系动词的转换映射
            relation_templates = {
                'causes': '{head} causes {tail}',
                'leads_to': '{head} leads to {tail}',
                'triggers': '{head} triggers {tail}',
                'influences': '{head} influences {tail}',
                'prevents': '{head} prevents {tail}',
                'enables': '{head} enables {tail}',
                'results_in': '{head} results in {tail}',
                'produces': '{head} produces {tail}',
                'generates': '{head} generates {tail}',
                'increases': '{head} increases {tail}',
                'decreases': '{head} decreases {tail}',
                'affects': '{head} affects {tail}',
                'contributes_to': '{head} contributes to {tail}',
                'depends_on': '{head} depends on {tail}',
                'requires': '{head} requires {tail}',
                'inhibits': '{head} inhibits {tail}',
                'accelerates': '{head} accelerates {tail}',
                'reduces': '{head} reduces {tail}',
                'improves': '{head} improves {tail}',
                'worsens': '{head} worsens {tail}',
                'regulates': '{head} regulates {tail}',
                'amplifies': '{head} amplifies {tail}',
                'mitigates': '{head} mitigates {tail}',
                'facilitates': '{head} facilitates {tail}',
                'hinders': '{head} hinders {tail}',
                'supports': '{head} supports {tail}',
                'undermines': '{head} undermines {tail}',
                'strengthens': '{head} strengthens {tail}',
                'weakens': '{head} weakens {tail}',
                'promotes': '{head} promotes {tail}',
                'suppresses': '{head} suppresses {tail}',
                'induces': '{head} induces {tail}',
                'blocks': '{head} blocks {tail}',
                'enhances': '{head} enhances {tail}',
                'limits': '{head} limits {tail}',
                'drives': '{head} drives {tail}',
                'fuels': '{head} fuels {tail}',
                'exacerbates': '{head} exacerbates {tail}',
                'alleviates': '{head} alleviates {tail}'
            }
            
            # 生成基本句子
            if relation in relation_templates:
                sentence = relation_templates[relation].format(head=head, tail=tail)
            else:
                # 处理未知的关系类型
                # 尝试将下划线转换为空格
                relation_words = relation.replace('_', ' ')
                
                # 检查是否需要添加冠词或介词
                if relation_words.endswith('by'):
                    sentence = f"{head} is {relation_words} {tail}"
                elif relation_words.startswith('is '):
                    sentence = f"{head} {relation_words} {tail}"
                else:
                    sentence = f"{head} {relation_words} {tail}"
            
            # 确保句子以句号结尾
            if not sentence.endswith('.'):
                sentence += '.'
            
            # 首字母大写
            sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
            
            # 如果需要包含描述
            if include_description and description:
                sentence = f"{sentence[:-1]}. ({description})"
                if not sentence.endswith('.'):
                    sentence += '.'
            
            return sentence
    
    def triples_to_sentences(
        self, 
        result: Dict, 
        include_description: bool = False,
        max_sentences: Optional[int] = None,
        use_llm: bool = True
    ) -> List[str]:
        """
        将三元组转换为自然语言句子
        
        Args:
            result: build_causal_chain的返回结果
            include_description: 是否包含关系描述
            max_sentences: 最多返回的句子数量
            use_llm: 是否使用LLM生成句子（True）或使用模板（False）
            
        Returns:
            自然语言句子列表
            
        Example:
            (Climate change, triggers, extreme weather) -> "Climate change triggers extreme weather."
        """
        edges = result.get('edges', [])
        sentences = []
        
        for i, edge in enumerate(edges):
            if max_sentences and i >= max_sentences:
                break
            
            # 使用triple_to_sentence方法处理单个三元组
            if include_description:
                # 传递完整的edge字典，包含description
                sentence = self.triple_to_sentence(edge, include_description=True, use_llm=use_llm)
            else:
                # 只传递基本的三元组
                triple = (edge['head'], edge['relation'], edge['tail'])
                sentence = self.triple_to_sentence(triple, include_description=False, use_llm=use_llm)
            
            sentences.append(sentence)
        
        return sentences
    
    def display_all_triples(
        self, 
        result: Dict, 
        display_format: str = "table",
        show_description: bool = True
    ) -> str:
        """
        以易读的格式显示所有三元组
        
        Args:
            result: build_causal_chain的返回结果
            display_format: 显示格式 ("table", "list", "sentences")
            show_description: 是否显示描述
            
        Returns:
            格式化的字符串
        """
        edges = result.get('edges', [])
        
        if not edges:
            return "No triples found in the graph."
        
        lines = []
        
        if display_format == "table":
            # 表格格式
            lines.append("\n" + "="*80)
            lines.append("ALL TRIPLES IN THE CAUSAL GRAPH")
            lines.append("="*80)
            lines.append(f"{'No.':<5} {'HEAD':<20} {'RELATION':<15} {'TAIL':<20} {'DEPTH':<7}")
            lines.append("-"*80)
            
            for i, edge in enumerate(edges, 1):
                head = edge['head'][:18] + '..' if len(edge['head']) > 20 else edge['head']
                tail = edge['tail'][:18] + '..' if len(edge['tail']) > 20 else edge['tail']
                relation = edge['relation'][:13] + '..' if len(edge['relation']) > 15 else edge['relation']
                
                lines.append(
                    f"{i:<5} {head:<20} {relation:<15} {tail:<20} {edge['depth']:<7}"
                )
                
                if show_description and edge.get('description'):
                    desc = edge['description']
                    # 换行显示描述，带缩进
                    wrapped_desc = self._wrap_text(desc, width=70, indent=7)
                    lines.append(f"      📝 {wrapped_desc}")
                    lines.append("")
            
            lines.append("="*80)
            lines.append(f"Total: {len(edges)} triples")
            
        elif display_format == "list":
            # 列表格式
            lines.append("\n" + "="*60)
            lines.append("ALL TRIPLES (LIST FORMAT)")
            lines.append("="*60)
            
            for i, edge in enumerate(edges, 1):
                lines.append(f"\n{i}. ({edge['head']}, {edge['relation']}, {edge['tail']})")
                if show_description and edge.get('description'):
                    lines.append(f"   Description: {edge['description']}")
                lines.append(f"   Depth: {edge['depth']}")
            
            lines.append("\n" + "="*60)
            lines.append(f"Total: {len(edges)} triples")
            
        elif display_format == "sentences":
            # 自然语言句子格式
            lines.append("\n" + "="*60)
            lines.append("ALL TRIPLES AS SENTENCES")
            lines.append("="*60)
            
            sentences = self.triples_to_sentences(result, include_description=show_description)
            for i, sentence in enumerate(sentences, 1):
                lines.append(f"{i}. {sentence}")
            
            lines.append("\n" + "="*60)
            lines.append(f"Total: {len(sentences)} relationships")
        
        return '\n'.join(lines)
    
    def _wrap_text(self, text: str, width: int = 70, indent: int = 0) -> str:
        """
        文本换行辅助函数
        
        Args:
            text: 要换行的文本
            width: 每行最大宽度
            indent: 续行缩进
            
        Returns:
            换行后的文本
        """
        if len(text) <= width:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length <= width:
                current_line.append(word)
                current_length += word_length
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # 添加缩进
        if indent > 0 and len(lines) > 1:
            indented_lines = [lines[0]]
            for line in lines[1:]:
                indented_lines.append(' ' * indent + line)
            return '\n'.join(indented_lines)
        
        return '\n'.join(lines)
    
    def export_triples_to_text(
        self, 
        result: Dict, 
        filename: str,
        format: str = "both"
    ) -> None:
        """
        将三元组导出为文本文件
        
        Args:
            result: build_causal_chain的返回结果
            filename: 输出文件名
            format: 导出格式 ("triples", "sentences", "both")
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("CAUSAL GRAPH TRIPLES EXPORT\n")
                f.write("="*60 + "\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                if format in ["triples", "both"]:
                    f.write("TRIPLES FORMAT\n")
                    f.write("-"*40 + "\n")
                    triples = self.get_all_triples(result, format="simple")
                    for i, triple in enumerate(triples, 1):
                        f.write(f"{i}. {triple}\n")
                    f.write(f"\nTotal: {len(triples)} triples\n\n")
                
                if format in ["sentences", "both"]:
                    f.write("NATURAL LANGUAGE FORMAT\n")
                    f.write("-"*40 + "\n")
                    sentences = self.triples_to_sentences(result, include_description=True)
                    for i, sentence in enumerate(sentences, 1):
                        f.write(f"{i}. {sentence}\n")
                    f.write(f"\nTotal: {len(sentences)} relationships\n\n")
                
                if format == "both":
                    f.write("="*60 + "\n")
                    f.write("DETAILED TRIPLES WITH DESCRIPTIONS\n")
                    f.write("="*60 + "\n")
                    structured_triples = self.get_all_triples(result, format="structured")
                    for i, item in enumerate(structured_triples, 1):
                        triple = item['triple']
                        desc = item['description']
                        depth = item['depth']
                        f.write(f"\n{i}. Triple: {triple}\n")
                        f.write(f"   Depth: {depth}\n")
                        if desc:
                            f.write(f"   Description: {desc}\n")
                
                self._log(f"✅ Triples exported to: {filename}")
                
        except Exception as e:
            self._log(f"⚠️ Failed to export triples: {e}")
    
    # ==================== 输出方法 ====================
    
    def _print_statistics(
        self, 
        seeds: Set[str], 
        entities: Set[str], 
        edges: List[Dict], 
        chains: List[List[str]]
    ) -> None:
        """打印统计信息"""
        self._log(f"\n" + "="*60)
        self._log("FINAL STATISTICS")
        self._log("="*60)
        self._log(
            f"Total entities: {len(entities)} "
            f"(seeds={len(seeds)}, "
            f"expanded={len(entities)-len(seeds)})"
        )
        self._log(f"Total edges: {len(edges)}")
        self._log(f"Causal chains: {len(chains)}")
        
        # 统计关系类型
        relation_types = defaultdict(int)
        for edge in edges:
            relation_types[edge['relation']] += 1
        
        self._log("\nRelation types:")
        for rel_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True):
            self._log(f"  - {rel_type}: {count}")
    
    def visualize(self, result: Dict) -> str:
        """
        可视化结果
        
        Args:
            result: build_causal_chain的返回结果
            
        Returns:
            格式化的可视化字符串
        """
        lines = ["\n" + "="*60, "CAUSAL GRAPH VISUALIZATION", "="*60]
        
        # 显示种子实体
        seeds_str = ', '.join(result.get('seeds', []))
        lines.append(f"\n📍 Seeds ({len(result['seeds'])}): {seeds_str}")
        
        # 显示边（带关系和描述）
        edges = result.get('edges', [])
        if edges:
            lines.append(f"\n🔗 Edges ({len(edges)}):")
            for i, edge in enumerate(edges[:10], 1):  # 只显示前10条
                lines.append(
                    f"  {i}. {edge['head']} --[{edge['relation']}]--> {edge['tail']} "
                    f"(depth={edge['depth']})"
                )
                if edge.get('description'):
                    lines.append(f"      💭 {edge['description']}")
        else:
            lines.append("\n⚠️ No edges found")
        
        # 显示因果链
        chains = result.get('chains', [])
        if chains:
            lines.append(f"\n⛓️ Causal Chains ({len(chains)}):")
            for i, chain in enumerate(chains[:5], 1):
                chain_str = ' → '.join(chain)
                lines.append(f"  {i}. {chain_str}")
        else:
            lines.append("\n⚠️ No causal chains found")
        
        lines.append("="*60)
        return '\n'.join(lines)
    
    def save_log(self, filename: str = "ego_expansion_log.txt") -> None:
        """
        保存日志到文件
        
        Args:
            filename: 日志文件名
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.log))
            self._log(f"\n💾 Log saved to: {filename}")
        except Exception as e:
            self._log(f"⚠️ Failed to save log: {e}")
    
    def export_graph(self, result: Dict, format: str = "dict") -> Dict:
        """
        导出图结构为不同格式
        
        Args:
            result: build_causal_chain的返回结果
            format: 导出格式（dict, networkx等）
            
        Returns:
            图结构数据
        """
        if format == "dict":
            return {
                'nodes': list(result['entities']),
                'edges': [
                    {
                        'source': edge['head'],
                        'target': edge['tail'],
                        'relation': edge['relation'],
                        'description': edge.get('description', ''),
                        'depth': edge['depth']
                    }
                    for edge in result['edges']
                ],
                'chains': result['chains']
            }
        elif format == "json":
            # 返回更详细的JSON格式
            nodes = []
            for entity in result['entities']:
                node_type = 'seed' if entity in result['seeds'] else 'expanded'
                nodes.append({
                    'id': entity,
                    'label': entity,
                    'type': node_type
                })
            
            edges = []
            for edge in result['edges']:
                edges.append({
                    'id': f"{edge['head']}-{edge['relation']}-{edge['tail']}",
                    'source': edge['head'],
                    'target': edge['tail'],
                    'relation': edge['relation'],
                    'label': edge['relation'],
                    'description': edge.get('description', ''),
                    'depth': edge['depth']
                })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'chains': result['chains']
            }
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_to_file(self, result: Dict, filename: str, format: str = "json") -> None:
        """
        将结果导出到文件
        
        Args:
            result: build_causal_chain的返回结果
            filename: 输出文件名
            format: 导出格式
        """
        try:
            graph_data = self.export_graph(result, format)
            
            if format == "json":
                import json
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, ensure_ascii=False, indent=2)
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(str(graph_data))
            
            self._log(f"\n📁 Graph exported to: {filename}")
        except Exception as e:
            self._log(f"⚠️ Failed to export graph: {e}")


# ==================== 使用示例 ====================

def main():
    """主函数：演示如何使用改进的因果图构建器"""
    
    # 测试用例
    test_cases = [
        {
            'name': 'Air Pollution',
            'question': """Pollute the air even more will not solve the problem but make it worse. 
            We need to find a way to release the heat trapped in the atmosphere."""
        },
        {
            'name': 'Climate Change',
            'question': """Climate change leads to increased extreme weather events. These extreme 
            weather events damage crop growth, thus affecting food production. Reduced food 
            production leads to price increases, ultimately impacting people's quality of life."""
        },
        {
            'name': 'Economic Impact',
            'question': """Rising inflation causes reduced consumer spending. This leads to 
            decreased business revenue, which triggers layoffs. Unemployment then further 
            reduces spending power, creating a vicious cycle."""
        },
        {
            'name': 'Technology Adoption',
            'question': """AI automation increases productivity but may displace workers. 
            This creates demand for reskilling programs while potentially widening income gaps. 
            Policy interventions become necessary to manage the transition."""
        }
    ]
    
    # 选择测试用例
    test_index = 1  # 可以改变这个来测试不同案例
    test = test_cases[test_index]
    
    print(f"\n{'='*60}")
    print(f"Testing: {test['name']}")
    print(f"Question: {test['question'][:100]}...")
    print(f"{'='*60}")
    
    # 创建构建器
    builder = EgoExpansionCausalBuilder(
    )
    
    # 构建因果图
    result = builder.build_causal_chain(test['question'])
    
    # 可视化结果
    print(builder.visualize(result))
    
    # 导出图结构
    graph_data = builder.export_graph(result, format="json")
    print(f"\n📊 Exported graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
    
    # 保存结果到文件
    output_name = test['name'].lower().replace(' ', '_')
    builder.save_log(f"ego_expansion_{output_name}_log.txt")
    builder.export_to_file(result, f"ego_expansion_{output_name}_graph.json", format="json")
    
    # 显示一些有描述的关系示例
    print("\n" + "="*60)
    print("SAMPLE RELATIONS WITH DESCRIPTIONS")
    print("="*60)
    for edge in result['edges'][:5]:
        print(f"\n📌 {edge['head']} --[{edge['relation']}]--> {edge['tail']}")
        if edge.get('description'):
            print(f"   Description: {edge['description']}")
    
    # ========== 演示新功能 ==========
    
    # 1. 获取所有三元组
    print("\n" + "="*60)
    print("ALL TRIPLES (Simple Format)")
    print("="*60)
    simple_triples = builder.get_all_triples(result, format="simple")
    for i, triple in enumerate(simple_triples[:5], 1):
        print(f"{i}. {triple}")
    if len(simple_triples) > 5:
        print(f"... and {len(simple_triples) - 5} more triples")
    
    # 2. 将三元组转换为自然语言句子
    print("\n" + "="*60)
    print("TRIPLES AS NATURAL LANGUAGE SENTENCES")
    print("="*60)
    sentences = builder.triples_to_sentences(result, include_description=False)
    for i, sentence in enumerate(sentences[:5], 1):
        print(f"{i}. {sentence}")
    if len(sentences) > 5:
        print(f"... and {len(sentences) - 5} more sentences")
    
    # 3. 显示所有三元组（表格格式）
    print(builder.display_all_triples(result, display_format="table", show_description=False))
    
    # 4. 导出三元组到文本文件
    triples_file = f"ego_expansion_{output_name}_triples.txt"
    builder.export_triples_to_text(result, triples_file, format="both")
    print(f"\n📄 Triples exported to: {triples_file}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
