"""
简化版因果图构建器 - 去掉置信度验证，只保留实体搜集
包含完整的LLM输入输出日志
"""

import ollama
import re
import json
from typing import List, Dict, Set
from datetime import datetime
import networkx as nx

class SimplifiedCausalGraphBuilder:
    """简化版因果图构建器 - 专注于实体搜集"""
    
    def __init__(self, model_name="gemma2:27b", log_file="causal_build_log.txt"):
        self.model_name = model_name
        self.log_file = log_file
        self.log_entries = []
        
        # 初始化日志
        self._log_header()
    
    def _log_header(self):
        """写入日志头"""
        header = f"""
{'='*80}
因果图构建日志
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型: {self.model_name}
{'='*80}
"""
        self.log_entries.append(header)
        print(header)
    
    def _log_llm_call(self, step: str, prompt: str, response: str):
        """记录LLM调用"""
        log = f"""
{'─'*80}
步骤: {step}
{'─'*80}

【LLM输入 Prompt】
{prompt}

【LLM输出 Response】
{response}

{'─'*80}
"""
        self.log_entries.append(log)
        print(log)
    
    def save_log(self):
        """保存日志到文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_entries))
        print(f"\n✅ 日志已保存到: {self.log_file}")
    
    def extract_entities(self, question: str) -> Set[str]:
        """步骤1：抽取实体"""
        prompt = f"""Extract all important entities (nouns, noun phrases, concepts) from the following question.

Question: {question}

Return ONLY entity names, one per line, no numbers or bullets.

Entities:"""
        
        response = ollama.generate(model=self.model_name, prompt=prompt)
        response_text = response['response'].strip()
        
        self._log_llm_call("Step 1: Extract Entities (E_Q)", prompt, response_text)
        
        # 解析实体
        entities = set()
        for line in response_text.split('\n'):
            entity = line.strip()
            
            # 清理
            if not entity or len(entity) < 2:
                continue
            if entity.lower().startswith(('here', 'entity', 'entities')):
                continue
            
            # 移除编号、符号
            entity = re.sub(r'^\d+[\.)]\s*', '', entity)
            entity = entity.lstrip('-•*>').strip().strip('"\'*').strip()
            
            if entity and len(entity) > 1:
                entities.add(entity)
        
        print(f"\n✅ 抽取到 {len(entities)} 个实体: {entities}\n")
        return entities
    
    def extract_class_a_triples(self, question: str, entities: Set[str]) -> List[Dict]:
        """步骤2：A类三元组（完全来自问题）"""
        prompt = f"""Analyze causal relationships in the question. Only use these entities.

Question: {question}

Entities: {', '.join(entities)}

Format: head -> relation -> tail | evidence

Return triples only:"""
        
        response = ollama.generate(model=self.model_name, prompt=prompt)
        response_text = response['response'].strip()
        
        self._log_llm_call("Step 2: Extract Class A Triples", prompt, response_text)
        
        # 解析A类三元组
        a_triples = []
        entity_lower_map = {e.lower(): e for e in entities}
        
        for line in response_text.split('\n'):
            line = line.strip()
            if '->' not in line:
                continue
            
            if '|' in line:
                triple_part, evidence = line.split('|', 1)
            else:
                triple_part = line
                evidence = "From question text"
            
            parts = [p.strip() for p in triple_part.split('->')]
            if len(parts) >= 3:
                head = parts[0].strip('"\'*').strip()
                relation = parts[1].strip('"\'*').strip()
                tail = parts[2].strip('"\'*').strip()
                
                # 简单匹配
                head_match = self._find_entity(head, entities, entity_lower_map)
                tail_match = self._find_entity(tail, entities, entity_lower_map)
                
                if head_match and tail_match:
                    a_triples.append({
                        'head': head_match,
                        'relation': relation,
                        'tail': tail_match,
                        'evidence': evidence.strip(),
                        'class': 'A'
                    })
        
        print(f"✅ 找到 {len(a_triples)} 个A类三元组")
        for t in a_triples:
            print(f"   {t['head']} --[{t['relation']}]--> {t['tail']}")
        print()
        
        return a_triples
    
    def _find_entity(self, text: str, entities: Set[str], entity_lower_map: Dict[str, str]) -> str:
        """查找匹配的实体"""
        text = text.strip('"\'*').strip()
        
        # 精确匹配
        if text in entities:
            return text
        
        # 小写匹配
        if text.lower() in entity_lower_map:
            return entity_lower_map[text.lower()]
        
        # 部分匹配
        for entity in entities:
            if text.lower() in entity.lower() or entity.lower() in text.lower():
                return entity
        
        return None
    
    def expand_class_b_triples(self, question: str, eq_entities: Set[str], k: int = 5) -> List[Dict]:
        """步骤3：B类三元组（问题实体 + 外部实体）"""
        b_triples = []
        eq_entities_lower = {e.lower() for e in eq_entities}
        
        # 对每个E_Q实体扩展
        for entity in list(eq_entities)[:3]:
            prompt = f"""Propose {k} NEW external entities related to "{entity}" (NOT in this list: {', '.join(eq_entities)}).

Question: {question}

Format: new_entity -> direction(cause/effect) -> {entity}

Return {k} triples:"""
            
            response = ollama.generate(model=self.model_name, prompt=prompt)
            response_text = response['response'].strip()
            
            self._log_llm_call(f"Step 3: Expand B-class for '{entity}'", prompt, response_text)
            
            # 解析
            for line in response_text.split('\n')[:k]:
                line = line.strip()
                if '->' not in line:
                    continue
                
                parts = [p.strip() for p in line.split('->')]
                if len(parts) >= 3:
                    entity1 = parts[0].strip('"\'*').strip()
                    direction = parts[1].lower()
                    entity2 = parts[2].strip('"\'*').strip()
                    
                    # 判断哪个是新实体
                    entity1_in_eq = entity1.lower() in eq_entities_lower
                    entity2_in_eq = entity2.lower() in eq_entities_lower
                    
                    # 确定head和tail
                    if entity1_in_eq and not entity2_in_eq:
                        # entity1在E_Q，entity2是新的 (EFFECT)
                        head, tail = entity1, entity2
                        external = entity2
                    elif not entity1_in_eq and entity2_in_eq:
                        # entity1是新的，entity2在E_Q (CAUSE)
                        head, tail = entity1, entity2
                        external = entity1
                    elif not entity1_in_eq and not entity2_in_eq:
                        # 尝试匹配当前entity
                        if entity.lower() in entity2.lower():
                            head, tail = entity1, entity
                            external = entity1
                        elif entity.lower() in entity1.lower():
                            head, tail = entity, entity2
                            external = entity2
                        else:
                            continue
                    else:
                        # 两个都在E_Q中
                        continue
                    
                    # 验证B类约束
                    head_in_eq = head.lower() in eq_entities_lower
                    tail_in_eq = tail.lower() in eq_entities_lower
                    
                    if (head_in_eq and not tail_in_eq) or (not head_in_eq and tail_in_eq):
                        b_triples.append({
                            'head': head,
                            'relation': 'causes',
                            'tail': tail,
                            'external_entity': external,
                            'class': 'B'
                        })
        
        print(f"✅ 找到 {len(b_triples)} 个B类三元组")
        for t in b_triples:
            print(f"   {t['head']} --> {t['tail']} (新实体: {t['external_entity']})")
        print()
        
        return b_triples
    
    def expand_class_c_triples(self, question: str, eq_entities: Set[str], 
                               b_entities: Set[str], k: int = 3) -> List[Dict]:
        """步骤4：C类三元组（桥接实体）"""
        c_triples = []
        
        if not b_entities:
            print("⚠️ 没有B类实体，跳过C类扩展\n")
            return c_triples
        
        # 清理B类实体
        b_entities_clean = {e.strip('*').strip() for e in b_entities}
        eq_entities_lower = {e.lower() for e in eq_entities}
        all_known = eq_entities.union(b_entities_clean)
        all_known_lower = {e.lower() for e in all_known}
        
        # 对每个B类实体扩展
        for b_entity in list(b_entities_clean)[:4]:
            # 确保是真正的B类实体
            if b_entity.lower() in eq_entities_lower:
                continue
            
            prompt = f"""Propose {k} NEW bridging concepts for "{b_entity}" (NOT in: {', '.join(all_known)}).

Question: {question}

Format: new_bridge -> relation -> {b_entity}

Return {k} triples:"""
            
            response = ollama.generate(model=self.model_name, prompt=prompt)
            response_text = response['response'].strip()
            
            self._log_llm_call(f"Step 4: Expand C-class for '{b_entity}'", prompt, response_text)
            
            # 解析
            for line in response_text.split('\n')[:k]:
                line = line.strip()
                if '->' not in line:
                    continue
                
                parts = [p.strip() for p in line.split('->')]
                if len(parts) >= 3:
                    bridge = parts[0].strip('"\'*').strip()
                    relation = parts[1].strip('"\'*').strip()
                    target = parts[2].strip('"\'*').strip()
                    
                    # 检查bridge是否是新实体
                    if bridge.lower() in all_known_lower:
                        continue
                    
                    # 检查target是否是B类实体
                    target_in_b = target.lower() in {e.lower() for e in b_entities_clean}
                    target_in_eq = target.lower() in eq_entities_lower
                    
                    if not target_in_b or target_in_eq:
                        target = b_entity
                    
                    # 验证C类约束
                    head_is_new = bridge.lower() not in all_known_lower
                    tail_is_b = (target.lower() in {e.lower() for e in b_entities_clean} and
                                target.lower() not in eq_entities_lower)
                    
                    if head_is_new and tail_is_b:
                        c_triples.append({
                            'head': bridge,
                            'relation': relation,
                            'tail': target,
                            'class': 'C'
                        })
                        
                        # 更新已知实体
                        all_known.add(bridge)
                        all_known_lower.add(bridge.lower())
        
        print(f"✅ 找到 {len(c_triples)} 个C类三元组")
        for t in c_triples:
            print(f"   {t['head']} --[{t['relation']}]--> {t['tail']}")
        print()
        
        return c_triples
    
    def build_graph(self, triples: List[Dict]) -> nx.DiGraph:
        """构建图"""
        G = nx.DiGraph()
        for triple in triples:
            G.add_edge(
                triple['head'],
                triple['tail'],
                relation=triple.get('relation', 'causes'),
                class_type=triple['class']
            )
        return G
    
    def visualize(self, G: nx.DiGraph) -> str:
        """可视化"""
        output = ["\n" + "="*60]
        output.append("因果关系图")
        output.append("="*60)
        output.append(f"节点: {G.number_of_nodes()}")
        output.append(f"边: {G.number_of_edges()}\n")
        
        for u, v, data in G.edges(data=True):
            output.append(
                f"{u} --[{data.get('relation', '?')}]--> {v} "
                f"(Class: {data.get('class_type', '?')})"
            )
        
        output.append("="*60)
        return '\n'.join(output)
    
    def process(self, question: str):
        """完整流程"""
        print("\n" + "="*80)
        print("开始处理...")
        print("="*80 + "\n")
        
        # Step 1: 抽取实体
        print("▶ Step 1: 抽取实体 (E_Q)")
        eq_entities = self.extract_entities(question)
        
        # Step 2: A类三元组
        print("▶ Step 2: A类三元组")
        a_triples = self.extract_class_a_triples(question, eq_entities)
        
        # Step 3: B类三元组
        print("▶ Step 3: B类三元组")
        b_triples = self.expand_class_b_triples(question, eq_entities, k=5)
        
        # 收集B类新实体
        eq_entities_lower = {e.lower() for e in eq_entities}
        b_entities = set()
        for t in b_triples:
            if 'external_entity' in t:
                if t['external_entity'].lower() not in eq_entities_lower:
                    b_entities.add(t['external_entity'])
        
        print(f"📊 B类新增实体: {b_entities}\n")
        
        # Step 4: C类三元组
        print("▶ Step 4: C类三元组")
        c_triples = self.expand_class_c_triples(question, eq_entities, b_entities, k=3)
        
        # 构建图
        all_triples = a_triples + b_triples + c_triples
        graph = self.build_graph(all_triples)
        
        # 可视化
        viz = self.visualize(graph)
        print(viz)
        
        # 统计
        print("\n" + "="*80)
        print("统计信息")
        print("="*80)
        print(f"E_Q实体数: {len(eq_entities)}")
        print(f"A类三元组: {len(a_triples)}")
        print(f"B类三元组: {len(b_triples)} (新增 {len(b_entities)} 个实体)")
        print(f"C类三元组: {len(c_triples)}")
        print(f"总节点数: {graph.number_of_nodes()}")
        print(f"总边数: {graph.number_of_edges()}")
        print("="*80 + "\n")
        
        # 保存日志
        self.save_log()
        
        return {
            'entities': eq_entities,
            'b_entities': b_entities,
            'triples': {
                'A': a_triples,
                'B': b_triples,
                'C': c_triples
            },
            'graph': graph
        }


# ===== 使用示例 =====
if __name__ == "__main__":
    question = """Climate change leads to increased extreme weather events. These extreme weather 
    events damage crop growth, thus affecting food production. Reduced food production leads to 
    price increases, ultimately impacting people's quality of life."""
    
    # 创建构建器
    builder = SimplifiedCausalGraphBuilder(
        model_name="gemma2:27b",
        log_file="causal_build_log.txt"
    )
    
    # 处理
    result = builder.process(question)
    
    print("\n✅ 完成！查看日志文件: causal_build_log.txt")