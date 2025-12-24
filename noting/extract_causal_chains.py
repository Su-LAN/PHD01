import json
from collections import defaultdict
from typing import Dict, List, Set


def load_graph(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_causal_adj_list(graph: Dict):
    nodes: Set[str] = set()
    indegree: Dict[str, int] = defaultdict(int)
    adj: Dict[str, List[str]] = defaultdict(list)

    for node in graph.get("nodes", []):
        node_id = node["id"]
        nodes.add(node_id)
        # ensure every node appears in indegree
        _ = indegree[node_id]

    for edge in graph.get("edges", []):
        if edge.get("relation") != "causal":
            continue
        src = edge["src"]
        dst = edge["dst"]
        nodes.add(src)
        nodes.add(dst)
        adj[src].append(dst)
        indegree[dst] += 1
        # ensure src appears in indegree even if only as parent
        _ = indegree[src]

    return nodes, adj, indegree


def find_sources(nodes: Set[str], indegree: Dict[str, int]) -> List[str]:
    return [n for n in nodes if indegree.get(n, 0) == 0]


def enumerate_causal_chains(adj: Dict[str, List[str]], sources: List[str]) -> List[List[str]]:
    """
    枚举图中所有简单路径（不包含重复节点），
    这些路径就是 causal chains。
    """
    all_chains: List[List[str]] = []

    def dfs(node: str, path: List[str]):
        children = adj.get(node, [])
        if not children:
            # 叶子节点，结束一条链
            all_chains.append(path.copy())
            return

        for child in children:
            if child in path:
                # 防止意外的环
                continue
            path.append(child)
            dfs(child, path)
            path.pop()

    for src in sources:
        dfs(src, [src])

    return all_chains


def extract_causal_chains(input_path: str, output_path: str):
    graph = load_graph(input_path)
    nodes, adj, indegree = build_causal_adj_list(graph)
    sources = find_sources(nodes, indegree)
    chains = enumerate_causal_chains(adj, sources)

    result = {
        "input_graph": input_path,
        "num_nodes": len(nodes),
        "num_chains": len(chains),
        "chains": chains,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="遍历 HTN DAG 图，导出所有 causal chains 到 JSON 文件。")
    parser.add_argument(
        "--input",
        "-i",
        default="htn_dag_graph.json",
        help="输入 HTN DAG JSON 文件路径（默认: htn_dag_graph.json）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="htn_causal_chains.json",
        help="输出 causal chains JSON 文件路径（默认: htn_causal_chains.json）",
    )

    args = parser.parse_args()
    extract_causal_chains(args.input, args.output)

