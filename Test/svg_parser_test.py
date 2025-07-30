# 测试svg_parser.py
import os
import json
import random
import math
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from data.svg_parser import parse_svg

CONFIG = {
    "distance_threshold": 5.0,
    "angle_threshold": 5.0,
    "collinearity_distance_threshold": 100.0,
    "max_edges_per_node": None
}

SVG_DIR = "D:/GRANT-CAD/floorplancad_v2/svg_folder"
NUM_SAMPLES = 10
RANDOM_SEED = 42

def sanity_checks(svg_path: str, graph: Dict[str, Any]):
    """对 parse_svg 的返回做完整的健壮性检查。"""
    assert isinstance(graph, dict), "graph_data 必须是 dict"
    for key in ["node_feats", "edge_index", "edge_feats"]:
        assert key in graph, f"缺少 key: {key}"

    node_feats = graph["node_feats"]
    edge_index = graph["edge_index"]
    edge_feats = graph["edge_feats"]

    # 1) 维度与类型检查
    assert isinstance(node_feats, list) and len(node_feats) > 0, "node_feats 为空"
    assert isinstance(edge_index, list), "edge_index 必须是 list"
    assert isinstance(edge_feats, list), "edge_feats 必须是 list"
    assert len(edge_index) == len(edge_feats), "edge_index 与 edge_feats 数量不匹配"

    num_nodes = len(node_feats)
    d_node = len(node_feats[0])
    assert d_node >= 3, f"节点特征维度异常: {d_node}"

    # 2) 边索引越界 / NaN / Inf 检查
    for (idx, pair) in enumerate(edge_index):
        assert isinstance(pair, list) and len(pair) == 2, "edge_index 中的每个元素必须是长度为2的list"
        u, v = pair
        assert isinstance(u, int) and isinstance(v, int), "edge_index 必须是 int 索引"
        assert 0 <= u < num_nodes and 0 <= v < num_nodes, f"edge_index 存在越界: {(u, v)}"
        # 检查 edge_feats 里是否有 NaN/Inf
        feats = edge_feats[idx]
        assert isinstance(feats, list), "edge_feats 中的每个元素必须是 list"
        for f in feats:
            assert not (isinstance(f, float) and (math.isnan(f) or math.isinf(f))), "edge_feats 存在 NaN/Inf"

    # 3) 双向边是否成对（svg_parser 的实现是对每条无向边都写入 u->v 与 v->u）
    directed = set()
    for (u, v) in edge_index:
        directed.add((u, v))
    for (u, v) in directed:
        if (v, u) not in directed:
            # 允许孤立节点没有边，但一旦有 (u, v) 应该也要有 (v, u)
            raise AssertionError(f"缺少反向边: ({u}, {v})")

    # 4) JSON 是否落地并和内存一致
    json_path = Path(svg_path).with_suffix(".json")
    assert json_path.exists(), f"未找到导出的 JSON: {json_path}"
    with open(json_path, "r") as f:
        disk_graph = json.load(f)
    for k in ["node_feats", "edge_index", "edge_feats"]:
        assert k in disk_graph, f"JSON 中缺少 {k}"
        assert len(disk_graph[k]) == len(graph[k]), f"JSON 与内存中 {k} 数量不一致"

    # 5) 简单的统计输出
    print(f"[OK] {Path(svg_path).name} | #nodes={num_nodes}, #edges={len(edge_index)} (directed)")

def main():
    random.seed(RANDOM_SEED)
    all_svgs = [str(p) for p in Path(SVG_DIR).glob("*.svg")]
    assert len(all_svgs) > 0, "该目录下没有 SVG 文件"

    sampled = random.sample(all_svgs, min(NUM_SAMPLES, len(all_svgs)))

    for svg in tqdm(sampled, desc="Testing svg_parser"):
        graph = parse_svg(svg, CONFIG)
        sanity_checks(svg, graph)

    print("\nAll tests passed for the sampled SVGs!")

if __name__ == "__main__":
    main()
