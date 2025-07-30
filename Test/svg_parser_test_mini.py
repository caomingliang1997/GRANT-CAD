import os, random, json, math
from data.svg_parser import parse_svg

# 1. 指定你的 SVG 存放文件夹
SVG_DIR = r"D:/GRANT-CAD/floorplancad_v2/svg_folder"
CONFIG = {
    "distance_threshold": 5.0,
    "angle_threshold": 5.0,
    "collinearity_distance_threshold": 100.0,
    "max_edges_per_node": None
}

# 2. 随机选 10 个 SVG
all_svgs = [os.path.join(SVG_DIR, f) for f in os.listdir(SVG_DIR) if f.lower().endswith(".svg")]
samples = random.sample(all_svgs, min(10, len(all_svgs)))

for svg in samples:
    print(f"\n>>> Testing {os.path.basename(svg)}")
    try:
        graph = parse_svg(svg, CONFIG)
    except Exception as e:
        import traceback
        print("✖ parse_svg 异常：", e)
        traceback.print_exc()  # 打印详细的异常堆栈信息
        continue

    # 3. 基本结构校验
    nodes = graph.get("node_feats")
    edges = graph.get("edge_index")
    edge_feats = graph.get("edge_feats")

    assert isinstance(nodes, list) and len(nodes)>0, "node_feats 应为非空 list"
    feat_len = len(nodes[0])
    for vec in nodes:
        assert isinstance(vec, list) and len(vec)==feat_len, "每个 node_feat 应长度一致"

    assert isinstance(edges, list) and len(edges)>0, "edge_index 应为非空 list"
    for e in edges:
        assert isinstance(e, (list,tuple)) and len(e)==2, f"edge_index 条目应为 [u, v]：发现 {e}"
        u,v = e
        assert isinstance(u,int) and isinstance(v,int), "u, v 应为 int"
        assert 0<=u<len(nodes) and 0<=v<len(nodes), f"u/v 越界：{u},{v}"

    assert isinstance(edge_feats, list) and len(edge_feats)==len(edges), "edge_feats 数量应=edges"
    for fe in edge_feats:
        assert isinstance(fe, list), "每个 edge_feat 应为 list"
        for f in fe:
            assert isinstance(f, float), "edge_feat 中都应是 float"
            assert not (math.isnan(f) or math.isinf(f)), "边特征不能有 NaN/Inf"

    # 4. JSON 缓存文件检查
    json_path = svg.rsplit(".",1)[0] + ".json"
    assert os.path.isfile(json_path), "未生成对应的 JSON 缓存文件"
    with open(json_path) as jf:
        data = json.load(jf)
    # 简单再读一次，防止写入损坏
    assert "node_feats" in data and "edge_index" in data

    print("✔ 通过")
