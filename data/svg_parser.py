# 负责处理 SVG 解析和基本元素提取
# 负责读取一个 .svg 文件，并将其转换为图形表示（节点用于表示线条、弧线、圆形等基本元素，边用于表示关系）
# 计算初始节点和边的特征，然后将图形保存为中间的 JSON 格式以便重复使用

import math
import json
from svgpathtools import svg2paths2, Path, Line, Arc, QuadraticBezier, CubicBezier
import random
import logging
logging.basicConfig(level=logging.DEBUG)    # 日志等级设定为 DEBUG
import networkx as nx
import matplotlib.pyplot as plt

def parse_svg(svg_file, config):

    """
    输入 SVG 文件路径+配置参数
    配置参数包括：
    distance_threshold: 两节点间距离低于该值则认为相邻。
    angle_threshold: 判断平行/垂直的角度差阈值。
    collinearity_distance_threshold: 共线邻接的最大间隙。
    max_edges_per_node: 控制单个节点的最大邻接边数量。

    输出 图结构数据 dict，包含 node_feats、edge_index、edge_feats
    """
    logging.debug(f"Parsing SVG file: {svg_file}")
    try:
        paths_attrs = svg2paths2(svg_file)
        logging.debug("SVG paths loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load SVG: {e}")
        raise

    # 来自配置文件的阈值和参数
    dist_thresh = config.get("distance_threshold", 5.0) # 比较两条图元端点间的最小欧氏距离，决定是否连边
    angle_thresh = config.get("angle_threshold", 5.0)   # 以角度为单位，平行/垂直的阈值角度
    collinearity_dist = config.get("collinearity_distance_threshold", 100.0)  # 共线性邻接的最大间距
    max_edges_per_node = config.get("max_edges_per_node", 30)  # 控制单个节点的最大邻接边数量，默认值为 30
    # tip1：如果想让截断过程更可控，而不是纯随机，也可以把截断策略换成“优先保留距离最近 or 角度最接近的边”，但这属于策略扩展范畴，可后续再优化

    # 加载 SVG 文件，并将所有类似路径的元素解析为 Path 对象
    paths_attrs = svg2paths2(svg_file)  # svg2paths2 把 SVG 文件解析成若干 Path 对象 及其 属性字典 等信息；返回值是一个元组
    paths, attr_list = paths_attrs[:2]  # paths 对象列表（每条路径内部又可拆成若干段 segment）；attr_list 与 paths 同长的属性字典列表，保存了原始 SVG 元素的几何/样式属性。
    nodes = []  # 节点属性字典列表:列表里的每个元素都是 字典，对应一条图元（线段、弧、圆、椭圆、多段线片段等）的抽象“节点”
    """
    节点的属性包括：
    type: 图元类型，如 "line", "arc", "circle", "ellipse", "curve" 等。
    length: 图元的长度或周长。
    orient_angle: 图元的主方向角（弧度值））。注：对于圆形或椭圆，orient_angle 通常为 0 ，因为它们没有明确的方向。
    center_x, center_y: 图元的中心坐标。
    
    """

    node_index_map = []  # 与 nodes 等长的列表，记录索引到原始 SVG 数据的映射,方便调试、可视化或额外特征扩充.
    """
    attr：该图元在 SVG 中的完整属性字典（便于追溯颜色、图层等信息）。
    seg_or_none：若图元来自 Path 中的某一段，则保存对应的 svgpathtools segment 对象；对于整体圆、矩形等一次就能确定的形状，则记为 None。
    """

    # 遍历解析后的元素，并为每个基本元素创建节点。
    for path_obj, attr in zip(paths, attr_list):
        # path_obj:描述 一条完整路径（可由若干 Segment 组成）
        # attr:与这条路径对应的属性字典
        # 根据属性确定元素类型
        if "cx" in attr:  # 圆形或椭圆形
            # 圆心或椭圆中心坐标
            cx = float(attr.get("cx", 0.0))
            cy = float(attr.get("cy", 0.0))
            if "rx" in attr or "ry" in attr:
                # 椭圆（或由 rx 和 ry 参数指定的圆形）
                # 椭圆的长、短半轴
                rx = float(attr.get("rx", attr.get("r", 0.0)))
                ry = float(attr.get("ry", attr.get("r", 0.0)))
                # 如果已知圆的半径“r”，那么 rx 等于 ry 。
                primitive_type = "ellipse" if rx != ry else "circle"    # 判断
                r_major, r_minor = max(rx, ry), min(rx, ry)   # 统一记录 长半轴、短半轴
            else:
                # 处理只给 r 而没有 cx/cy/rx/ry 组合
                r = float(attr.get("r", 0.0))
                primitive_type = "circle"
                r_major = r_minor = r
            # 将整个圆/椭圆视为一个节点
            # 计算长度特征：周长（椭圆时取近似值）
            if primitive_type == "circle":
                length = 2 * math.pi * r_major
            else:
                # 椭圆周长的拉马努金近似值
                a = r_major
                b = r_minor
                length = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
            # 方向说明：对于完整的圆或椭圆，没有明确的方向。为了简便起见，使用 0 。
            angle = 0.0
            # 准备节点信息
            node = {
                "type": primitive_type,
                "length": length,
                "orient_angle": angle,
                "center_x": cx,
                "center_y": cy
            }
            nodes.append(node)
            node_index_map.append((attr, None))
        elif "x1" in attr and "y1" in attr:
            # 线元素
            x1 = float(attr["x1"])
            y1 = float(attr["y1"])
            x2 = float(attr["x2"])
            y2 = float(attr["y2"])
            primitive_type = "line"
            # 保持一致的布局（从小到大依次排列以确保稳定）
            if (x2 < x1) or (x2 == x1 and y2 < y1):
                # 交换
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            # 计算长度
            dx = x2 - x1
            dy = y2 - y1
            # length = math.hypot(dx, dy)  # 计算欧几里得距离
            length = max(math.hypot(dx, dy), 1e-6)  # 防止长度为 0
            # 根据 x 轴计算方向角度（以弧度为单位）
            angle = math.atan2(dy, dx)  # range [-pi, pi]
            angle_deg = math.degrees(angle)
            # 将角度归一化至区间 [0, 2*pi
            if angle_deg < 0:
                angle_deg += 360.0
            angle = math.radians(angle_deg)
            # （位置的）中点
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            node = {
                "type": primitive_type,
                "length": length,
                "orient_angle": angle,  # 存储弧度值
                "mid_x": cx,
                "mid_y": cy,
                "p1": (x1, y1),
                "p2": (x2, y2)
            }
            nodes.append(node)
            node_index_map.append((attr, None))
        elif "points" in attr:
            # 多线段/多边形元素（记录的是一系列的点）
            pts_str = attr["points"]
            # 该点字符串中的值可以由逗号或空格分隔。
            pts = []
            for part in pts_str.replace(",", " ").split():
                try:
                    pts.append(float(part))
                except:
                    continue
            # pts now is [x0, y0, x1, y1, ..., xn, yn]
            if len(pts) < 4:
                continue  # not a valid line
            # 如果为多边形（封闭图形），最后一个点可能会与第一个点重合，或者该图形实际上是封闭的（无需明确指定封闭属性）。
            is_polygon = attr.get("fill", None) is not None and attr.get("fill") != "none"
            # 依次遍历相邻的点对，将其视为线段。
            num_points = len(pts) // 2
            for i in range(num_points - 1):
                x1 = pts[2 * i]
                y1 = pts[2 * i + 1]
                x2 = pts[2 * (i + 1)]
                y2 = pts[2 * (i + 1) + 1]
                primitive_type = "line"
                # （将每个部分视为一个线节点）
                # 使方向保持一致
                if (x2 < x1) or (x2 == x1 and y2 < y1):
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                dx = x2 - x1
                dy = y2 - y1
                # length = math.hypot(dx, dy)
                length = max(math.hypot(dx, dy), 1e-6)
                angle = math.atan2(dy, dx)
                angle_deg = math.degrees(angle)
                if angle_deg < 0:
                    angle_deg += 360.0
                angle = math.radians(angle_deg)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                node = {
                    "type": primitive_type,
                    "length": length,
                    "orient_angle": angle,
                    "mid_x": cx,
                    "mid_y": cy,
                    "p1": (x1, y1),
                    "p2": (x2, y2)
                }
                nodes.append(node)
                # 由于本项目是手动创建的，所以对于该部分使用“None”即可。
                node_index_map.append((attr, None))
            # 如果为多边形，则还需添加从最后一个点到第一个点的线段，以闭合图形。
            if is_polygon:
                x1 = pts[2 * (num_points - 1)]
                y1 = pts[2 * (num_points - 1) + 1]
                x2 = pts[0]
                y2 = pts[1]
                primitive_type = "line"
                if (x2 < x1) or (x2 == x1 and y2 < y1):
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                dx = x2 - x1
                dy = y2 - y1
                # length = math.hypot(dx, dy)
                length = max(math.hypot(dx, dy), 1e-6)
                angle = math.atan2(dy, dx)
                angle_deg = math.degrees(angle)
                if angle_deg < 0:
                    angle_deg += 360.0
                angle = math.radians(angle_deg)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                node = {
                    "type": primitive_type,
                    "length": length,
                    "orient_angle": angle,
                    "mid_x": cx,
                    "mid_y": cy,
                    "p1": (x1, y1),
                    "p2": (x2, y2)
                }
                nodes.append(node)
                node_index_map.append((attr, None))
        elif "d" in attr:
            # 路径元素：可包含多个段（直线、弧线、贝塞尔曲线等）
            # 遍历复杂路径里的各个子路径
            for seg in path_obj:
                if isinstance(seg, Line):  # 判断该段是不是 直线段
                    primitive_type = "line"
                    x1, y1 = seg.start.real, seg.start.imag  # 使用复数 x + yj 存 2D 坐标
                    x2, y2 = seg.end.real, seg.end.imag
                    # 保证两条共线线段被拆出来时端点顺序一致
                    if (x2 < x1) or (x2 == x1 and y2 < y1):
                        x1, x2 = x2, x1
                        y1, y2 = y2, y1
                    dx = x2 - x1
                    dy = y2 - y1
                    # length = math.hypot(dx, dy)
                    length = max(math.hypot(dx, dy), 1e-6)
                    angle = math.atan2(dy, dx)  # 计算从原点(0,0)到点(x,y)的线段与x轴正方向之间的平面角度，范围 (-π, π]
                    angle_deg = math.degrees(angle) # # 转成度 [-180,180]
                    if angle_deg < 0:
                        angle_deg += 360.0  # # 映射到 [0,360)
                    angle = math.radians(angle_deg) # 再转回弧度 [0,2π)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    node = {
                        "type": primitive_type,
                        "length": length,
                        "orient_angle": angle,
                        "mid_x": cx,
                        "mid_y": cy,
                        "p1": (x1, y1),
                        "p2": (x2, y2)
                    }
                elif isinstance(seg, Arc):  # 判断该段是不是 弧线段
                    primitive_type = "arc"
                    # 在 svgpathtools 中的弧形：具有起始点、结束点、半径、旋转角度等属性。
                    x1, y1 = seg.start.real, seg.start.imag
                    x2, y2 = seg.end.real, seg.end.imag

                    length = seg.length(error=1e-4) # 计算弧的长度，error 控制精度
                    # 新版本svgpathtools内置了精度与效率自由权衡的长度计算方法，所以以下下内容被注释
                    # # 通过在弧线上采样点来计算近似长度
                    # arc_length = 0.0
                    # prev_pt = seg.start
                    # # 在弧线上选取 20 个样本点
                    # for t in range(1, 21):
                    #     pt = seg.point(t / 20.0)
                    #     arc_length += abs(pt - prev_pt)
                    #     prev_pt = pt
                    # length = arc_length

                    # 将方向定义为从起点到终点的弦线所成的角度（作为替代指标）
                    dx = x2 - x1
                    dy = y2 - y1
                    angle = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle)
                    if angle_deg < 0:
                        angle_deg += 360.0
                    angle = math.radians(angle_deg)
                    # 弧的中点（为了简便起见，可近似视为弦的中点，或者可以取 t = 0.5 进行采样）
                    mid_pt = seg.point(0.5)
                    cx = mid_pt.real
                    cy = mid_pt.imag
                    node = {
                        "type": primitive_type,
                        "length": length,
                        "orient_angle": angle,
                        "mid_x": cx,
                        "mid_y": cy,
                        "p1": (x1, y1),
                        "p2": (x2, y2)
                    }
                elif isinstance(seg, (CubicBezier, QuadraticBezier)):   # 判断该段是不是 曲线
                    primitive_type = "curve"
                    x1, y1 = seg.start.real, seg.start.imag
                    x2, y2 = seg.end.real, seg.end.imag

                    length = seg.length(error=1e-4)
                    # 新版本svgpathtools内置了精度与效率自由权衡的长度计算方法，所以以下下内容被注释
                    # # 近似曲线长度
                    # curve_length = 0.0
                    # prev_pt = seg.start
                    # for t in range(1, 21):
                    #     pt = seg.point(t / 20.0)
                    #     curve_length += abs(pt - prev_pt)
                    #     prev_pt = pt
                    # length = curve_length

                    # 方向表示：采用弦角法
                    dx = x2 - x1
                    dy = y2 - y1
                    angle = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle)
                    if angle_deg < 0:
                        angle_deg += 360.0
                    angle = math.radians(angle_deg)
                    mid_pt = seg.point(0.5)
                    cx = mid_pt.real
                    cy = mid_pt.imag
                    node = {
                        "type": primitive_type,
                        "length": length,
                        "orient_angle": angle,
                        "mid_x": cx,
                        "mid_y": cy,
                        "p1": (x1, y1),
                        "p2": (x2, y2)
                    }
                else:
                    # 未知的类型，跳过
                    continue
                nodes.append(node)
                node_index_map.append((attr, seg))
        elif ("width" in attr and "height" in attr):
            # 矩形元素（如果未由 svg2paths2 处理过，以防万一=_=）
            x = float(attr.get("x", 0.0))
            y = float(attr.get("y", 0.0))
            w = float(attr.get("width", 0.0))
            h = float(attr.get("height", 0.0))
            # 矩形的四条边线段
            rect_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            for i in range(4):
                x1, y1 = rect_points[i]
                x2, y2 = rect_points[(i + 1) % 4]
                primitive_type = "line"
                if (x2 < x1) or (x2 == x1 and y2 < y1):
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                dx = x2 - x1
                dy = y2 - y1
                # length = math.hypot(dx, dy)
                length = max(math.hypot(dx, dy), 1e-6)
                angle = math.atan2(dy, dx)
                angle_deg = math.degrees(angle)
                if angle_deg < 0:
                    angle_deg += 360.0
                angle = math.radians(angle_deg)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                node = {
                    "type": primitive_type,
                    "length": length,
                    "orient_angle": angle,
                    "mid_x": cx,
                    "mid_y": cy,
                    "p1": (x1, y1),
                    "p2": (x2, y2)
                }
                nodes.append(node)
                node_index_map.append((attr, None))
        else:
            # 注意，本项目只关注基本的几何元素，因此对于其他元素（如文本等），将跳过它们。
            continue


    num_nodes = len(nodes)  # 节点数量
    # 准备节点特征向量（以浮点数列表的形式）
    node_features = []
    # 确定类型类别（直线、弧线、圆形、椭圆形、曲线）
    type_to_idx = {"line": 0, "arc": 1, "circle": 2, "ellipse": 3, "curve": 4}
    num_types = len(type_to_idx)    # 类型数量
    for n in nodes:
        # 方向编码：利用角度的余弦和正弦值实现方向的不变性
        theta = n.get("orient_angle", 0.0)  # 取出本节点的方向角（弧度制）
        cos_val = math.cos(theta)   # 计算 cos θ
        sin_val = math.sin(theta)   # 计算 sin θ
        length = n.get("length", 0.0)   # 取出本节点的几何长度
        # 类型：独热编码
        type_idx = type_to_idx.get(n["type"], len(type_to_idx) - 1)  # 若未知情况出现，则默认选择“曲线”选项。
        type_onehot = [0.0] * num_types # 先创建一个全 0 的 one-hot 向量
        if type_idx < num_types:    # 若已知类型，则将对应位置设为 1.0，创建 one-hot 向量=_=
            type_onehot[type_idx] = 1.0
        # 拼接数值特征与类型 one-hot
        # 基础特征向量：[余弦θ值、正弦θ值、长度、类型1、...、类型N]
        # 例如：[0.5, 0.866, 10.0, 0.0, 1.0, 0.0, 0.0] 表示一个长度为 10 的直线，其方向为 30 度（弧度值约为 0.523）。
        feat_vec = [cos_val, sin_val, length] + type_onehot
        node_features.append(feat_vec)  # 把该节点特征推入总列表

    # 边的构建：根据几何关系确定相邻关系
    edges_set = set()  # 存储无向边 (i,j)（其中i,j表示两个节点的索引，对应 nodes 列表的下标，并且 i < j）
    # 收集边缘特征向量
    edge_features = []

    # 功能：用于添加带有特征的边的函数
    def add_edge(i, j):
        if i == j:
            return  # 忽略自环
        # 保证索引从小到大，得到 (u, v)
        u, v = (i, j) if i < j else (j, i)
        # 若这条无向边已存在则跳过
        if (u, v) in edges_set:
            return
        edges_set.add((u, v))   # 有序元组，确保无向边的唯一性

    # 利用空间距离：若端点非常接近或重合（在距离阈值 dist_thresh 内），则进行连接。
    # 使用共线性/平行性原则：若基本元素共线或平行且间距较小，则进行连接。
    for i in range(num_nodes):  # 外层循环遍历 节点索引 i
        for j in range(i + 1, num_nodes):   # 内层从 i+1 开始，保证只枚举一次无向组合 (i, j)，避免重复和自环
            ni = nodes[i]   # 取出节点 i 的特征向量
            nj = nodes[j]   # 取出节点 j 的特征向量
            ti = ni["type"] # 取出节点 i 的类型
            tj = nj["type"] # 取出节点 j 的类型
            # 计算 i 的任意一个端点与 j 的任意一个端点之间的距离
            close = False   # 标志位：稍后若发现端点距离 < distance_threshold 就置 True
            share_ep = False    # 标志位：若两条线段 共用同一端点（或距离极小），置 True
            min_dist = float("inf") # 初始化“端点最近距离”记录，用于后续比较并决定是否连边
            # 只有在基本图形并非完整闭合形状（如圆形/椭圆形则没有端点）的情况下，才需考虑其端点。
            endpoints_i = []    # 存储节点 i 的端点坐标
            endpoints_j = []    # 存储节点 j 的端点坐标
            if "p1" in ni and "p2" in ni:   # 若节点 i 的两个坐标到列表。
                endpoints_i.extend([ni["p1"], ni["p2"]])
            if "p1" in nj and "p2" in nj:   # 若节点 j 的两个坐标到列表。
                endpoints_j.extend([nj["p1"], nj["p2"]])
            # 如果已知端点坐标，则计算距离
            if endpoints_i and endpoints_j:     # 两节点确实都有端点时才执行
                for (xi, yi) in endpoints_i:        # 遍历 i 的端点
                    for (xj, yj) in endpoints_j:    # 遍历 j 的端点
                        dx = xi - xj
                        dy = yi - yj
                        d = math.hypot(dx, dy)  # 计算欧几里得距离
                        if d < min_dist:        # 更新最小距离
                            min_dist = d
                        if d < dist_thresh:     # 若距离小于 distance_threshold
                            close = True        # 标记“接近”，稍后可连边
                        if d < 1e-6:            # 本质上是同一个观点
                            share_ep = True     # 标记“共端点”，稍后可连边
                            close = True
                            min_dist = 0.0      # 若存在共端点，则距离为 0
            """
            为什么没有相交关系的选项?
            为了简化，通常先把最常用且计算最稳的四类关系——(1) 端点接近(2) 共端点3) 平行/共线(4) 垂直——作为连边依据。
            相交边数量往往远大于“端点-端点”边，若直接全部连边，图会变得非常稠密，反而 稀释注意力、增加显存消耗。
            若数据中交叉处已将墙线拆段并加端点，则“交叉”信息已隐式存在，无需重复
            """
            # 连接规则 1：共享端点或相邻端点
            if close:
                add_edge(i, j)
            # 连接规则 2：共线性或对齐性
            if ti == "line" and tj == "line":
                # 检查线条是否近乎平行（角度差约为 0 或 180 度）
                ang_i = math.degrees(ni["orient_angle"])    # 取出节点 i 的方向角（弧度制）
                ang_j = math.degrees(nj["orient_angle"])    # 取出节点 j 的方向角（弧度制）
                diff = abs(ang_i - ang_j)    # 计算角度差
                if diff > 180:
                    diff = 360 - diff
                parallel = diff < angle_thresh or abs(diff - 180) < angle_thresh    # 若角度差小于 angle_thresh 则认为平行
                perpendicular = abs(diff - 90) < angle_thresh    # 若角度差小于 angle_thresh 则认为垂直
                if parallel:
                    # 如果为平行线，则检查它们是否位于同一条直线上（即共线），还是只是平行但不同的两条直线。
                    # 通过检查一条线的端点到另一条线的距离来近似判断两条线的平行关系。
                    # 计算第 i 条线的方程（以 Ax + By + C 的形式表示），以便测量 j 点的中点与该线的距离
                    x1_i, y1_i = ni["p1"]    # 取出节点 i 的两个端点坐标
                    x2_i, y2_i = ni["p2"]    # 取出节点 i 的两个端点坐标
                    # 第 i 条线的系数
                    A = y2_i - y1_i
                    B = -(x2_i - x1_i)
                    C = - (A * x1_i + B * y1_i)
                    # 从 j 点的中点到直线 i 的距离
                    mx_j = nj.get("mid_x", (nj.get("p1", (0, 0))[0] + nj.get("p2", (0, 0))[0]) / 2.0)
                    my_j = nj.get("mid_y", (nj.get("p1", (0, 0))[1] + nj.get("p2", (0, 0))[1]) / 2.0)
                    # 线点距离
                    # dist_line = abs(A * mx_j + B * my_j + C) / math.hypot(A, B) if math.hypot(A, B) > 1e-6 else float(
                    #     "inf")
                    denom = max(math.hypot(A, B), 1e-6)
                    dist_line = abs(A * mx_j + B * my_j + C) / denom

                    if dist_line < collinearity_dist:
                        # 可能是同一共线结构的一部分（例如，由缝隙分隔的墙段）
                        add_edge(i, j)
                if perpendicular:
                    # 如果为垂直且共用端点（类似于一个角）的形状（这是由 share_ep 函数已经完成的操作）。
                    # 如果并非是共享端点而是存在非常接近的重合部分，可以添加这一项，但可以跳过某些内容以避免过多重复。
                    pass
            # 附加规则：如果一个是圆形/椭圆形，而另一个与之相接（距离 d 约等于半径之差），则由于复杂性原因可忽略不计。
            # （为了简便起见，除非遵循上述“靠近端点”的规则，否则不会将圆/椭圆与其他图形相连接。）
    # 如果指定了，则限制每个节点的最大边数（以控制图的度数）
    if max_edges_per_node:
        # 把无向边集合 edges_set 转换成“邻接表”字典
        adj = {i: set() for i in range(num_nodes)}  # 初始化邻接表,创建一个字典adj,键：节点索引 i (0 → num_nodes-1);值：空集合 set()，稍后用于存放该节点的所有邻居。
        for (u, v) in edges_set:    # edges_set 中的每个元组 (u, v) 表示一条无向边
            adj[u].add(v)   # 把 u 的邻居集合里加 v
            adj[v].add(u)   # 把 v 的邻居集合里加 u
        # 加强限制
        for i in range(num_nodes):  # 遍历每个节点 i
            if len(adj[i]) > max_edges_per_node:    # 若节点 i 的邻居数量超过 max_edges_per_node
                # 随机截断相邻节点，使其不超过每个节点的最大边数限制。
                # （或者也可以选择“最接近”或者采用一些启发式方法，例如边权学习、动态阈值等）
                nbrs = list(adj[i])     # 把节点 i 的邻居集合转换为列表
                random.shuffle(nbrs)    # 随机打乱邻居列表
                # 去除多余的部分
                for nb in nbrs[max_edges_per_node:]:    # 仅保留前 max_edges_per_node 个，其余邻居全部删除
                    # 从边集合和邻接表中移除边 (i, nb)
                    u, v = (i, nb) if i < nb else (nb, i)   # 先把无向边写成 (小索引, 大索引) 形式，与 edges_set 存储格式一致
                    if (u, v) in edges_set:         # 若边 (u, v) 在边集合中
                        edges_set.remove((u, v))    # 从 edges_set 中删除该边
                        if nb in adj[i]: adj[i].remove(nb)    # 从节点 i 的邻居集合中删除节点 nb
                        if i in adj[nb]: adj[nb].remove(i)    # 从节点 nb 的邻居集合中删除节点 i
    # 现在，edges_set 包含了最终的无向边。
    # 为每条边计算边缘特征，并准备有向边列表
    edge_index = [] # 初始化边索引列表
    for (u, v) in edges_set:    # 遍历之前构建好的无向边集合
        ni = nodes[u]   # 取出两端点的 节点属性字典，方便后面访问端点坐标、类型等
        nj = nodes[v]
        # 预先计算两个方向共有的特征
        min_dist = float("inf")    # 为了确保准确性，重新计算 min_dist（此前已在连接性循环中存储过）
        # 如果节点中至少有一个不含端点（如圆、椭圆），endpoints_i 或 endpoints_j 为空，双重循环不会执行；此时 min_dist 仍是 inf
        endpoints_i = []
        endpoints_j = []
        # 将两个端点的坐标加入列表
        if "p1" in ni and "p2" in ni:
            endpoints_i.extend([ni["p1"], ni["p2"]])
        if "p1" in nj and "p2" in nj:
            endpoints_j.extend([nj["p1"], nj["p2"]])
        if endpoints_i and endpoints_j:
            for (xi, yi) in endpoints_i:
                for (xj, yj) in endpoints_j:
                    d = math.hypot(xi - xj, yi - yj)
                    if d < min_dist:
                        min_dist = d
        if min_dist == float("inf"):
            # 例如，将一个没有端点的圆与其他对象相连接，则改用“中心点之间距离”作为替代
            xi = ni.get("mid_x", ni.get("center_x", 0.0))   # 若节点 i 不含中点坐标，则取中心坐标
            yi = ni.get("mid_y", ni.get("center_y", 0.0))
            xj = nj.get("mid_x", nj.get("center_x", 0.0))   # 若节点 j 不含中点坐标，则取中心坐标
            yj = nj.get("mid_y", nj.get("center_y", 0.0))
            min_dist = math.hypot(xi - xj, yi - yj)   # 计算两个节点中心之间的距离
        # 从 i 点的中心到 j 点的中心的偏移向量
        xi = ni.get("mid_x", ni.get("center_x", 0.0))
        yi = ni.get("mid_y", ni.get("center_y", 0.0))
        xj = nj.get("mid_x", nj.get("center_x", 0.0))
        yj = nj.get("mid_y", nj.get("center_y", 0.0))
        offset_x = xj - xi  # 得到从节点 i 指向节点 j 的平移量(Δx, Δy)
        offset_y = yj - yi
        # 方向差异（以度为单位，范围为 0 - 180 度）
        ang_i = math.degrees(ni.get("orient_angle", 0.0))   # 若节点 i 不含方向角度，则取 0 度
        ang_j = math.degrees(nj.get("orient_angle", 0.0))   # 若节点 j 不含方向角度，则取 0 度
        diff = abs(ang_i - ang_j)   # 计算两个节点方向角度的差异
        if diff > 180: diff = 360 - diff    # 确保差异角度在 0-180 度之间
        orient_diff = diff   # 记录两个节点方向角度的差异
        # 长度比
        len_i = ni.get("length", 0.0)   # 若节点 i 不含长度，则取 0
        len_j = nj.get("length", 0.0)   # 若节点 j 不含长度，则取 0
        if len_i < 1e-6 or len_j < 1e-6:    # 若节点 i 或 j 不含长度，则长度比取 0
            length_ratio = 0.0
        else:   # 若节点 i 和 j 都含有长度，则计算长度比（较短长度 ÷ 较长长度）
            smaller = len_i if len_i < len_j else len_j
            larger = len_j if len_j > len_i else len_i
            length_ratio = smaller / larger
        # 几何关系标志
        parallel_flag = 0.0         # 平行标志
        perpendicular_flag = 0.0    # 垂直标志
        share_ep_flag = 0.0         # 共享端点标志
        # 使用“orient_diff”函数来确定平行/垂直关系
        if orient_diff < angle_thresh or abs(orient_diff - 180) < angle_thresh: # 若两段方向角差非常接近 0° 或 180°（允许 angle_thresh 的容忍带，例如 ±5°）
            parallel_flag = 1.0    # 说明它们平行
        if abs(orient_diff - 90) < angle_thresh: # 若两段方向角差非常接近 90°（允许 angle_thresh 的容忍带，例如 ±5°）
            perpendicular_flag = 1.0    # 说明它们垂直
        if endpoints_i and endpoints_j:    # 若两个节点都有端点
            for (xi, yi) in endpoints_i:    # 遍历节点 i 的所有端点
                for (xj, yj) in endpoints_j:    # 遍历节点 j 的所有端点
                    if math.hypot(xi - xj, yi - yj) < 1e-6: # 若两个端点的距离非常接近 0（允许 1e-6 的容忍带）
                        share_ep_flag = 1.0    # 说明它们共享一个端点
                        break   # 找到共端点即可跳出内层
                if share_ep_flag == 1.0:
                    break   # 已经确认共端点，再跳出外层
        # 构建边缘特征向量：[距离、偏移量_x、偏移量_y、方向差异、长度比例、是否平行、是否垂直、是否共享端点]]
        edge_feat_uv = [
            min_dist,           # 两图元最小端点距（或中心距）
            offset_x, offset_y, # 从 u → v 的位移向量 Δx, Δy
            orient_diff,        # 方向角差 |θu – θv|（度）
            length_ratio,       # 较短 / 较长 的长度比
            parallel_flag,      # 是否平行   (0/1)
            perpendicular_flag, # 是否垂直   (0/1)
            share_ep_flag       # 是否共端点 (0/1)
        ]
        # 构建逆向边的特征向量（与正向边相同，只是方向相反）
        edge_feat_vu = [
            min_dist,               # 两图元最小端点距（或中心距） → 不变
            -offset_x, -offset_y,   # 方向反过来 ↔ 位移取负
            orient_diff,            # 方向差对称 → 不变
            length_ratio,           # 长度比对称 → 不变
            parallel_flag,          # 平行标志对称 → 不变
            perpendicular_flag,     # 垂直标志对称 → 不变
            share_ep_flag           # 共享端点标志对称 → 不变
        ]
        # 添加有向边和特征
        edge_index.append([u, v])   # u → v，逐行存 [src, dst]
        edge_features.append(edge_feat_uv)  # 写入 u → v 的特征
        edge_index.append([v, u])   # v → u，逐行存 [src, dst]
        edge_features.append(edge_feat_vu)  # 写入 v → u 的特征
    # 准备结果结构
    graph_data = {
        "node_feats": node_features,    # 节点特征矩阵，含余弦、正弦、长度、one-hot 等
        "edge_index": edge_index,       # 边索引矩阵，每行 [src, dst] 表示一条有向边
        "edge_feats": edge_features     # 边特征矩阵，每行 [距离、偏移量_x、偏移量_y、方向差异、长度比例、是否平行、是否垂直、是否共享端点]
    }
    # 保存为 JSON 格式以进行缓存
    json_path = svg_file.rsplit(".", 1)[0] + ".json"
    try:
        with open(json_path, "w") as f:
            json.dump(graph_data, f)
    except Exception as e:
        print(f"Warning: Could not save JSON cache for {svg_file}: {e}")
    return graph_data,nodes

def visualize_graph(graph_data, nodes_raw=None,
                    figsize=(8, 8), node_size=40,
                    edge_width=1.0, save_path=None, dpi=300):
    """
    将 parse_svg() 生成的 graph_data 以 NetworkX 可视化。
    ----------
    参数
    • graph_data : dict
        包含 "node_feats", "edge_index", "edge_feats" 3 个键。
    • figsize    : tuple
        matplotlib 画布尺寸。
    • node_size  : int/float
        节点散点大小。
    • edge_width : float
        边线宽度。
    • save_path  : str or None
        若给定文件路径（例如 "out.png"），则保存图像；否则直接 plt.show()。
    • dpi        : int
        保存图片的分辨率，仅 save_path 不为 None 时生效。
    ----------
    注意
    - 节点坐标优先使用 `mid_x/mid_y`；若不存在则使用 `center_x/center_y`。
    - 若位置缺失则放置在 (0,0)。
    """
    G = nx.Graph()
    for src, dst in graph_data["edge_index"]:
        G.add_edge(src, dst)

    # 1) 生成节点坐标
    if nodes_raw is not None:
        pos = {}
        for idx, n in enumerate(nodes_raw):
            x = n.get("mid_x", n.get("center_x", 0.0))
            y = n.get("mid_y", n.get("center_y", 0.0))
            pos[idx] = (x, y)
    else:
        pos = nx.spring_layout(G, seed=42)

    # 2) 画图
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="#ff8c00")
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.6)
    plt.axis("off")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from pprint import pprint

    def parse_svg_test(svg_file):
        """
        读取 SVG，逐条打印 path 对象和对应的属性字典。
        :param svg_file: str, SVG 文件路径
        """
        # svg2paths2 返回 (paths, attr_list, svg_attr)
        paths, attr_list, _ = svg2paths2(svg_file)

        print(f"\n=== 共解析到 {len(paths)} 条路径 ===")
        for idx, (p, attr) in enumerate(zip(paths, attr_list)):
            print(f"\n-- Path #{idx} --")
            print("几何对象 (Path)：", p)
            print("Segment 数量   ：", len(p))
            print("属性字典 (attr)：")
            pprint(attr)

    svg_path = r"D:/GRANT-CAD/floorplancad_v2/svg_folder/0006-0043.svg"

    # 测试svg2paths2解析后的SVG文件内容
    # parse_svg_test(svg_path)

    # 测试 visualize_graph() 可视化图(需要让 parse_svg 同时返回 nodes)
    cfg = {}
    graph_data,nodes = parse_svg(svg_path,cfg)
    visualize_graph(
        graph_data,
        nodes_raw=nodes,  # 传入 nodes 才能用真实几何坐标
        figsize=(10, 10),
        node_size=30,
        edge_width=0.8,
        save_path="sample_graph.png"
    )
