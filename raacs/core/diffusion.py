# raacs/core/diffusion.py
"""
扩散层算法实现 - Personalized PageRank (PPR) 上下文窗口计算。

提供：
- CodeGraphBuilder: 从 pydeps JSON 构建加权依赖图
- GraphVisualizer: 生成交互式依赖图可视化
- run_ppr: 运行 PPR 算法的便捷函数

注意：此模块依赖 networkx。如果未安装，导入类时会抛出 ImportError。
"""

import json
import os
import ast
from typing import Dict, List, Tuple, Optional

# --- RAACS 策略配置 ---
# 定义边权常量
WEIGHT_INHERITANCE = 3.0  # 继承/Mixin：极强依赖
WEIGHT_TYPE_HINT = 2.0    # 类型注解：强语义依赖
WEIGHT_IMPORT = 1.0       # 普通引用：基础依赖

# 定义颜色常量 (用于可视化)
COLOR_INHERITANCE = "#FF4500"  # OrangeRed: 显眼，代表强血缘
COLOR_TYPE_HINT = "#1E90FF"    # DodgerBlue: 清晰，代表类型约束
COLOR_IMPORT = "#808080"       # Gray: 低调，作为背景噪音


def _get_networkx():
    """延迟加载 networkx"""
    try:
        import networkx as nx
        return nx
    except ImportError:
        raise ImportError("请安装 networkx: pip install networkx")


def _get_viz_deps():
    """延迟加载可视化依赖"""
    try:
        from pyvis.network import Network
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        return Network, cm, mcolors
    except ImportError:
        raise ImportError("请安装可视化依赖: pip install pyvis matplotlib")


class CodeGraphBuilder:
    """负责解析 JSON、分析 AST 并构建加权图"""

    def __init__(self, pydeps_json_path: str):
        self.pydeps_json_path = pydeps_json_path
        nx = _get_networkx()
        self.graph = nx.DiGraph()
        self.module_data = {}
        self._nx = nx

    def load_data(self):
        if not os.path.exists(self.pydeps_json_path):
            print(f"[!] Error: File {self.pydeps_json_path} not found.")
            return
        with open(self.pydeps_json_path, 'r', encoding='utf-8') as f:
            self.module_data = json.load(f)
        print(f"[*] Loaded {len(self.module_data)} modules.")

    def _analyze_ast_weight(self, source_path: str, target_module_names: List[str]) -> Dict[str, float]:
        """通过 AST 区分引用类型：继承 vs 类型 vs 普通"""
        default_weights = {name: WEIGHT_IMPORT for name in target_module_names}

        if not source_path or not os.path.exists(source_path):
            return default_weights

        try:
            with open(source_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception:
            return default_weights

        # 1. 建立 Import 别名表
        local_alias_map = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    local_alias_map[name.asname or name.name] = name.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for name in node.names:
                        full_name = f"{node.module}.{name.name}"
                        local_alias_map[name.asname or name.name] = full_name

        refined_weights = default_weights.copy()

        # 2. 扫描 AST 提升权重
        for node in ast.walk(tree):
            # 策略 A: 继承检测 (Inheritance)
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_id = None
                    if isinstance(base, ast.Name):
                        base_id = base.id
                    elif isinstance(base, ast.Attribute):
                        base_id = base.attr

                    if base_id and base_id in local_alias_map:
                        imported_name = local_alias_map[base_id]
                        for target in target_module_names:
                            if imported_name.startswith(target):
                                refined_weights[target] = max(refined_weights[target], WEIGHT_INHERITANCE)

            # 策略 B: 类型检测 (Type Hint)
            elif isinstance(node, ast.FunctionDef):
                # 检查返回值
                if node.returns:
                    type_id = None
                    if isinstance(node.returns, ast.Name):
                        type_id = node.returns.id
                    if type_id and type_id in local_alias_map:
                        imported_name = local_alias_map[type_id]
                        for target in target_module_names:
                            if imported_name.startswith(target):
                                refined_weights[target] = max(refined_weights[target], WEIGHT_TYPE_HINT)
                # 检查参数
                for arg in node.args.args:
                    if arg.annotation and isinstance(arg.annotation, ast.Name):
                        type_id = arg.annotation.id
                        if type_id and type_id in local_alias_map:
                            imported_name = local_alias_map[type_id]
                            for target in target_module_names:
                                if imported_name.startswith(target):
                                    refined_weights[target] = max(refined_weights[target], WEIGHT_TYPE_HINT)

        return refined_weights

    def build_graph(self):
        print("[*] Building Semantic Graph...")
        for module_name, metadata in self.module_data.items():
            self.graph.add_node(module_name)
            source_path = metadata.get('path')
            imports = metadata.get('imports', [])
            if not imports:
                continue

            weights_map = self._analyze_ast_weight(source_path, imports)

            for target_module in imports:
                if target_module in self.module_data:
                    w = weights_map.get(target_module, WEIGHT_IMPORT)
                    self.graph.add_edge(module_name, target_module, weight=w)
        print(f"[*] Graph Built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def run_ppr(self, target_module: str, top_k: int = 10, alpha: float = 0.85) -> List[Tuple[str, float]]:
        """
        运行 Personalized PageRank 算法。

        Args:
            target_module: 目标模块名
            top_k: 返回前 k 个相关模块
            alpha: PageRank 阻尼系数

        Returns:
            [(module_name, score), ...] 按分数降序排列
        """
        if target_module not in self.graph:
            return []
        personalization = {n: 0.0 for n in self.graph.nodes()}
        personalization[target_module] = 1.0
        try:
            scores = self._nx.pagerank(self.graph, alpha=alpha, personalization=personalization, weight='weight')
        except ZeroDivisionError:
            return []
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(n, s) for n, s in sorted_scores if n != target_module and s > 0.0001][:top_k]


class GraphVisualizer:
    """依赖图可视化器"""

    def __init__(self, builder: CodeGraphBuilder):
        self._Network, self._cm, self._mcolors = _get_viz_deps()
        self.builder = builder
        self.graph = builder.graph

    def generate_interactive_graph(self, target_node: str, ppr_scores: list, output_file: str = "ppr_graph.html"):
        """
        生成交互式依赖图可视化。

        Args:
            target_node: 目标节点（中心）
            ppr_scores: PPR 分数列表 [(node, score), ...]
            output_file: 输出 HTML 文件路径
        """
        print(f"[*] Generating visualization for target: {target_node}...")

        # 初始化画布: 深色背景，白色文字
        net = self._Network(height="900px", width="100%", bgcolor="#222222", font_color="white",
                            select_menu=True, filter_menu=True, cdn_resources='in_line')

        # 准备子图数据
        top_nodes = {node for node, score in ppr_scores}
        if target_node in self.graph:
            top_nodes.add(target_node)
        subgraph = self.graph.subgraph(top_nodes)

        # 节点颜色映射 (PPR Score -> Heatmap Color)
        max_score = ppr_scores[0][1] if ppr_scores else 1.0
        cmap = self._cm.get_cmap('plasma')  # 使用 'plasma' 配色方案
        score_map = {node: score for node, score in ppr_scores}
        score_map[target_node] = max_score * 1.2

        # --- 添加节点 ---
        for node in subgraph.nodes():
            score = score_map.get(node, 0.0)

            # Target 节点特殊样式
            if node == target_node:
                color = "#00FF00"  # 荧光绿
                shape = "star"
                size = 50
                label = f"🎯 {node}"
                title = "Target Context Window Center"
            else:
                # 普通节点根据分数变色
                ratio = score / max_score if max_score > 0 else 0
                rgba = cmap(ratio)
                color = self._mcolors.to_hex(rgba)
                shape = "dot"
                size = 10 + (ratio * 30)  # 分数越高节点越大
                label = node
                title = f"{node}\nPPR Score: {score:.4f}"

            net.add_node(
                node, label=label, title=title, color=color, size=size, shape=shape,
                borderWidth=1, borderWidthSelected=3,
                font={'size': 14, 'face': 'arial', 'color': 'white'}
            )

        # --- 添加边 ---
        for source, target, data in subgraph.edges(data=True):
            weight = data.get('weight', 1.0)

            # 默认样式 (Import)
            color = COLOR_IMPORT
            width = 1
            dashes = False
            title = f"Import (w={weight})"

            # 继承关系 (高亮)
            if weight >= WEIGHT_INHERITANCE:
                color = COLOR_INHERITANCE
                width = 4
                dashes = False
                title = f"Inherits (w={weight})"

            # 类型引用 (虚线)
            elif weight >= WEIGHT_TYPE_HINT:
                color = COLOR_TYPE_HINT
                width = 2
                dashes = True
                title = f"Type Hint (w={weight})"

            net.add_edge(
                source, target,
                color=color,
                width=width,
                dashes=dashes,
                title=title,
                arrows={'to': {'enabled': True, 'scaleFactor': 1.0}}
            )

        # 物理模拟设置
        net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=200)

        try:
            net.save_graph(output_file)
            print(f"[*] Visualization saved to: {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"[!] Error saving visualization: {e}")


def run_ppr(pydeps_json_path: str, target_module: str, top_k: int = 10, alpha: float = 0.85) -> List[Tuple[str, float]]:
    """
    便捷函数：运行 PPR 算法获取上下文窗口。

    Args:
        pydeps_json_path: pydeps JSON 文件路径
        target_module: 目标模块名
        top_k: 返回前 k 个相关模块
        alpha: PageRank 阻尼系数

    Returns:
        [(module_name, score), ...] 按分数降序排列
    """
    builder = CodeGraphBuilder(pydeps_json_path)
    builder.load_data()
    if not builder.module_data:
        return []
    builder.build_graph()
    return builder.run_ppr(target_module, top_k=top_k, alpha=alpha)


__all__ = [
    "CodeGraphBuilder",
    "GraphVisualizer",
    "run_ppr",
    "WEIGHT_INHERITANCE",
    "WEIGHT_TYPE_HINT",
    "WEIGHT_IMPORT",
    "COLOR_INHERITANCE",
    "COLOR_TYPE_HINT",
    "COLOR_IMPORT",
]
