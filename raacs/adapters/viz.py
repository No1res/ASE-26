# raacs/adapters/viz.py
"""
RAACS 可视化适配器 - 生成交互式 HTML 依赖图可视化。

提供：
- RoleGraphVisualizer: 角色感知的依赖图可视化器（基于角色着色）
- PPRGraphVisualizer: PPR 分数可视化器（基于分数热力图着色）
- generate_role_viz: 便捷函数

依赖：pyvis, matplotlib
"""

import os
import ast
import warnings
from typing import Dict, List, Optional, Any, Tuple


# === 边权重常量 (用于 AST 分析) ===
WEIGHT_INHERITANCE = 3.0  # 继承/Mixin：极强依赖
WEIGHT_TYPE_HINT = 2.0    # 类型注解：强语义依赖
WEIGHT_IMPORT = 1.0       # 普通引用：基础依赖

# === 边颜色常量 ===
COLOR_INHERITANCE = "#FF4500"  # OrangeRed: 显眼，代表强血缘
COLOR_TYPE_HINT = "#1E90FF"    # DodgerBlue: 清晰，代表类型约束
COLOR_IMPORT = "#808080"       # Gray: 低调，作为背景噪音

# === 角色颜色映射 ===
ROLE_COLORS = {
    "TEST": "#9E9E9E",       # 灰色 - 测试代码
    "NAMESPACE": "#607D8B",   # 蓝灰 - 命名空间/包
    "INTERFACE": "#00BCD4",   # 青色 - 接口/抽象
    "SCHEMA": "#8BC34A",      # 浅绿 - 数据模型
    "ADAPTER": "#FF9800",     # 橙色 - 适配器
    "CONFIG": "#9C27B0",      # 紫色 - 配置
    "SCRIPT": "#F44336",      # 红色 - 脚本/入口
    "UTIL": "#2196F3",        # 蓝色 - 工具函数
    "LOGIC": "#4CAF50",       # 绿色 - 业务逻辑
    "UNKNOWN": "#757575",     # 深灰 - 未知
}


def _get_pyvis():
    """延迟加载 pyvis"""
    try:
        from pyvis.network import Network
        return Network
    except ImportError:
        raise ImportError("请安装可视化依赖: pip install pyvis")


def _get_matplotlib_colors():
    """延迟加载 matplotlib 颜色工具"""
    try:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        return cm, mcolors
    except ImportError:
        raise ImportError("请安装可视化依赖: pip install matplotlib")


def _analyze_ast_weight(source_path: str, target_module_names: List[str]) -> Dict[str, float]:
    """
    通过 AST 区分引用类型：继承 vs 类型 vs 普通

    Returns:
        {module_name: weight} 映射
    """
    default_weights = {name: WEIGHT_IMPORT for name in target_module_names}

    if not source_path or not os.path.exists(source_path):
        return default_weights

    try:
        with open(source_path, "r", encoding="utf-8") as f:
            source = f.read()
        # 抑制 SyntaxWarning（被分析代码中的无效转义序列等）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            tree = ast.parse(source)
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


class RoleGraphVisualizer:
    """
    角色感知的依赖图可视化器。

    节点颜色 = 角色
    节点大小 = 度数
    边颜色 = 依赖类型（继承/类型/普通）
    """

    def __init__(self, dep_map: Dict[str, Dict], role_results: Optional[Dict[str, Any]] = None):
        """
        Args:
            dep_map: 依赖图字典 (来自 StaticImportScanner)
            role_results: 角色分析结果字典 (可选)
        """
        self.dep_map = dep_map
        self.role_results = role_results or {}

    def _filter_nodes(self, max_nodes: Optional[int] = None,
                      min_degree: int = 0,
                      exclude_roles: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        过滤节点，返回子图。
        
        Args:
            max_nodes: 最大节点数（按度数重要性排序选取）
            min_degree: 最小度数阈值（总度数 = in + out）
            exclude_roles: 排除的角色列表（如 ['TEST', 'NAMESPACE']）
            
        Returns:
            过滤后的 dep_map 子图
        """
        if not max_nodes and min_degree == 0 and not exclude_roles:
            return self.dep_map
        
        # 计算每个节点的度数
        node_degrees = {}
        for module, data in self.dep_map.items():
            in_deg = len(data.get('imported_by', []))
            out_deg = len(data.get('imports', []))
            node_degrees[module] = in_deg + out_deg
        
        # 过滤条件
        exclude_roles = set(exclude_roles or [])
        filtered_modules = set()
        
        for module, data in self.dep_map.items():
            # 检查度数
            if node_degrees[module] < min_degree:
                continue
            
            # 检查角色
            if exclude_roles:
                role_info = self._get_role_info(module, data.get('path', ''))
                if role_info.get('role', 'UNKNOWN') in exclude_roles:
                    continue
            
            filtered_modules.add(module)
        
        # 如果指定了 max_nodes，按度数排序取前 N
        if max_nodes and len(filtered_modules) > max_nodes:
            sorted_modules = sorted(filtered_modules, 
                                   key=lambda m: node_degrees[m], 
                                   reverse=True)
            filtered_modules = set(sorted_modules[:max_nodes])
        
        # 构建子图
        filtered_dep_map = {}
        for module in filtered_modules:
            data = self.dep_map[module].copy()
            # 只保留子图内部的边
            data['imports'] = [imp for imp in data.get('imports', []) 
                              if imp in filtered_modules]
            data['imported_by'] = [imp for imp in data.get('imported_by', []) 
                                   if imp in filtered_modules]
            filtered_dep_map[module] = data
        
        return filtered_dep_map

    def generate_html(self, output_path: str = "dependency_graph.html",
                      title: str = "RAACS Dependency Graph",
                      height: str = "900px",
                      show_labels: bool = True,
                      physics: bool = True,
                      max_nodes: Optional[int] = None,
                      min_degree: int = 0,
                      exclude_roles: Optional[List[str]] = None,
                      layout: str = "hierarchical") -> str:
        """
        生成交互式 HTML 可视化。
        
        Args:
            output_path: 输出 HTML 文件路径
            title: 图表标题
            height: 图表高度
            show_labels: 是否显示节点标签
            physics: 是否启用物理模拟
            max_nodes: 最大节点数（按度数重要性排序选取，None 表示不限制）
            min_degree: 最小度数阈值（总度数 = in + out，默认 0）
            exclude_roles: 排除的角色列表（如 ['TEST', 'NAMESPACE']）
            layout: 布局方式 - "hierarchical"(层次布局), "physics"(力导向), "role"(按角色分组)
            
        Returns:
            输出文件的绝对路径
        """
        Network = _get_pyvis()

        # 过滤节点
        filtered_dep_map = self._filter_nodes(max_nodes, min_degree, exclude_roles)
        
        if not filtered_dep_map:
            print("[Viz] Warning: No nodes to display after filtering")
            return ""
        
        original_count = len(self.dep_map)
        filtered_count = len(filtered_dep_map)
        if filtered_count < original_count:
            print(f"[Viz] Filtered: {original_count} -> {filtered_count} nodes "
                  f"(max_nodes={max_nodes}, min_degree={min_degree}, exclude_roles={exclude_roles})")

        # 创建网络图：深色背景，白色文字
        net = Network(
            height=height,
            width="100%",
            bgcolor="#222222",
            font_color="white",
            directed=True,
            select_menu=True,
            filter_menu=True,
            cdn_resources='in_line'
        )
        
        # 计算节点层级（用于层次布局）
        node_levels = self._compute_node_levels(filtered_dep_map) if layout == "hierarchical" else {}

        # 计算节点度数（使用过滤后的图）
        in_degrees = {}
        out_degrees = {}
        for module, data in filtered_dep_map.items():
            out_degrees[module] = len(data.get('imports', []))
            in_degrees[module] = len(data.get('imported_by', []))

        max_degree = max(max(in_degrees.values(), default=1), max(out_degrees.values(), default=1))

        # === 添加节点（使用过滤后的图） ===
        for module_name, data in filtered_dep_map.items():
            role_info = self._get_role_info(module_name, data.get('path', ''))
            role = role_info.get('role', 'UNKNOWN')
            confidence = role_info.get('confidence', 0.5)
            layer = role_info.get('layer', 'UNKNOWN')

            # 节点颜色（根据角色）
            color = ROLE_COLORS.get(role, ROLE_COLORS['UNKNOWN'])

            # 节点大小（根据度数）
            degree = in_degrees.get(module_name, 0) + out_degrees.get(module_name, 0)
            size = 10 + (degree / max_degree) * 30 if max_degree > 0 else 15

            # 节点标签
            short_name = module_name.split('.')[-1] if show_labels else ""

            # 悬停提示
            tooltip = self._build_tooltip(module_name, role, confidence, layer,
                                          in_degrees.get(module_name, 0),
                                          out_degrees.get(module_name, 0))

            # 根据布局计算节点位置
            node_kwargs = {
                'label': short_name,
                'title': tooltip,
                'color': color,
                'size': size,
                'borderWidth': 1,
                'borderWidthSelected': 3,
                'font': {'size': 14, 'face': 'arial', 'color': 'white'}
            }
            
            # 层次布局：设置 level
            if layout == "hierarchical" and module_name in node_levels:
                node_kwargs['level'] = node_levels[module_name]
            
            # 角色分组布局：按角色设置 x 位置
            elif layout == "role":
                role_x_positions = {
                    'SCRIPT': 0, 'ADAPTER': 1, 'LOGIC': 2, 'UTIL': 3,
                    'SCHEMA': 4, 'CONFIG': 5, 'INTERFACE': 6, 
                    'TEST': 7, 'NAMESPACE': 8, 'UNKNOWN': 9
                }
                x_pos = role_x_positions.get(role, 5) * 200
                node_kwargs['x'] = x_pos
                node_kwargs['physics'] = False  # 固定 x 轴
            
            net.add_node(module_name, **node_kwargs)

        # === 添加边（带权重着色，使用过滤后的图） ===
        for module_name, data in filtered_dep_map.items():
            imports = data.get('imports', [])
            source_path = data.get('path', '')

            # 分析 AST 获取边权重
            weights_map = _analyze_ast_weight(source_path, imports)

            for imported in imports:
                if imported in filtered_dep_map:
                    weight = weights_map.get(imported, WEIGHT_IMPORT)

                    # 根据权重设置边样式
                    if weight >= WEIGHT_INHERITANCE:
                        color = COLOR_INHERITANCE
                        width = 4
                        dashes = False
                        edge_title = f"Inherits (w={weight})"
                    elif weight >= WEIGHT_TYPE_HINT:
                        color = COLOR_TYPE_HINT
                        width = 2
                        dashes = True
                        edge_title = f"Type Hint (w={weight})"
                    else:
                        color = COLOR_IMPORT
                        width = 1
                        dashes = False
                        edge_title = f"Import (w={weight})"

                    net.add_edge(
                        module_name,
                        imported,
                        color=color,
                        width=width,
                        dashes=dashes,
                        title=edge_title,
                        arrows={'to': {'enabled': True, 'scaleFactor': 1.0}}
                    )

        # 布局和物理模拟设置
        if layout == "hierarchical":
            # 层次布局：从上到下，依赖方向向下
            net.set_options('''
            {
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "direction": "UD",
                        "sortMethod": "directed",
                        "levelSeparation": 150,
                        "nodeSpacing": 120,
                        "treeSpacing": 200,
                        "blockShifting": true,
                        "edgeMinimization": true,
                        "parentCentralization": true
                    }
                },
                "physics": {
                    "enabled": false
                },
                "edges": {
                    "smooth": {
                        "type": "cubicBezier",
                        "forceDirection": "vertical"
                    }
                }
            }
            ''')
        elif layout == "role":
            # 角色分组布局：按角色分列，物理模拟只作用于 y 轴
            net.set_options('''
            {
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -3000,
                        "centralGravity": 0.1,
                        "springLength": 150,
                        "springConstant": 0.02,
                        "damping": 0.5
                    },
                    "stabilization": {
                        "enabled": true,
                        "iterations": 200
                    }
                },
                "edges": {
                    "smooth": {
                        "type": "curvedCW",
                        "roundness": 0.2
                    }
                }
            }
            ''')
        elif physics:
            # 力导向布局：优化参数使布局更稳定
            net.set_options('''
            {
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "gravitationalConstant": -5000,
                        "centralGravity": 0.5,
                        "springLength": 180,
                        "springConstant": 0.04,
                        "damping": 0.3,
                        "avoidOverlap": 0.5
                    },
                    "stabilization": {
                        "enabled": true,
                        "iterations": 300,
                        "updateInterval": 25
                    }
                },
                "edges": {
                    "smooth": {
                        "type": "continuous"
                    }
                }
            }
            ''')
        else:
            net.toggle_physics(False)

        # 保存图表
        net.save_graph(output_path)

        # 注入图例
        self._inject_legend(output_path, title)

        abs_path = os.path.abspath(output_path)
        print(f"[Viz] HTML visualization saved to: {abs_path}")
        return abs_path

    def _compute_node_levels(self, dep_map: Dict[str, Dict]) -> Dict[str, int]:
        """
        计算节点的层级（用于层次布局）。
        
        使用拓扑排序的思想：
        - 没有出度的节点（叶子节点/被依赖最多）在底层
        - 依赖其他节点的模块在上层
        
        Returns:
            {module_name: level} 映射，level 越小越靠上
        """
        # 计算每个节点的"深度" - 从该节点到叶子节点的最长路径
        levels = {}
        
        def get_level(module: str, visited: set) -> int:
            if module in levels:
                return levels[module]
            if module in visited:
                return 0  # 循环依赖，返回 0
            
            visited.add(module)
            imports = dep_map.get(module, {}).get('imports', [])
            
            if not imports:
                # 叶子节点
                levels[module] = 0
                return 0
            
            # 该节点的层级 = max(子节点层级) + 1
            max_child_level = 0
            for imp in imports:
                if imp in dep_map:
                    child_level = get_level(imp, visited.copy())
                    max_child_level = max(max_child_level, child_level)
            
            levels[module] = max_child_level + 1
            return levels[module]
        
        # 计算所有节点的层级
        for module in dep_map:
            if module not in levels:
                get_level(module, set())
        
        return levels

    def _get_role_info(self, module_name: str, file_path: str) -> Dict:
        """获取模块的角色信息"""
        if file_path and file_path in self.role_results:
            result = self.role_results[file_path]
            return {
                'role': self._extract_role(result),
                'confidence': self._extract_confidence(result),
                'layer': self._extract_layer(result)
            }

        for path, result in self.role_results.items():
            if hasattr(result, 'module_name') and result.module_name == module_name:
                return {
                    'role': self._extract_role(result),
                    'confidence': self._extract_confidence(result),
                    'layer': self._extract_layer(result)
                }

        return {'role': 'UNKNOWN', 'confidence': 0.5, 'layer': 'UNKNOWN'}

    def _extract_role(self, result) -> str:
        if hasattr(result, 'final_role'):
            role = result.final_role
            return role.value if hasattr(role, 'value') else str(role)
        if isinstance(result, dict):
            role = result.get('final_role', result.get('role', 'UNKNOWN'))
            return role.value if hasattr(role, 'value') else str(role)
        return 'UNKNOWN'

    def _extract_confidence(self, result) -> float:
        if hasattr(result, 'final_confidence'):
            return result.final_confidence
        if isinstance(result, dict):
            return result.get('final_confidence', result.get('confidence', 0.5))
        return 0.5

    def _extract_layer(self, result) -> str:
        if hasattr(result, 'architectural_layer'):
            layer = result.architectural_layer
            return layer.value if hasattr(layer, 'value') else str(layer)
        if isinstance(result, dict):
            layer = result.get('architectural_layer', 'UNKNOWN')
            return layer.value if hasattr(layer, 'value') else str(layer)
        return 'UNKNOWN'

    def _build_tooltip(self, module_name: str, role: str, confidence: float,
                       layer: str, in_deg: int, out_deg: int) -> str:
        return f"""<div style="font-family: Arial; padding: 8px;">
<b>{module_name}</b><br/>
<hr style="margin: 5px 0;"/>
Role: {role}<br/>
Confidence: {confidence:.2f}<br/>
Layer: {layer}<br/>
<hr style="margin: 5px 0;"/>
In-degree: {in_deg}<br/>
Out-degree: {out_deg}
</div>"""

    def _inject_legend(self, html_path: str, title: str):
        """向 HTML 文件注入图例和标题"""
        # 角色图例
        role_items = []
        for role, color in ROLE_COLORS.items():
            role_items.append(
                f'<div style="display:flex;align-items:center;margin:2px 0;">'
                f'<span style="width:12px;height:12px;background:{color};'
                f'border-radius:50%;display:inline-block;margin-right:6px;"></span>'
                f'<span>{role}</span></div>'
            )

        # 边类型图例
        edge_items = [
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="width:20px;height:4px;background:{COLOR_INHERITANCE};'
            f'display:inline-block;margin-right:6px;"></span>'
            f'<span>Inheritance</span></div>',

            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="width:20px;height:2px;background:{COLOR_TYPE_HINT};'
            f'display:inline-block;margin-right:6px;border-style:dashed;"></span>'
            f'<span>Type Hint</span></div>',

            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="width:20px;height:1px;background:{COLOR_IMPORT};'
            f'display:inline-block;margin-right:6px;"></span>'
            f'<span>Import</span></div>',
        ]

        legend_html = f'''
        <div id="raacs-legend" style="
            position:fixed; top:10px; right:10px;
            background:rgba(30,30,50,0.95); padding:12px;
            border-radius:8px; color:white;
            font-family:Arial,sans-serif; font-size:11px;
            z-index:1000; box-shadow:0 4px 6px rgba(0,0,0,0.3);
        ">
            <div style="font-weight:bold;margin-bottom:8px;font-size:13px;">Roles</div>
            {''.join(role_items)}
            <hr style="margin:8px 0;border-color:#555;"/>
            <div style="font-weight:bold;margin-bottom:8px;font-size:13px;">Edges</div>
            {''.join(edge_items)}
        </div>
        '''

        title_html = f'''
        <div style="
            position:fixed; top:10px; left:10px;
            background:rgba(30,30,50,0.95); padding:10px 20px;
            border-radius:8px; color:white;
            font-family:Arial,sans-serif; font-size:18px; font-weight:bold;
            z-index:1000; box-shadow:0 4px 6px rgba(0,0,0,0.3);
        ">{title}</div>
        '''

        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace('</body>', f'{legend_html}{title_html}</body>')
        content = content.replace('<title>Network</title>', f'<title>{title}</title>')

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)


class PPRGraphVisualizer:
    """
    PPR 分数可视化器。

    节点颜色 = PPR 分数热力图
    节点大小 = PPR 分数
    目标节点 = 绿色星星
    边颜色 = 依赖类型
    """

    def __init__(self, dep_map: Dict[str, Dict]):
        """
        Args:
            dep_map: 依赖图字典 (来自 StaticImportScanner)
        """
        self.dep_map = dep_map

    def generate_html(self, target_node: str, ppr_scores: List[Tuple[str, float]],
                      output_path: str = "ppr_graph.html",
                      title: str = "PPR Context Window") -> str:
        """
        生成 PPR 分数可视化。

        Args:
            target_node: 目标节点（中心）
            ppr_scores: PPR 分数列表 [(node, score), ...]
            output_path: 输出文件路径
            title: 图表标题
        """
        Network = _get_pyvis()
        cm, mcolors = _get_matplotlib_colors()

        # 创建网络图
        net = Network(
            height="900px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            directed=True,
            select_menu=True,
            filter_menu=True,
            cdn_resources='in_line'
        )

        # 准备子图数据
        top_nodes = {node for node, score in ppr_scores}
        top_nodes.add(target_node)

        # 分数映射
        max_score = ppr_scores[0][1] if ppr_scores else 1.0
        score_map = {node: score for node, score in ppr_scores}
        score_map[target_node] = max_score * 1.2

        # 颜色映射
        cmap = cm.get_cmap('plasma')

        # === 添加节点 ===
        for node in top_nodes:
            if node not in self.dep_map:
                continue

            score = score_map.get(node, 0.0)

            if node == target_node:
                # 目标节点特殊样式
                color = "#00FF00"  # 荧光绿
                shape = "star"
                size = 50
                label = f"🎯 {node}"
                tooltip = "Target Context Window Center"
            else:
                # 普通节点根据分数变色
                ratio = score / max_score if max_score > 0 else 0
                rgba = cmap(ratio)
                color = mcolors.to_hex(rgba)
                shape = "dot"
                size = 10 + (ratio * 30)
                label = node.split('.')[-1]
                tooltip = f"{node}\nPPR Score: {score:.4f}"

            net.add_node(
                node,
                label=label,
                title=tooltip,
                color=color,
                size=size,
                shape=shape,
                borderWidth=1,
                borderWidthSelected=3,
                font={'size': 14, 'face': 'arial', 'color': 'white'}
            )

        # === 添加边（带权重着色） ===
        for module_name in top_nodes:
            if module_name not in self.dep_map:
                continue

            data = self.dep_map[module_name]
            imports = data.get('imports', [])
            source_path = data.get('path', '')

            # 只添加子图内部的边
            internal_imports = [imp for imp in imports if imp in top_nodes]

            if internal_imports:
                weights_map = _analyze_ast_weight(source_path, internal_imports)

                for imported in internal_imports:
                    weight = weights_map.get(imported, WEIGHT_IMPORT)

                    if weight >= WEIGHT_INHERITANCE:
                        color = COLOR_INHERITANCE
                        width = 4
                        dashes = False
                        edge_title = f"Inherits (w={weight})"
                    elif weight >= WEIGHT_TYPE_HINT:
                        color = COLOR_TYPE_HINT
                        width = 2
                        dashes = True
                        edge_title = f"Type Hint (w={weight})"
                    else:
                        color = COLOR_IMPORT
                        width = 1
                        dashes = False
                        edge_title = f"Import (w={weight})"

                    net.add_edge(
                        module_name,
                        imported,
                        color=color,
                        width=width,
                        dashes=dashes,
                        title=edge_title,
                        arrows={'to': {'enabled': True, 'scaleFactor': 1.0}}
                    )

        # 物理模拟
        net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=200)

        # 保存
        net.save_graph(output_path)

        # 注入标题
        self._inject_title(output_path, title, target_node)

        abs_path = os.path.abspath(output_path)
        print(f"[Viz] PPR visualization saved to: {abs_path}")
        return abs_path

    def _inject_title(self, html_path: str, title: str, target: str):
        """注入标题和图例"""
        edge_items = [
            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="width:20px;height:4px;background:{COLOR_INHERITANCE};'
            f'display:inline-block;margin-right:6px;"></span>'
            f'<span>Inheritance</span></div>',

            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="width:20px;height:2px;background:{COLOR_TYPE_HINT};'
            f'display:inline-block;margin-right:6px;border-style:dashed;"></span>'
            f'<span>Type Hint</span></div>',

            f'<div style="display:flex;align-items:center;margin:2px 0;">'
            f'<span style="width:20px;height:1px;background:{COLOR_IMPORT};'
            f'display:inline-block;margin-right:6px;"></span>'
            f'<span>Import</span></div>',
        ]

        legend_html = f'''
        <div style="
            position:fixed; top:10px; right:10px;
            background:rgba(30,30,50,0.95); padding:12px;
            border-radius:8px; color:white;
            font-family:Arial,sans-serif; font-size:11px;
            z-index:1000; box-shadow:0 4px 6px rgba(0,0,0,0.3);
        ">
            <div style="font-weight:bold;margin-bottom:8px;font-size:13px;">Edge Types</div>
            {''.join(edge_items)}
            <hr style="margin:8px 0;border-color:#555;"/>
            <div style="font-size:10px;color:#aaa;">
                Node color = PPR score (plasma colormap)<br/>
                Node size = PPR score<br/>
                🎯 = Target node
            </div>
        </div>
        '''

        title_html = f'''
        <div style="
            position:fixed; top:10px; left:10px;
            background:rgba(30,30,50,0.95); padding:10px 20px;
            border-radius:8px; color:white;
            font-family:Arial,sans-serif; font-size:16px; font-weight:bold;
            z-index:1000; box-shadow:0 4px 6px rgba(0,0,0,0.3);
        ">
            {title}<br/>
            <span style="font-size:12px;font-weight:normal;color:#aaa;">Target: {target}</span>
        </div>
        '''

        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace('</body>', f'{legend_html}{title_html}</body>')
        content = content.replace('<title>Network</title>', f'<title>{title}</title>')

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)


def generate_role_viz(project_root: str,
                      output_path: str = "dependency_graph.html",
                      role_results: Optional[Dict] = None,
                      debug: bool = False,
                      max_nodes: Optional[int] = None,
                      min_degree: int = 0,
                      exclude_roles: Optional[List[str]] = None,
                      layout: str = "hierarchical") -> Optional[str]:
    """
    便捷函数：从项目生成角色感知的依赖图可视化。
    
    Args:
        project_root: 项目根目录
        output_path: 输出 HTML 文件路径
        role_results: 角色分析结果字典（可选）
        debug: 调试模式
        max_nodes: 最大节点数（按度数重要性排序选取）
        min_degree: 最小度数阈值
        exclude_roles: 排除的角色列表
        layout: 布局方式 - "hierarchical"(层次), "physics"(力导向), "role"(按角色分组)
    """
    from raacs.adapters.import_scanner import StaticImportScanner

    try:
        if debug:
            print("[Viz] Scanning imports...")
        scanner = StaticImportScanner(project_root, debug=debug)
        dep_map = scanner.scan()

        if not dep_map:
            print("[Viz] Warning: Empty dependency map")
            return None

        if debug:
            print(f"[Viz] Found {len(dep_map)} modules")

        visualizer = RoleGraphVisualizer(dep_map, role_results)
        return visualizer.generate_html(
            output_path,
            max_nodes=max_nodes,
            min_degree=min_degree,
            exclude_roles=exclude_roles,
            layout=layout
        )

    except ImportError as e:
        print(f"[Viz] Missing dependency: {e}")
        print("[Viz] Install with: pip install pyvis matplotlib")
        return None
    except Exception as e:
        print(f"[Viz] Error generating visualization: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return None


__all__ = [
    "RoleGraphVisualizer",
    "PPRGraphVisualizer",
    "generate_role_viz",
    "ROLE_COLORS",
    "WEIGHT_INHERITANCE",
    "WEIGHT_TYPE_HINT",
    "WEIGHT_IMPORT",
    "COLOR_INHERITANCE",
    "COLOR_TYPE_HINT",
    "COLOR_IMPORT",
]
