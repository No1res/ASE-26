"""
图结构角色分析器 (Graph Role Analyzer) v9.0

核心洞察：很多架构角色是"在网络中的位置"定义的，而不是"自己长什么样"定义的。

设计思路：
1. 从依赖图中提取图结构特征（入度、出度、中心性）
2. 分析调用者/被调用者的角色分布
3. 推断图结构角色（HUB、ORCHESTRATOR、BRIDGE、LEAF）
4. 与 AST 角色融合，得到最终的"架构角色"

v9.0 改进：
- 动态阈值系统：基于仓库统计分布自动计算判定阈值
- 消除硬编码常量，提高跨仓库鲁棒性
"""

import json
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple
import math


# ============================================================================
# 动态阈值系统
# ============================================================================

@dataclass
class RepositoryStats:
    """仓库统计信息"""
    module_count: int = 0
    
    # 入度统计
    in_degree_min: int = 0
    in_degree_max: int = 0
    in_degree_mean: float = 0.0
    in_degree_median: float = 0.0
    in_degree_std: float = 0.0
    in_degree_p75: float = 0.0   # 75th percentile
    in_degree_p90: float = 0.0   # 90th percentile
    in_degree_p95: float = 0.0   # 95th percentile
    
    # 出度统计
    out_degree_min: int = 0
    out_degree_max: int = 0
    out_degree_mean: float = 0.0
    out_degree_median: float = 0.0
    out_degree_std: float = 0.0
    out_degree_p75: float = 0.0
    out_degree_p90: float = 0.0
    out_degree_p95: float = 0.0
    
    # 仓库规模分类
    @property
    def size_category(self) -> str:
        """仓库规模分类"""
        if self.module_count < 30:
            return "tiny"      # 微型
        elif self.module_count < 100:
            return "small"     # 小型
        elif self.module_count < 300:
            return "medium"    # 中型
        elif self.module_count < 1000:
            return "large"     # 大型
        else:
            return "huge"      # 超大型


@dataclass
class DynamicThresholds:
    """
    动态阈值配置
    
    所有阈值基于仓库统计分布动态计算，而非硬编码。
    设计原则：
    1. 使用百分位数而非绝对值
    2. 根据仓库规模调整敏感度
    3. 保证边界情况的合理性
    """
    
    # === 图角色判定阈值 ===
    
    # HUB: 高入度模块
    hub_in_degree_threshold: float = 0.0       # 入度 >= 此值判定为 HUB
    hub_in_degree_percentile: float = 90.0     # 使用的百分位数
    
    # ORCHESTRATOR: 高出度模块  
    orchestrator_out_degree_threshold: float = 0.0
    orchestrator_out_degree_percentile: float = 90.0
    
    # 备选条件（当 max 值异常时使用）
    hub_fallback_multiplier: float = 2.0       # mean * multiplier
    hub_fallback_min: int = 3                  # 最小绝对值
    orchestrator_fallback_multiplier: float = 2.0
    orchestrator_fallback_min: int = 3
    
    # === 架构层次推断阈值 ===
    
    # 基础设施层: 高入度 + 低出度
    infra_in_degree_threshold: float = 0.0     # 入度 >= P75
    infra_out_degree_threshold: float = 0.0    # 出度 <= P25
    
    # 应用层: 主要被 ADAPTER/TEST 调用
    app_layer_caller_ratio: float = 0.6        # 默认 60%（根据规模调整）
    
    # 领域层: 高出度 + 调用 SCHEMA/UTIL
    domain_out_degree_threshold: float = 0.0   # 出度 >= P50
    domain_callee_ratio: float = 0.3           # 调用的 SCHEMA/UTIL 占比
    
    # === BRIDGE 判定阈值 ===
    bridge_min_caller_roles: int = 2           # 调用者最少角色数
    bridge_min_callee_roles: int = 2           # 被调用者最少角色数
    bridge_jaccard_threshold: float = 0.5      # Jaccard 相似度阈值
    
    # === 调整建议阈值 ===
    high_centrality_threshold: float = 0.0     # 高中心性阈值 (入度 >= P80)
    
    @classmethod
    def from_stats(cls, stats: RepositoryStats) -> 'DynamicThresholds':
        """
        根据仓库统计信息计算动态阈值
        
        核心思想：
        1. 使用百分位数而非绝对值，适应不同规模
        2. 根据仓库规模调整敏感度
        3. 设置合理的下限，避免小仓库判定失效
        """
        thresholds = cls()
        
        # 根据仓库规模调整百分位数
        size = stats.size_category
        if size == "tiny":
            # 微型仓库：降低阈值，否则可能没有 HUB
            hub_percentile = 80.0
            orch_percentile = 80.0
            app_ratio = 0.5
        elif size == "small":
            hub_percentile = 85.0
            orch_percentile = 85.0
            app_ratio = 0.55
        elif size == "medium":
            hub_percentile = 90.0
            orch_percentile = 90.0
            app_ratio = 0.6
        elif size == "large":
            hub_percentile = 92.0
            orch_percentile = 92.0
            app_ratio = 0.65
        else:  # huge
            hub_percentile = 95.0
            orch_percentile = 95.0
            app_ratio = 0.7
        
        # HUB 阈值
        thresholds.hub_in_degree_percentile = hub_percentile
        thresholds.hub_in_degree_threshold = max(
            stats.in_degree_p90 if hub_percentile >= 90 else stats.in_degree_p75,
            stats.in_degree_mean + stats.in_degree_std,  # 至少高于 mean+1std
            2  # 最小值
        )
        
        # ORCHESTRATOR 阈值
        thresholds.orchestrator_out_degree_percentile = orch_percentile
        thresholds.orchestrator_out_degree_threshold = max(
            stats.out_degree_p90 if orch_percentile >= 90 else stats.out_degree_p75,
            stats.out_degree_mean + stats.out_degree_std,
            2
        )
        
        # 备选条件的最小值
        thresholds.hub_fallback_min = max(2, int(stats.in_degree_mean))
        thresholds.orchestrator_fallback_min = max(2, int(stats.out_degree_mean))
        
        # 基础设施层阈值
        thresholds.infra_in_degree_threshold = max(stats.in_degree_p75, 3)
        thresholds.infra_out_degree_threshold = max(stats.out_degree_median, 1)
        
        # 应用层比例
        thresholds.app_layer_caller_ratio = app_ratio
        
        # 领域层阈值
        thresholds.domain_out_degree_threshold = max(stats.out_degree_median, 2)
        
        # 高中心性阈值
        thresholds.high_centrality_threshold = max(
            stats.in_degree_p75,
            stats.in_degree_mean + stats.in_degree_std,
            3
        )
        
        return thresholds
    
    def __str__(self) -> str:
        return (
            f"DynamicThresholds(\n"
            f"  HUB: in_degree >= {self.hub_in_degree_threshold:.1f} (P{self.hub_in_degree_percentile})\n"
            f"  ORCHESTRATOR: out_degree >= {self.orchestrator_out_degree_threshold:.1f} (P{self.orchestrator_out_degree_percentile})\n"
            f"  INFRA: in >= {self.infra_in_degree_threshold:.1f}, out <= {self.infra_out_degree_threshold:.1f}\n"
            f"  APP: caller_ratio >= {self.app_layer_caller_ratio:.2f}\n"
            f"  HIGH_CENTRALITY: >= {self.high_centrality_threshold:.1f}\n"
            f")"
        )


def compute_repository_stats(in_degrees: List[int], out_degrees: List[int]) -> RepositoryStats:
    """
    计算仓库统计信息
    
    Args:
        in_degrees: 所有模块的入度列表
        out_degrees: 所有模块的出度列表
    """
    stats = RepositoryStats()
    
    if not in_degrees or not out_degrees:
        return stats
    
    stats.module_count = len(in_degrees)
    
    # 入度统计
    stats.in_degree_min = min(in_degrees)
    stats.in_degree_max = max(in_degrees)
    stats.in_degree_mean = statistics.mean(in_degrees)
    stats.in_degree_median = statistics.median(in_degrees)
    stats.in_degree_std = statistics.stdev(in_degrees) if len(in_degrees) > 1 else 0
    
    sorted_in = sorted(in_degrees)
    n = len(sorted_in)
    stats.in_degree_p75 = sorted_in[int(n * 0.75)] if n > 0 else 0
    stats.in_degree_p90 = sorted_in[int(n * 0.90)] if n > 0 else 0
    stats.in_degree_p95 = sorted_in[int(n * 0.95)] if n > 0 else 0
    
    # 出度统计
    stats.out_degree_min = min(out_degrees)
    stats.out_degree_max = max(out_degrees)
    stats.out_degree_mean = statistics.mean(out_degrees)
    stats.out_degree_median = statistics.median(out_degrees)
    stats.out_degree_std = statistics.stdev(out_degrees) if len(out_degrees) > 1 else 0
    
    sorted_out = sorted(out_degrees)
    stats.out_degree_p75 = sorted_out[int(n * 0.75)] if n > 0 else 0
    stats.out_degree_p90 = sorted_out[int(n * 0.90)] if n > 0 else 0
    stats.out_degree_p95 = sorted_out[int(n * 0.95)] if n > 0 else 0
    
    return stats


class GraphRole(Enum):
    """图结构角色"""
    HUB = "HUB"                     # 被广泛依赖的核心模块
    ORCHESTRATOR = "ORCHESTRATOR"   # 协调/聚合多个模块
    BRIDGE = "BRIDGE"               # 连接不同角色层的桥梁
    LEAF = "LEAF"                   # 叶子节点（只被调用）
    SINK = "SINK"                   # 汇点（只调用，不被调用）
    ISOLATE = "ISOLATE"             # 孤立节点
    NORMAL = "NORMAL"               # 普通节点


class ArchitecturalLayer(Enum):
    """架构层次（基于调用关系推断）"""
    INFRASTRUCTURE = "INFRASTRUCTURE"  # 基础设施层（被所有层使用）
    DOMAIN = "DOMAIN"                  # 领域层（核心业务逻辑）
    APPLICATION = "APPLICATION"        # 应用层（用例/服务）
    INTERFACE = "INTERFACE"            # 接口层（入口/适配器）
    UNKNOWN = "UNKNOWN"


@dataclass
class GraphFeatures:
    """图结构特征"""
    module_name: str
    file_path: str
    
    # 度量特征
    in_degree: int = 0              # 被导入次数
    out_degree: int = 0             # 导入次数
    internal_in_degree: int = 0     # 内部模块导入次数
    internal_out_degree: int = 0    # 导入内部模块次数
    external_in_degree: int = 0     # 被外部模块导入次数
    external_out_degree: int = 0    # 导入外部模块次数
    
    # 调用者/被调用者
    callers: List[str] = field(default_factory=list)    # 谁调用它
    callees: List[str] = field(default_factory=list)    # 它调用谁
    
    # AST角色分布
    caller_roles: Dict[str, int] = field(default_factory=dict)   # 调用者的角色分布
    callee_roles: Dict[str, int] = field(default_factory=dict)   # 被调用者的角色分布
    
    # 中心性度量
    betweenness_centrality: float = 0.0
    pagerank: float = 0.0
    
    # 距离特征
    bacon_number: int = 0  # 到入口点的距离


@dataclass
class GraphRoleResult:
    """图结构角色分析结果"""
    module_name: str
    file_path: str
    
    # 图结构角色
    graph_role: GraphRole = GraphRole.NORMAL
    graph_role_confidence: float = 0.0
    graph_role_reasoning: str = ""
    
    # 架构层次推断
    architectural_layer: ArchitecturalLayer = ArchitecturalLayer.UNKNOWN
    layer_confidence: float = 0.0
    
    # 原始特征
    features: Optional[GraphFeatures] = None
    
    # 角色调整建议
    role_adjustment_suggestion: Optional[str] = None


class DependencyGraphAnalyzer:
    """
    依赖图分析器
    
    使用动态阈值系统，根据仓库统计分布自动计算判定阈值。
    """
    
    @staticmethod
    def build_from_symbol_table(symbol_table) -> Dict:
        """
        Build dependency graph directly from AST symbol table (Replacing pydeps).
        
        Args:
            symbol_table: ProjectSymbolTable from ast_analyzer
            
        Returns:
            dep_map dictionary compatible with DependencyGraphAnalyzer
        """
        dep_map = {}
        
        # 1. Build module_path -> file_path lookup
        mod_to_file = {}
        for file_path, file_symbols in symbol_table.files.items():
            if file_symbols.module_path:
                mod_to_file[file_symbols.module_path] = file_path
        
        # 2. Initialize nodes
        for mod_name, file_path in mod_to_file.items():
            dep_map[mod_name] = {
                "path": file_path,
                "imports": [],
                "imported_by": [],
                "bacon": 0
            }
            
        # 3. Resolve edges (Imports)
        for file_path, file_symbols in symbol_table.files.items():
            source_mod = file_symbols.module_path
            if not source_mod or source_mod not in dep_map:
                continue
            
            seen_imports = set()
            
            # Iterate through imports collected by AST
            for import_info in file_symbols.imports.values():
                target = import_info.source_module
                if not target:
                    continue
                
                resolved_target = None
                
                # Case A: Exact match
                if target in mod_to_file:
                    resolved_target = target
                # Case B: Parent package match (e.g., from raacs.utils import x -> depends on raacs.utils)
                else:
                    parts = target.split('.')
                    for i in range(len(parts), 0, -1):
                        sub_mod = '.'.join(parts[:i])
                        if sub_mod in mod_to_file:
                            resolved_target = sub_mod
                            break
                
                # Add edge
                if resolved_target and resolved_target != source_mod:
                    if resolved_target not in seen_imports:
                        dep_map[source_mod]["imports"].append(resolved_target)
                        seen_imports.add(resolved_target)

        # 4. Calculate reverse edges (imported_by)
        for source, data in dep_map.items():
            for target in data["imports"]:
                if target in dep_map:
                    if source not in dep_map[target]["imported_by"]:
                        dep_map[target]["imported_by"].append(source)
                        
        return dep_map
    
    def __init__(self, dep_map: Dict, project_root: str = None, debug: bool = False):
        """
        Args:
            dep_map: pydeps 生成的依赖映射
            project_root: 项目根路径（用于过滤内部模块）
            debug: 调试模式
        """
        self.dep_map = dep_map
        self.project_root = project_root
        self.debug = debug
        
        # 识别内部模块
        self.internal_modules = self._identify_internal_modules()
        
        # 缓存
        self._features_cache: Dict[str, GraphFeatures] = {}
        self._ast_roles: Dict[str, str] = {}  # module_name -> AST role
        
        # 动态阈值（延迟初始化）
        self._stats: Optional[RepositoryStats] = None
        self._thresholds: Optional[DynamicThresholds] = None
    
    def _identify_internal_modules(self) -> Set[str]:
        """识别项目内部模块"""
        internal = set()
        for name, info in self.dep_map.items():
            path = info.get('path')
            if path:
                if self.project_root and path.startswith(self.project_root):
                    internal.add(name)
                elif 'site-packages' not in path and not path.startswith('/usr/'):
                    internal.add(name)
        return internal
    
    def set_ast_roles(self, ast_roles: Dict[str, str]):
        """设置 AST 角色（用于调用者/被调用者角色分析）"""
        self._ast_roles = ast_roles
    
    def _ensure_thresholds(self):
        """确保动态阈值已计算"""
        if self._thresholds is not None:
            return
        
        # 收集所有模块的入度和出度
        in_degrees = []
        out_degrees = []
        
        for module_name in self.internal_modules:
            features = self.extract_features(module_name)
            if features:
                in_degrees.append(features.internal_in_degree)
                out_degrees.append(features.internal_out_degree)
        
        # 计算统计信息
        self._stats = compute_repository_stats(in_degrees, out_degrees)
        
        # 计算动态阈值
        self._thresholds = DynamicThresholds.from_stats(self._stats)
        
        if self.debug:
            print(f"\n[DynamicThresholds] Repository stats:")
            print(f"  Modules: {self._stats.module_count} ({self._stats.size_category})")
            print(f"  In-degree: min={self._stats.in_degree_min}, max={self._stats.in_degree_max}, "
                  f"mean={self._stats.in_degree_mean:.1f}, median={self._stats.in_degree_median:.1f}, "
                  f"P90={self._stats.in_degree_p90:.1f}")
            print(f"  Out-degree: min={self._stats.out_degree_min}, max={self._stats.out_degree_max}, "
                  f"mean={self._stats.out_degree_mean:.1f}, median={self._stats.out_degree_median:.1f}, "
                  f"P90={self._stats.out_degree_p90:.1f}")
            print(f"\n{self._thresholds}")
    
    def extract_features(self, module_name: str) -> Optional[GraphFeatures]:
        """提取模块的图结构特征"""
        if module_name in self._features_cache:
            return self._features_cache[module_name]
        
        if module_name not in self.dep_map:
            return None
        
        info = self.dep_map[module_name]
        
        # 基本信息
        features = GraphFeatures(
            module_name=module_name,
            file_path=info.get('path', ''),
            bacon_number=info.get('bacon', 0)
        )
        
        # 计算入度和出度
        imported_by = info.get('imported_by', [])
        imports = info.get('imports', [])
        
        features.in_degree = len(imported_by)
        features.out_degree = len(imports)
        
        # 区分内部/外部
        for caller in imported_by:
            if caller in self.internal_modules:
                features.internal_in_degree += 1
                features.callers.append(caller)
            else:
                features.external_in_degree += 1
        
        for callee in imports:
            if callee in self.internal_modules:
                features.internal_out_degree += 1
                features.callees.append(callee)
            else:
                features.external_out_degree += 1
        
        # 调用者/被调用者的角色分布
        if self._ast_roles:
            for caller in features.callers:
                role = self._ast_roles.get(caller, 'UNKNOWN')
                features.caller_roles[role] = features.caller_roles.get(role, 0) + 1
            
            for callee in features.callees:
                role = self._ast_roles.get(callee, 'UNKNOWN')
                features.callee_roles[role] = features.callee_roles.get(role, 0) + 1
        
        self._features_cache[module_name] = features
        return features
    
    def analyze_graph_role(self, module_name: str) -> GraphRoleResult:
        """
        分析模块的图结构角色
        
        使用动态阈值系统，消除硬编码常量。
        """
        features = self.extract_features(module_name)
        
        if not features:
            return GraphRoleResult(
                module_name=module_name,
                file_path="",
                graph_role=GraphRole.ISOLATE,
                graph_role_confidence=1.0,
                graph_role_reasoning="Module not found in dependency map"
            )
        
        # 确保动态阈值已计算
        self._ensure_thresholds()
        
        result = GraphRoleResult(
            module_name=module_name,
            file_path=features.file_path,
            features=features
        )
        
        if not self._stats or not self._thresholds:
            return result
        
        in_deg = features.internal_in_degree
        out_deg = features.internal_out_degree
        
        # 获取动态阈值
        t = self._thresholds
        s = self._stats
        
        # === 图结构角色判定（使用动态阈值） ===
        
        # 1. HUB: 高入度，被广泛依赖
        # 条件：入度 >= 动态阈值 或 入度 > mean + std 且 >= 最小值
        hub_threshold = t.hub_in_degree_threshold
        hub_fallback = s.in_degree_mean + t.hub_fallback_multiplier * s.in_degree_std
        hub_min = t.hub_fallback_min
        
        if in_deg >= hub_threshold or (in_deg >= hub_fallback and in_deg >= hub_min):
            result.graph_role = GraphRole.HUB
            result.graph_role_confidence = min(in_deg / (s.in_degree_max or 1), 1.0)
            result.graph_role_reasoning = (
                f"High in-degree ({in_deg} >= threshold {hub_threshold:.1f}), "
                f"widely depended upon"
            )
        
        # 2. ORCHESTRATOR: 高出度，聚合/协调多个模块
        elif out_deg >= t.orchestrator_out_degree_threshold or \
             (out_deg >= s.out_degree_mean + t.orchestrator_fallback_multiplier * s.out_degree_std and \
              out_deg >= t.orchestrator_fallback_min):
            result.graph_role = GraphRole.ORCHESTRATOR
            result.graph_role_confidence = min(out_deg / (s.out_degree_max or 1), 1.0)
            result.graph_role_reasoning = (
                f"High out-degree ({out_deg} >= threshold {t.orchestrator_out_degree_threshold:.1f}), "
                f"coordinates multiple modules"
            )
        
        # 3. LEAF: 只被调用，不调用内部模块（确定性判断，无需动态阈值）
        elif in_deg > 0 and out_deg == 0:
            result.graph_role = GraphRole.LEAF
            result.graph_role_confidence = 0.9
            result.graph_role_reasoning = f"Leaf node (in={in_deg}, out=0)"
        
        # 4. SINK: 只调用，不被调用（确定性判断）
        elif in_deg == 0 and out_deg > 0:
            result.graph_role = GraphRole.SINK
            result.graph_role_confidence = 0.9
            result.graph_role_reasoning = f"Sink node (in=0, out={out_deg})"
        
        # 5. BRIDGE: 连接不同角色层的桥梁
        elif self._is_bridge(features, t):
            result.graph_role = GraphRole.BRIDGE
            result.graph_role_confidence = 0.7
            result.graph_role_reasoning = "Connects different role layers"
        
        # 6. ISOLATE: 孤立节点（确定性判断）
        elif in_deg == 0 and out_deg == 0:
            result.graph_role = GraphRole.ISOLATE
            result.graph_role_confidence = 1.0
            result.graph_role_reasoning = "Isolated node"
        
        # 7. NORMAL: 普通节点
        else:
            result.graph_role = GraphRole.NORMAL
            result.graph_role_confidence = 0.5
            result.graph_role_reasoning = f"Normal node (in={in_deg}, out={out_deg})"
        
        # === 架构层次推断 ===
        result.architectural_layer, result.layer_confidence = self._infer_architectural_layer(features, t, s)
        
        # === 角色调整建议 ===
        result.role_adjustment_suggestion = self._suggest_role_adjustment(features, result, t)
        
        return result
    
    def _is_bridge(self, features: GraphFeatures, thresholds: DynamicThresholds) -> bool:
        """
        判断是否为桥梁节点
        
        使用动态阈值中的 bridge 参数。
        """
        if not features.caller_roles or not features.callee_roles:
            return False
        
        # 如果调用者和被调用者的角色分布差异大，可能是桥梁
        caller_role_set = set(features.caller_roles.keys())
        callee_role_set = set(features.callee_roles.keys())
        
        # 移除 UNKNOWN
        caller_role_set.discard('UNKNOWN')
        callee_role_set.discard('UNKNOWN')
        
        # 使用动态阈值
        if (len(caller_role_set) >= thresholds.bridge_min_caller_roles and 
            len(callee_role_set) >= thresholds.bridge_min_callee_roles):
            # 角色分布差异
            intersection = caller_role_set & callee_role_set
            union = caller_role_set | callee_role_set
            jaccard = len(intersection) / len(union) if union else 0
            return jaccard < thresholds.bridge_jaccard_threshold
        
        return False
    
    def _infer_architectural_layer(self, features: GraphFeatures,
                                    thresholds: DynamicThresholds,
                                    stats: RepositoryStats) -> Tuple[ArchitecturalLayer, float]:
        """
        推断架构层次
        
        使用动态阈值，消除硬编码常量。
        """
        in_deg = features.internal_in_degree
        out_deg = features.internal_out_degree
        
        # 规则1: 高入度 + 低出度 → 基础设施层
        # 使用动态阈值：入度 >= P75，出度 <= median
        if in_deg >= thresholds.infra_in_degree_threshold and out_deg <= thresholds.infra_out_degree_threshold:
            return ArchitecturalLayer.INFRASTRUCTURE, 0.8
        
        # 规则2: 只被 ADAPTER/TEST 调用 → 应用层
        if features.caller_roles:
            total_callers = sum(features.caller_roles.values())
            if total_callers > 0:
                adapter_test_count = (
                    features.caller_roles.get('ADAPTER', 0) + 
                    features.caller_roles.get('TEST', 0)
                )
                adapter_test_ratio = adapter_test_count / total_callers
                if adapter_test_ratio >= thresholds.app_layer_caller_ratio:
                    return ArchitecturalLayer.APPLICATION, 0.7
        
        # 规则3: 高出度 + 调用 SCHEMA/UTIL → 领域层
        if out_deg >= thresholds.domain_out_degree_threshold and features.callee_roles:
            total_callees = sum(features.callee_roles.values())
            if total_callees > 0:
                domain_deps = (
                    features.callee_roles.get('SCHEMA', 0) +
                    features.callee_roles.get('UTIL', 0)
                )
                if domain_deps / total_callees >= thresholds.domain_callee_ratio:
                    return ArchitecturalLayer.DOMAIN, 0.6
        
        # 规则4: bacon_number = 1 或 0 → 接口层
        # 这是确定性判断，不需要动态阈值
        if features.bacon_number <= 1:
            return ArchitecturalLayer.INTERFACE, 0.5
        
        return ArchitecturalLayer.UNKNOWN, 0.0
    
    def _suggest_role_adjustment(self, features: GraphFeatures, 
                                  result: GraphRoleResult,
                                  thresholds: DynamicThresholds) -> Optional[str]:
        """
        建议角色调整
        
        使用动态阈值判断"高入度"。
        """
        ast_role = self._ast_roles.get(features.module_name, 'UNKNOWN')
        
        # 高入度 + AST角色是 LOGIC → 可能是核心 UTIL
        if result.graph_role == GraphRole.HUB and ast_role == 'LOGIC':
            if features.internal_in_degree >= thresholds.high_centrality_threshold:
                return (f"High hub centrality (in={features.internal_in_degree} >= "
                        f"{thresholds.high_centrality_threshold:.0f}) suggests this might be core UTIL")
        
        # SINK + AST角色不是 SCRIPT/TEST → 可能是入口
        if result.graph_role == GraphRole.SINK and ast_role not in ('SCRIPT', 'TEST'):
            return "Sink node (only calls, never called) suggests this might be an entry point/SCRIPT"
        
        # LEAF + AST角色是 LOGIC → 可能是 UTIL 或 CONFIG
        if result.graph_role == GraphRole.LEAF and ast_role == 'LOGIC':
            return "Leaf node (never calls others) might be a stateless UTIL or CONFIG"
        
        return None
    
    def compute_all_features(self) -> Dict[str, GraphFeatures]:
        """计算所有内部模块的特征"""
        return {m: self.extract_features(m) for m in self.internal_modules if self.extract_features(m)}
    
    def analyze_all(self) -> Dict[str, GraphRoleResult]:
        """分析所有内部模块"""
        return {m: self.analyze_graph_role(m) for m in self.internal_modules}
    
    def get_role_distribution_summary(self) -> Dict[str, Dict[str, int]]:
        """获取每个模块的调用者角色分布摘要"""
        summary = {}
        for name in self.internal_modules:
            features = self.extract_features(name)
            if features and features.caller_roles:
                summary[name] = features.caller_roles
        return summary


# ============================================================================
# 融合 AST 角色与图结构角色
# ============================================================================

@dataclass
class FusedRoleResult:
    """融合后的角色结果"""
    module_name: str
    file_path: str
    
    # AST 角色（原始）
    ast_role: str
    ast_confidence: float
    
    # 图结构角色
    graph_role: GraphRole
    graph_confidence: float
    
    # 架构层次
    architectural_layer: ArchitecturalLayer
    
    # 融合后的角色
    fused_role: str
    fused_confidence: float
    
    # 调整说明
    adjustment_reason: str = ""
    
    # 详细特征
    in_degree: int = 0
    out_degree: int = 0
    caller_roles: Dict[str, int] = field(default_factory=dict)


class RoleFusionEngine:
    """角色融合引擎"""
    
    # 融合规则：(AST角色, 图角色) -> 融合角色
    FUSION_RULES = {
        # HUB 节点
        ('LOGIC', GraphRole.HUB): ('UTIL', 0.8, "High centrality hub suggests core utility"),
        ('UNKNOWN', GraphRole.HUB): ('UTIL', 0.7, "Hub with unknown AST role"),
        
        # ORCHESTRATOR 节点
        ('LOGIC', GraphRole.ORCHESTRATOR): ('LOGIC', 0.9, "Orchestrator confirms LOGIC role"),
        ('UTIL', GraphRole.ORCHESTRATOR): ('LOGIC', 0.7, "Orchestrator with many deps suggests LOGIC"),
        
        # SINK 节点
        ('LOGIC', GraphRole.SINK): ('SCRIPT', 0.6, "Sink node (entry point) suggests SCRIPT"),
        ('ADAPTER', GraphRole.SINK): ('ADAPTER', 0.9, "Sink confirms ADAPTER as entry"),
        
        # LEAF 节点
        ('LOGIC', GraphRole.LEAF): ('UTIL', 0.6, "Leaf node (no deps) might be UTIL"),
    }
    
    def __init__(self, graph_analyzer: DependencyGraphAnalyzer):
        self.graph_analyzer = graph_analyzer
    
    def fuse(self, module_name: str, ast_role: str, ast_confidence: float) -> FusedRoleResult:
        """融合 AST 角色和图结构角色"""
        graph_result = self.graph_analyzer.analyze_graph_role(module_name)
        features = graph_result.features or GraphFeatures(module_name=module_name, file_path="")
        
        # 查找融合规则
        rule_key = (ast_role, graph_result.graph_role)
        
        if rule_key in self.FUSION_RULES:
            fused_role, confidence_boost, reason = self.FUSION_RULES[rule_key]
            fused_confidence = min(ast_confidence * confidence_boost + 0.1, 1.0)
        else:
            # 默认保持 AST 角色
            fused_role = ast_role
            fused_confidence = ast_confidence
            reason = ""
        
        # 考虑架构层次
        if graph_result.architectural_layer == ArchitecturalLayer.INFRASTRUCTURE:
            if fused_role == 'LOGIC':
                fused_role = 'UTIL'
                reason = f"Infrastructure layer suggests UTIL; {reason}"
        elif graph_result.architectural_layer == ArchitecturalLayer.INTERFACE:
            if fused_role == 'LOGIC':
                fused_role = 'ADAPTER'
                fused_confidence *= 0.7
                reason = f"Interface layer suggests ADAPTER; {reason}"
        
        return FusedRoleResult(
            module_name=module_name,
            file_path=features.file_path,
            ast_role=ast_role,
            ast_confidence=ast_confidence,
            graph_role=graph_result.graph_role,
            graph_confidence=graph_result.graph_role_confidence,
            architectural_layer=graph_result.architectural_layer,
            fused_role=fused_role,
            fused_confidence=fused_confidence,
            adjustment_reason=reason,
            in_degree=features.internal_in_degree,
            out_degree=features.internal_out_degree,
            caller_roles=features.caller_roles
        )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph Role Analyzer")
    parser.add_argument("dep_map", help="Path to pydeps JSON dependency map")
    parser.add_argument("--project-root", help="Project root path")
    parser.add_argument("--module", help="Analyze specific module")
    parser.add_argument("--top-hubs", type=int, default=10, help="Show top N hubs")
    parser.add_argument("--top-orchestrators", type=int, default=10, help="Show top N orchestrators")
    args = parser.parse_args()
    
    # 加载依赖图
    with open(args.dep_map) as f:
        dep_map = json.load(f)
    
    analyzer = DependencyGraphAnalyzer(dep_map, args.project_root)
    
    if args.module:
        result = analyzer.analyze_graph_role(args.module)
        print(f"\n{'='*60}")
        print(f"Module: {result.module_name}")
        print(f"{'='*60}")
        print(f"Graph Role: {result.graph_role.value} (conf={result.graph_role_confidence:.2f})")
        print(f"Reasoning: {result.graph_role_reasoning}")
        print(f"Architectural Layer: {result.architectural_layer.value}")
        if result.features:
            f = result.features
            print(f"\nFeatures:")
            print(f"  In-degree: {f.internal_in_degree} internal, {f.external_in_degree} external")
            print(f"  Out-degree: {f.internal_out_degree} internal, {f.external_out_degree} external")
            if f.caller_roles:
                print(f"  Caller roles: {f.caller_roles}")
            if f.callee_roles:
                print(f"  Callee roles: {f.callee_roles}")
        if result.role_adjustment_suggestion:
            print(f"\n⚠️  Suggestion: {result.role_adjustment_suggestion}")
    else:
        # 分析所有模块
        results = analyzer.analyze_all()
        
        # 按图角色分组
        by_role = defaultdict(list)
        for name, result in results.items():
            by_role[result.graph_role].append(result)
        
        print(f"\n{'='*70}")
        print("Graph Role Distribution")
        print(f"{'='*70}")
        for role in GraphRole:
            count = len(by_role[role])
            if count > 0:
                print(f"  {role.value:<15}: {count}")
        
        print(f"\n{'='*70}")
        print(f"Top {args.top_hubs} HUB Modules (most depended upon)")
        print(f"{'='*70}")
        hubs = sorted(by_role[GraphRole.HUB], key=lambda r: r.features.internal_in_degree if r.features else 0, reverse=True)
        for r in hubs[:args.top_hubs]:
            f = r.features
            in_deg = f.internal_in_degree if f else 0
            print(f"  {r.module_name:<50} in={in_deg}")
        
        print(f"\n{'='*70}")
        print(f"Top {args.top_orchestrators} ORCHESTRATOR Modules (most dependencies)")
        print(f"{'='*70}")
        orchs = sorted(by_role[GraphRole.ORCHESTRATOR], key=lambda r: r.features.internal_out_degree if r.features else 0, reverse=True)
        for r in orchs[:args.top_orchestrators]:
            f = r.features
            out_deg = f.internal_out_degree if f else 0
            print(f"  {r.module_name:<50} out={out_deg}")

