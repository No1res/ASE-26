__version__ = "9.3"

import os
import sys
import json
import argparse
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

# 导入 RAACS 核心库
from raacs import (
    # AST 分析器
    CodeRoleClassifier, Role, FileAnalysis,
    SymbolCollector, RolePropagator, ProjectSymbolTable,
    RoleSource,  # v9.2: 角色来源追踪
    # 图结构分析器
    DependencyGraphAnalyzer, GraphRole, ArchitecturalLayer,
    GraphRoleResult,
)

# 谱加权 PPR（可选，需要 numpy, scipy, networkx）
try:
    from raacs.spectral_ppr import SpectralAnalyzer, SpectralPPR
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False


@dataclass
class IntegratedRoleResult:
    """集成角色分析结果"""
    module_name: str
    file_path: str

    # 三层角色
    ast_role: Role
    ast_confidence: float
    ast_reasoning: str

    graph_role: GraphRole
    graph_confidence: float
    graph_reasoning: str

    architectural_layer: ArchitecturalLayer

    # 融合后的最终角色
    final_role: Role
    final_confidence: float
    fusion_reasoning: str

    # 实体级信息
    entities: List[dict] = field(default_factory=list)
    role_purity: float = 1.0

    # 图特征
    in_degree: int = 0
    out_degree: int = 0
    caller_roles: Dict[str, int] = field(default_factory=dict)

    # 继承信息
    inherited_from: str = ""

    # 谱分析特征 (v9.3)
    spectral_cluster: int = -1
    fiedler_value: float = 0.0


class IntegratedRoleAnalyzer:
    """
    集成角色分析器

    分析流程：
    1. 第一遍：AST 分析 + 符号表构建
    2. 第二遍：角色传播（跨文件继承）
    3. 第三遍：图结构分析
    4. 第四遍：谱分析（Laplacian + PPR）(v9.3)
    5. 融合：多层信号融合
    """
    
    # 角色融合规则
    # (AST角色, 图角色) -> (融合角色, 置信度调整, 说明)
    FUSION_RULES = {
        # HUB 节点：被广泛依赖
        (Role.LOGIC, GraphRole.HUB): (Role.UTIL, 0.7, "High centrality suggests core utility"),
        (Role.UNKNOWN, GraphRole.HUB): (Role.UTIL, 0.6, "Hub with unknown AST"),
        
        # ORCHESTRATOR 节点：协调多个模块
        (Role.LOGIC, GraphRole.ORCHESTRATOR): (Role.LOGIC, 1.0, "Orchestrator confirms business logic"),
        (Role.UTIL, GraphRole.ORCHESTRATOR): (Role.LOGIC, 0.8, "Orchestrator suggests coordination logic"),
        (Role.SCHEMA, GraphRole.ORCHESTRATOR): (Role.SCHEMA, 0.9, "Schema with many deps (aggregation)"),
        
        # SINK 节点：只调用，不被调用（入口点）
        (Role.LOGIC, GraphRole.SINK): (Role.SCRIPT, 0.5, "Sink node (entry point)"),
        (Role.ADAPTER, GraphRole.SINK): (Role.ADAPTER, 1.0, "Entry adapter"),
        
        # LEAF 节点：只被调用，不调用其他
        (Role.LOGIC, GraphRole.LEAF): (Role.UTIL, 0.6, "Leaf node (stateless utility)"),
        (Role.SCHEMA, GraphRole.LEAF): (Role.SCHEMA, 1.0, "Pure schema/entity"),
        (Role.CONFIG, GraphRole.LEAF): (Role.CONFIG, 1.0, "Pure config"),
        
        # BRIDGE 节点：连接不同层
        (Role.LOGIC, GraphRole.BRIDGE): (Role.ADAPTER, 0.6, "Bridge suggests adapter layer"),
    }
    
    def __init__(self, project_root: str, dep_map_path: Optional[str] = None,
                 debug: bool = False, enable_spectral: bool = True):
        """
        Args:
            project_root: 项目根目录
            dep_map_path: 预生成的依赖图 JSON 路径（可选）
            debug: 调试模式
            enable_spectral: 是否启用谱分析（需要 numpy, scipy, networkx）
        """
        self.project_root = os.path.abspath(project_root)
        self.debug = debug
        self.enable_spectral = enable_spectral and SPECTRAL_AVAILABLE

        # AST 分析器（带符号表）
        self.ast_classifier: Optional[CodeRoleClassifier] = None
        self.symbol_table: Optional[ProjectSymbolTable] = None

        # 图结构分析器
        self.graph_analyzer: Optional[DependencyGraphAnalyzer] = None

        # 谱分析器 (v9.3)
        self.spectral_analyzer: Optional['SpectralAnalyzer'] = None
        self.spectral_ppr: Optional['SpectralPPR'] = None
        self._spectral_clusters: Dict[str, int] = {}
        self._fiedler_values: Dict[str, float] = {}

        # 缓存
        self._ast_results: Dict[str, FileAnalysis] = {}
        self._graph_results: Dict[str, GraphRoleResult] = {}
        self._dep_map: Optional[Dict] = None

        # 初始化
        self._initialize_analyzers(dep_map_path)
    
    def _initialize_analyzers(self, dep_map_path: Optional[str]):
        """初始化分析器"""
        if self.debug:
            print(f"[Init] RAACS Role Classifier v{__version__}")
            print("[Init] Building symbol table...")
        
        # 1. 符号表构建
        collector = SymbolCollector(self.project_root)
        self.symbol_table = collector.collect()
        
        # 2. 角色传播
        propagator = RolePropagator(self.symbol_table)
        propagated = propagator.propagate()
        
        if self.debug:
            # 统计角色来源分布
            role_sources = Counter()
            for cls in self.symbol_table.global_classes_by_fqn.values():
                role_sources[cls.role_source.value] += 1
            
            print(f"[Init] Symbol table: {len(self.symbol_table.global_classes_by_fqn)} classes")
            print(f"[Init] Role sources: {dict(role_sources)}")
            print(f"[Init] Role propagation: {propagated} roles propagated")
        
        # 3. AST 分析器
        self.ast_classifier = CodeRoleClassifier(debug=self.debug, symbol_table=self.symbol_table)
        
        # 4. 图结构分析器
        dep_map = None
        
        # 优先使用提供的依赖图 JSON 文件
        if dep_map_path and os.path.exists(dep_map_path):
            if self.debug:
                print(f"[Init] Loading dependency graph from {dep_map_path}")
            with open(dep_map_path) as f:
                dep_map = json.load(f)
        
        # 否则从符号表原生构建（无需 pydeps）
        elif self.symbol_table:
            if self.debug:
                print("[Init] Building dependency graph from symbol table (native)...")
            dep_map = DependencyGraphAnalyzer.build_from_symbol_table(self.symbol_table)
        
        if dep_map:
            self._dep_map = dep_map
            self.graph_analyzer = DependencyGraphAnalyzer(dep_map, self.project_root, debug=self.debug)
            if self.debug:
                print(f"[Init] Dependency graph: {len(self.graph_analyzer.internal_modules)} modules")

            # 初始化谱分析器 (v9.3)
            if self.enable_spectral:
                self._initialize_spectral(dep_map)
        elif self.debug:
            print("[Init] No dependency graph available, graph layer disabled")

    def _initialize_spectral(self, dep_map: Dict):
        """初始化谱分析器"""
        if not SPECTRAL_AVAILABLE:
            if self.debug:
                print("[Init] Spectral analysis disabled (missing numpy/scipy/networkx)")
            return

        try:
            if self.debug:
                print("[Init] Initializing spectral analyzer...")

            self.spectral_analyzer = SpectralAnalyzer(dep_map)
            embedding = self.spectral_analyzer.compute_embedding()

            # 计算聚类
            n_clusters = min(5, max(2, len(dep_map) // 20))
            self._spectral_clusters = self.spectral_analyzer.get_clusters(n_clusters)

            # 保存 Fiedler 值
            for node in embedding.node_list:
                fiedler = embedding.get_fiedler_value(node)
                if fiedler is not None:
                    self._fiedler_values[node] = fiedler

            # 创建 SpectralPPR
            self.spectral_ppr = SpectralPPR(dep_map, sigma=1.0)

            if self.debug:
                print(f"[Init] Spectral analysis: {n_clusters} clusters, {len(self._fiedler_values)} nodes embedded")

        except Exception as e:
            if self.debug:
                print(f"[Init] Spectral analysis failed: {e}")
            self.spectral_analyzer = None
            self.spectral_ppr = None
    
    def analyze_project(self) -> Dict[str, IntegratedRoleResult]:
        """分析整个项目"""
        results = {}
        
        # 1. AST 分析所有文件
        if self.debug:
            print("\n[Phase 1] AST analysis...")
        
        for root, _, files in os.walk(self.project_root):
            for f in files:
                if f.endswith('.py'):
                    file_path = os.path.join(root, f)
                    ast_result = self.ast_classifier.analyze_file(file_path)
                    self._ast_results[file_path] = ast_result
        
        if self.debug:
            print(f"[Phase 1] Analyzed {len(self._ast_results)} files")
        
        # 2. 设置 AST 角色到图分析器
        if self.graph_analyzer:
            ast_roles_for_graph = {}
            for path, result in self._ast_results.items():
                # 将文件路径转换为模块名
                module_name = self._path_to_module(path)
                if module_name:
                    ast_roles_for_graph[module_name] = result.role_score.primary_role.value
            
            self.graph_analyzer.set_ast_roles(ast_roles_for_graph)
            
            if self.debug:
                print(f"\n[Phase 2] Graph analysis...")
            
            # 3. 图结构分析
            for module_name in self.graph_analyzer.internal_modules:
                graph_result = self.graph_analyzer.analyze_graph_role(module_name)
                self._graph_results[module_name] = graph_result
        
        # 4. 融合
        if self.debug:
            print(f"\n[Phase 3] Fusion...")
        
        for file_path, ast_result in self._ast_results.items():
            module_name = self._path_to_module(file_path)
            
            # 获取图结构结果
            graph_result = self._graph_results.get(module_name) if module_name else None
            
            # 融合
            integrated = self._fuse_roles(file_path, ast_result, graph_result, module_name)
            results[file_path] = integrated
        
        return results
    
    def analyze_file(self, file_path: str) -> IntegratedRoleResult:
        """分析单个文件"""
        file_path = os.path.abspath(file_path)
        
        # AST 分析
        ast_result = self.ast_classifier.analyze_file(file_path)
        
        # 图结构分析
        module_name = self._path_to_module(file_path)
        graph_result = None
        if self.graph_analyzer and module_name:
            graph_result = self.graph_analyzer.analyze_graph_role(module_name)
        
        return self._fuse_roles(file_path, ast_result, graph_result, module_name)
    
    def _fuse_roles(self, file_path: str, ast_result: FileAnalysis,
                    graph_result: Optional[GraphRoleResult], 
                    module_name: Optional[str]) -> IntegratedRoleResult:
        """融合三层信号"""
        
        ast_role = ast_result.role_score.primary_role
        ast_confidence = min(ast_result.role_score.primary_score / 10, 1.0)
        
        # 默认值
        graph_role = GraphRole.NORMAL
        graph_confidence = 0.0
        graph_reasoning = "No dependency graph"
        arch_layer = ArchitecturalLayer.UNKNOWN
        in_degree = 0
        out_degree = 0
        caller_roles = {}
        
        if graph_result:
            graph_role = graph_result.graph_role
            graph_confidence = graph_result.graph_role_confidence
            graph_reasoning = graph_result.graph_role_reasoning
            arch_layer = graph_result.architectural_layer
            if graph_result.features:
                in_degree = graph_result.features.internal_in_degree
                out_degree = graph_result.features.internal_out_degree
                caller_roles = graph_result.features.caller_roles
        
        # 融合规则
        fusion_key = (ast_role, graph_role)
        if fusion_key in self.FUSION_RULES:
            final_role, confidence_mult, fusion_reasoning = self.FUSION_RULES[fusion_key]
            final_confidence = ast_confidence * confidence_mult
        else:
            final_role = ast_role
            final_confidence = ast_confidence
            fusion_reasoning = "No fusion rule applied"
        
        # 架构层次调整
        if arch_layer == ArchitecturalLayer.INFRASTRUCTURE and final_role == Role.LOGIC:
            final_role = Role.UTIL
            fusion_reasoning += "; Infrastructure layer -> UTIL"
        elif arch_layer == ArchitecturalLayer.INTERFACE and final_role == Role.LOGIC:
            final_role = Role.ADAPTER
            final_confidence *= 0.7
            fusion_reasoning += "; Interface layer -> ADAPTER"
        
        # 提取实体信息
        entities = []
        for e in ast_result.entities:
            entities.append({
                'name': e.name,
                'type': e.entity_type,
                'role': e.role.value,
                'confidence': e.confidence,
                'inherited_from': getattr(e, 'inherited_from', '')
            })
        
        # 谱分析特征 (v9.3)
        spectral_cluster = self._spectral_clusters.get(module_name, -1) if module_name else -1
        fiedler_value = self._fiedler_values.get(module_name, 0.0) if module_name else 0.0

        return IntegratedRoleResult(
            module_name=module_name or "",
            file_path=file_path,
            ast_role=ast_role,
            ast_confidence=ast_confidence,
            ast_reasoning=ast_result.role_score.reasoning,
            graph_role=graph_role,
            graph_confidence=graph_confidence,
            graph_reasoning=graph_reasoning,
            architectural_layer=arch_layer,
            final_role=final_role,
            final_confidence=final_confidence,
            fusion_reasoning=fusion_reasoning,
            entities=entities,
            role_purity=ast_result.role_purity,
            in_degree=in_degree,
            out_degree=out_degree,
            caller_roles=caller_roles,
            spectral_cluster=spectral_cluster,
            fiedler_value=fiedler_value
        )
    
    def _path_to_module(self, file_path: str) -> Optional[str]:
        """Convert file path to module name (Consistent with SymbolCollector)"""
        try:
            # FIX: Use project_root, not parent_dir, to align with SymbolCollector/Graph keys
            rel_path = os.path.relpath(file_path, self.project_root)
            
            module_path = rel_path.replace(os.sep, '.').replace('/', '.')
            if module_path.endswith('.py'):
                module_path = module_path[:-3]
            if module_path.endswith('.__init__'):
                module_path = module_path[:-9]
            return module_path
        except:
            return None


# ============================================================================
# CLI
# ============================================================================

COLORS = {
    Role.TEST: '\033[90m',
    Role.NAMESPACE: '\033[36m',
    Role.INTERFACE: '\033[35m',
    Role.SCHEMA: '\033[34m',
    Role.ADAPTER: '\033[33m',
    Role.CONFIG: '\033[37m',
    Role.SCRIPT: '\033[31m',
    Role.UTIL: '\033[96m',
    Role.LOGIC: '\033[32m',
    Role.UNKNOWN: '\033[0m'
}
RESET = '\033[0m'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"RAACS Role Classifier v{__version__} - Three-Layer Fusion Analyzer"
    )
    parser.add_argument("project_root", help="Project root directory")
    parser.add_argument("--dep-map", help="Path to pre-generated dependency JSON (optional)")
    parser.add_argument("--save-deps", help="Save generated dependency graph to file")
    parser.add_argument("--file", help="Analyze specific file")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-fusion", action="store_true", help="Show fusion reasoning")
    parser.add_argument("--show-graph", action="store_true", help="Show graph features")
    parser.add_argument("--show-spectral", action="store_true", help="Show spectral analysis info (cluster, Fiedler)")
    parser.add_argument("--no-spectral", action="store_true", help="Disable spectral analysis")
    parser.add_argument("--ppr-target", metavar="MODULE", help="Run PPR for target module and show context window")
    parser.add_argument("--ppr-top-k", type=int, default=15, help="Number of context nodes for PPR (default: 15)")
    parser.add_argument("--version", action="version", version=f"v{__version__}")
    args = parser.parse_args()
    
    # 如果需要保存依赖图，先构建符号表并保存
    dep_map_path = args.dep_map
    if args.save_deps and not dep_map_path:
        if args.debug:
            print("[Init] Building symbol table for dependency export...")
        collector = SymbolCollector(os.path.abspath(args.project_root))
        temp_symbol_table = collector.collect()
        dep_map = DependencyGraphAnalyzer.build_from_symbol_table(temp_symbol_table)
        with open(args.save_deps, 'w') as f:
            json.dump(dep_map, f, indent=4)
        print(f"[Saved] Dependency graph saved to {args.save_deps}")
        dep_map_path = args.save_deps
    
    analyzer = IntegratedRoleAnalyzer(
        args.project_root,
        dep_map_path=dep_map_path,
        debug=args.debug,
        enable_spectral=not args.no_spectral
    )
    
    if args.file:
        result = analyzer.analyze_file(args.file)
        
        print(f"\n{'='*70}")
        print(f"File: {os.path.basename(result.file_path)}")
        print(f"Module: {result.module_name}")
        print(f"{'='*70}")
        
        # AST 角色
        print(f"\n[AST Layer]")
        print(f"  Role: {COLORS[result.ast_role]}{result.ast_role.value}{RESET} (conf={result.ast_confidence:.2f})")
        print(f"  Reasoning: {result.ast_reasoning}")
        
        # 图结构角色
        if args.show_graph:
            print(f"\n[Graph Layer]")
            print(f"  Graph Role: {result.graph_role.value} (conf={result.graph_confidence:.2f})")
            print(f"  Reasoning: {result.graph_reasoning}")
            print(f"  Arch Layer: {result.architectural_layer.value}")
            print(f"  In-degree: {result.in_degree}, Out-degree: {result.out_degree}")
            if result.caller_roles:
                print(f"  Caller roles: {result.caller_roles}")

        # 谱分析信息 (v9.3)
        if args.show_spectral and analyzer.spectral_analyzer:
            print(f"\n[Spectral Layer]")
            print(f"  Cluster: {result.spectral_cluster}")
            print(f"  Fiedler: {result.fiedler_value:.4f}")
        
        # 融合角色
        print(f"\n[Fused Role]")
        print(f"  ★ {COLORS[result.final_role]}{result.final_role.value}{RESET} (conf={result.final_confidence:.2f})")
        if args.show_fusion:
            print(f"  Fusion: {result.fusion_reasoning}")
        
        # 实体
        if result.entities:
            print(f"\n[Entities] ({len(result.entities)})")
            for e in result.entities[:10]:
                inherited = f" [← {e['inherited_from']}]" if e.get('inherited_from') else ""
                print(f"  [{e['type']}] {e['name']:<25} -> {e['role']}{inherited}")
    
    else:
        results = analyzer.analyze_project()
        
        # 按最终角色分组统计
        by_role = defaultdict(list)
        for path, result in results.items():
            by_role[result.final_role].append(result)
        
        print(f"\n{'='*70}")
        print("Role Distribution (AST + Graph Fused)")
        print(f"{'='*70}")
        for role in Role:
            if role != Role.UNKNOWN:
                count = len(by_role[role])
                if count > 0:
                    print(f"  {COLORS[role]}{role.value:<12}{RESET}: {count}")
        
        # 打印详细列表
        print(f"\n{'Path':<50} | {'AST':<10} | {'Graph':<12} | {'Final':<10} | {'in/out'}")
        print("-" * 110)
        
        for path in sorted(results.keys()):
            result = results[path]
            rel_path = os.path.relpath(path, args.project_root)[:50]

            # 标记角色变化
            changed = "→" if result.ast_role != result.final_role else " "

            print(f"{rel_path:<50} | "
                  f"{COLORS[result.ast_role]}{result.ast_role.value:<10}{RESET} | "
                  f"{result.graph_role.value:<12} | "
                  f"{changed}{COLORS[result.final_role]}{result.final_role.value:<10}{RESET} | "
                  f"{result.in_degree}/{result.out_degree}")

    # PPR 上下文窗口 (v9.3)
    if args.ppr_target and analyzer.spectral_ppr:
        print(f"\n{'='*70}")
        print(f"PPR Context Window for: {args.ppr_target}")
        print(f"{'='*70}")

        context = analyzer.spectral_ppr.run_ppr_with_cluster_boost(
            args.ppr_target,
            top_k=args.ppr_top_k,
            cluster_boost=1.5
        )

        if not context:
            print(f"  Module '{args.ppr_target}' not found in dependency graph")
        else:
            # 获取目标模块的聚类信息
            target_cluster = analyzer._spectral_clusters.get(args.ppr_target, -1)
            target_fiedler = analyzer._fiedler_values.get(args.ppr_target, 0.0)
            print(f"  Target Cluster: {target_cluster}, Fiedler: {target_fiedler:.4f}")
            print()

            print(f"  {'Rank':<5} {'Module':<45} {'Score':<10} {'Cluster':<8}")
            print("  " + "-" * 75)

            for rank, (module, score) in enumerate(context, 1):
                cluster = analyzer._spectral_clusters.get(module, -1)
                same_cluster = "★" if cluster == target_cluster else " "
                print(f"  {rank:<5} {module:<45} {score:.6f}   {same_cluster}C{cluster}")

            print()
            print(f"  ★ = Same cluster as target")

    elif args.ppr_target and not analyzer.spectral_ppr:
        print(f"\n[Warning] PPR not available. Install dependencies: pip install numpy scipy networkx")

