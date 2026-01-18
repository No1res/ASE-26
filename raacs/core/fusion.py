# raacs/core/fusion.py
"""
角色融合层 - 集成 AST 分析、图结构分析和角色传播的三层信号融合。

提供：
- IntegratedRoleResult: 集成角色分析结果数据类
- IntegratedRoleAnalyzer: 集成角色分析器
- DependencyGraphGenerator: 依赖图生成器
"""

import os
import json
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .roles import Role, RoleSource, ROLE_SOURCE_STRENGTH
from .ast import (
    CodeRoleClassifier, FileAnalysis,
    SymbolCollector, RolePropagator, ProjectSymbolTable,
)
from .graph import (
    DependencyGraphAnalyzer, GraphRole, ArchitecturalLayer,
    GraphRoleResult,
)


# ============================================================================
# 依赖图生成器
# ============================================================================

class DependencyGraphGenerator:
    """依赖图生成器 - 内置 pydeps 调用"""

    @staticmethod
    def generate(project_root: str, output_path: Optional[str] = None,
                 debug: bool = False) -> Optional[Dict]:
        """
        使用 pydeps 生成依赖图

        Args:
            project_root: 项目根目录
            output_path: 输出 JSON 路径（可选）
            debug: 调试模式

        Returns:
            依赖图字典，失败返回 None
        """
        project_root = os.path.abspath(project_root)
        project_name = os.path.basename(project_root)

        # 检查 pydeps 是否可用
        try:
            result = subprocess.run(
                ['pydeps', '--version'],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                if debug:
                    print("[Warning] pydeps not available, skipping graph analysis")
                return None
        except FileNotFoundError:
            if debug:
                print("[Warning] pydeps not installed, skipping graph analysis")
                print("[Hint] Install with: pip install pydeps")
            return None

        try:
            if debug:
                print(f"[DependencyGraph] Generating dependency map for {project_root}...")

            # 调用 pydeps - JSON 输出到 stdout
            cmd = [
                'pydeps',
                project_name,  # 使用项目名而非完整路径
                '--show-deps',
                '--no-show',
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(project_root)  # 在父目录执行
            )

            # pydeps 即使成功也可能返回非零退出码，检查输出
            if not result.stdout or not result.stdout.strip().startswith('{'):
                if debug:
                    print(f"[Warning] pydeps failed: {result.stderr}")
                return None

            # 解析 JSON 输出
            dep_map = json.loads(result.stdout)

            if debug:
                print(f"[DependencyGraph] Generated dependency map with {len(dep_map)} modules")

            # 如果指定了输出路径，保存到文件
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(dep_map, f, indent=4)
                if debug:
                    print(f"[DependencyGraph] Saved to {output_path}")

            return dep_map

        except json.JSONDecodeError as e:
            if debug:
                print(f"[Warning] Failed to parse pydeps output: {e}")
            return None
        except Exception as e:
            if debug:
                print(f"[Warning] Failed to generate dependency graph: {e}")
            return None


# ============================================================================
# 集成角色分析结果
# ============================================================================

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


# ============================================================================
# 集成角色分析器
# ============================================================================

class IntegratedRoleAnalyzer:
    """
    集成角色分析器

    分析流程：
    1. 第一遍：AST 分析 + 符号表构建
    2. 第二遍：角色传播（跨文件继承）
    3. 第三遍：图结构分析
    4. 融合：三层信号融合
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
                 auto_generate_deps: bool = True, debug: bool = False):
        """
        Args:
            project_root: 项目根目录
            dep_map_path: 预生成的依赖图 JSON 路径（可选）
            auto_generate_deps: 如果没有提供 dep_map_path，是否自动生成依赖图
            debug: 调试模式
        """
        self.project_root = os.path.abspath(project_root)
        self.debug = debug
        self.auto_generate_deps = auto_generate_deps

        # AST 分析器（带符号表）
        self.ast_classifier: Optional[CodeRoleClassifier] = None
        self.symbol_table: Optional[ProjectSymbolTable] = None

        # 图结构分析器
        self.graph_analyzer: Optional[DependencyGraphAnalyzer] = None

        # 缓存
        self._ast_results: Dict[str, FileAnalysis] = {}
        self._graph_results: Dict[str, GraphRoleResult] = {}

        # 初始化
        self._initialize_analyzers(dep_map_path)

    def _initialize_analyzers(self, dep_map_path: Optional[str]):
        """初始化分析器"""
        if self.debug:
            print("[Init] RAACS Integrated Role Analyzer")
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

        # 优先使用提供的依赖图
        if dep_map_path and os.path.exists(dep_map_path):
            if self.debug:
                print(f"[Init] Loading dependency graph from {dep_map_path}")
            with open(dep_map_path) as f:
                dep_map = json.load(f)

        # 如果没有提供且允许自动生成
        elif self.auto_generate_deps:
            if self.debug:
                print("[Init] Auto-generating dependency graph...")
            dep_map = DependencyGraphGenerator.generate(
                self.project_root,
                debug=self.debug
            )

        if dep_map:
            self.graph_analyzer = DependencyGraphAnalyzer(dep_map, self.project_root, debug=self.debug)
            if self.debug:
                print(f"[Init] Dependency graph: {len(self.graph_analyzer.internal_modules)} modules")
        elif self.debug:
            print("[Init] No dependency graph available, graph layer disabled")

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
                print("\n[Phase 2] Graph analysis...")

            # 3. 图结构分析
            for module_name in self.graph_analyzer.internal_modules:
                graph_result = self.graph_analyzer.analyze_graph_role(module_name)
                self._graph_results[module_name] = graph_result

        # 4. 融合
        if self.debug:
            print("\n[Phase 3] Fusion...")

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
            caller_roles=caller_roles
        )

    def _path_to_module(self, file_path: str) -> Optional[str]:
        """将文件路径转换为模块名（与 pydeps 格式匹配）"""
        try:
            # 获取相对于项目父目录的路径，这样会包含项目名
            # 例如: /path/to/federation/entities/base.py -> federation.entities.base
            parent_dir = os.path.dirname(self.project_root)
            rel_path = os.path.relpath(file_path, parent_dir)

            module_path = rel_path.replace(os.sep, '.').replace('/', '.')
            if module_path.endswith('.py'):
                module_path = module_path[:-3]
            if module_path.endswith('.__init__'):
                module_path = module_path[:-9]
            return module_path
        except:
            return None


__all__ = [
    "IntegratedRoleResult",
    "IntegratedRoleAnalyzer",
    "DependencyGraphGenerator",
]
