# raacs/__init__.py
"""
RAACS 顶层包，导出常用类和函数。

外部用户可以通过 `from raacs import Role, CodeRoleClassifier, DependencyGraphAnalyzer` 等方式使用。

注意：PPR 扩散相关类 (CodeGraphBuilder, GraphVisualizer, run_ppr) 需要 networkx 依赖，
请使用 `from raacs.core.diffusion import ...` 单独导入。
"""

# 角色定义
from .core.roles import Role, RoleSource, ROLE_SOURCE_STRENGTH, ROLE_COMPATIBILITY, SignalWeight

# AST 分析
from .core.ast import CodeRoleClassifier, FileAnalysis, SymbolCollector, ProjectSymbolTable, RolePropagator

# 图分析
from .core.graph import GraphRole, ArchitecturalLayer, GraphRoleResult, DependencyGraphAnalyzer, RepositoryStats, DynamicThresholds

# 融合层
from .core.fusion import IntegratedRoleAnalyzer, IntegratedRoleResult, DependencyGraphGenerator

# PPR 扩散（延迟导入，需要 networkx）
# 使用 from raacs.core.diffusion import run_ppr, CodeGraphBuilder, GraphVisualizer

__all__ = [
    # 角色定义
    "Role",
    "RoleSource",
    "ROLE_SOURCE_STRENGTH",
    "ROLE_COMPATIBILITY",
    "SignalWeight",
    # AST 分析
    "CodeRoleClassifier",
    "FileAnalysis",
    "SymbolCollector",
    "ProjectSymbolTable",
    "RolePropagator",
    # 图分析
    "GraphRole",
    "ArchitecturalLayer",
    "GraphRoleResult",
    "DependencyGraphAnalyzer",
    "RepositoryStats",
    "DynamicThresholds",
    # 融合层
    "IntegratedRoleAnalyzer",
    "IntegratedRoleResult",
    "DependencyGraphGenerator",
]
