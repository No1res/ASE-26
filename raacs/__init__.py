# raacs/__init__.py
"""
RAACS 顶层包，导出常用类和函数。

外部用户可以通过 `from raacs import Role, CodeRoleClassifier, DependencyGraphAnalyzer` 等方式使用。
"""

from .core.roles import Role, RoleSource, ROLE_SOURCE_STRENGTH, ROLE_COMPATIBILITY, SignalWeight
from .core.ast import CodeRoleClassifier, FileAnalysis, SymbolCollector, ProjectSymbolTable, RolePropagator
from .core.graph import GraphRole, ArchitecturalLayer, GraphRoleResult, DependencyGraphAnalyzer, RepositoryStats, DynamicThresholds
from .core.fusion import IntegratedRoleAnalyzer, IntegratedRoleResult
from .core.diffusion import run_ppr, CodeGraphBuilder, GraphVisualizer

__all__ = [
    "Role",
    "RoleSource",
    "ROLE_SOURCE_STRENGTH",
    "ROLE_COMPATIBILITY",
    "SignalWeight",
    "CodeRoleClassifier",
    "FileAnalysis",
    "SymbolCollector",
    "ProjectSymbolTable",
    "RolePropagator",
    "GraphRole",
    "ArchitecturalLayer",
    "GraphRoleResult",
    "DependencyGraphAnalyzer",
    "RepositoryStats",
    "DynamicThresholds",
    "IntegratedRoleAnalyzer",
    "IntegratedRoleResult",
    "run_ppr",
    "CodeGraphBuilder",
    "GraphVisualizer",
]