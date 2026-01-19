# raacs/core/graph.py
"""
图结构层分析入口。

封装来自 `raacs.graph_analyzer` 的关键定义和类，方便其它模块使用：
- `GraphRole`
- `ArchitecturalLayer`
- `GraphRoleResult`
- `DependencyGraphAnalyzer`
- `RepositoryStats`
- `DynamicThresholds`
"""

from raacs.graph_analyzer import (
    GraphRole,
    ArchitecturalLayer,
    GraphRoleResult,
    DependencyGraphAnalyzer,
    RepositoryStats,
    DynamicThresholds,
)

__all__ = [
    "GraphRole",
    "ArchitecturalLayer",
    "GraphRoleResult",
    "DependencyGraphAnalyzer",
    "RepositoryStats",
    "DynamicThresholds",
]