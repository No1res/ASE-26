"""
RAACS (Role-Aware Adaptive Context System) - 核心库

提供代码角色分类的核心功能：
- ast_analyzer: AST 层分析 + 符号表层分析 (v8.1)
- graph_analyzer: 图结构层分析 + 动态阈值系统

v8.1 新增：
- RoleSource: 角色来源追踪
- BaseInfo: 结构化基类信息
- 分离索引结构
- 弱信号覆盖逻辑
"""

__version__ = "9.2"

# AST 分析器
from .ast_analyzer import (
    CodeRoleClassifier,
    Role,
    RoleSource,           # 新增
    FileAnalysis,
    SymbolCollector,
    RolePropagator,
    ProjectSymbolTable,
    EntityRole,
    RoleScore,
    BaseInfo,             # 新增
    ClassSymbol,          # 新增
    ROLE_SOURCE_STRENGTH, # 新增
)

# 图结构分析器
from .graph_analyzer import (
    DependencyGraphAnalyzer,
    GraphRole,
    ArchitecturalLayer,
    GraphRoleResult,
    GraphFeatures,
    RepositoryStats,
    DynamicThresholds,
    compute_repository_stats,
)

__all__ = [
    # AST 分析
    'CodeRoleClassifier',
    'Role',
    'RoleSource',
    'FileAnalysis',
    'SymbolCollector',
    'RolePropagator',
    'ProjectSymbolTable',
    'EntityRole',
    'RoleScore',
    'BaseInfo',
    'ClassSymbol',
    'ROLE_SOURCE_STRENGTH',
    
    # 图结构分析
    'DependencyGraphAnalyzer',
    'GraphRole',
    'ArchitecturalLayer',
    'GraphRoleResult',
    'GraphFeatures',
    'RepositoryStats',
    'DynamicThresholds',
    'compute_repository_stats',
]

