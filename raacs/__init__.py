"""
RAACS (Role-Aware Adaptive Context System) - 核心库

提供代码角色分类的核心功能：
- ast_analyzer: AST 层分析 + 符号表层分析 (v8.1)
- graph_analyzer: 图结构层分析 + 动态阈值系统
- spectral_ppr: 谱加权 PPR 分析（Laplacian + PageRank）

v9.3 新增：
- SpectralPPR: 结合 Laplacian Eigenvector 的 PPR 分析
- SpectralAnalyzer: 图谱分析（聚类、中心性）
"""

__version__ = "9.3"

# 抑制被分析代码中的 SyntaxWarning
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning)

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

# 谱加权 PPR（延迟导入，需要 numpy, scipy, networkx）
# 使用: from raacs.spectral_ppr import SpectralPPR, SpectralAnalyzer

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

