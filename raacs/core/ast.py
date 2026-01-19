# raacs/core/ast.py
"""
AST 层分析入口。

本模块引入并重新导出 `raacs.ast_analyzer` 中实现的核心类：
- `CodeRoleClassifier`
- `FileAnalysis`
- `SymbolCollector`
- `ProjectSymbolTable`
- `RolePropagator`

这样可以保证现有逻辑不变，同时为后续拆分提供接口。
"""

from raacs.ast_analyzer import (
    CodeRoleClassifier,
    FileAnalysis,
    SymbolCollector,
    ProjectSymbolTable,
    RolePropagator,
    safe_parse_source,
)

# 对外暴露这些类
__all__ = [
    "CodeRoleClassifier",
    "FileAnalysis",
    "SymbolCollector",
    "ProjectSymbolTable",
    "RolePropagator",
    "safe_parse_source",
]