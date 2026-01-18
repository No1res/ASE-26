# raacs/core/fusion.py
"""
角色融合层。

目前直接导入 `IntegratedRoleAnalyzer` 和 `IntegratedRoleResult`，
后续可根据需要重构融合细则。
"""

from raacs.role_classifier import IntegratedRoleAnalyzer, IntegratedRoleResult

__all__ = [
    "IntegratedRoleAnalyzer",
    "IntegratedRoleResult",
]