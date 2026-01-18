# raacs/core/diffusion.py
"""
扩散层算法实现。

当前暴露 `run_ppr()` 作为外部调用入口，其内部使用原 ppr 中的实现。
"""

from raacs.ppr import run_ppr, CodeGraphBuilder, GraphVisualizer

__all__ = [
    "run_ppr",
    "CodeGraphBuilder",
    "GraphVisualizer",
]