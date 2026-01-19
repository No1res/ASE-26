# raacs/pipeline/analyze.py
"""
分析管线入口。

提供 `run_analysis()`：输入仓库路径并返回分析结果 (list of IntegratedRoleResult)。
"""

import os
from typing import List, Optional

from raacs.core.fusion import IntegratedRoleAnalyzer, IntegratedRoleResult


def run_analysis(project_root: str, debug: bool = False) -> List[IntegratedRoleResult]:
    """
    执行完整的 RAACS 分析流程：
    1. 解析依赖图 (pydeps)
    2. 执行 AST 分析和角色传播
    3. 执行图结构分析
    4. 三层信号融合，生成最终角色

    Args:
        project_root: 待分析的代码仓库根目录
        debug: 调试模式

    Returns:
        集成角色分析结果列表。
    """
    project_root = os.path.abspath(project_root)

    # IntegratedRoleAnalyzer 是自包含的，会自动执行：
    # 1. 符号表构建
    # 2. 角色传播
    # 3. AST 分析
    # 4. 依赖图生成（如果 pydeps 可用）
    # 5. 图结构分析
    analyzer = IntegratedRoleAnalyzer(
        project_root=project_root,
        auto_generate_deps=True,
        debug=debug
    )

    # 分析整个项目并返回结果
    results_dict = analyzer.analyze_project()

    # 转换为列表返回
    return list(results_dict.values())
