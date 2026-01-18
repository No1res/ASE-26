# raacs/pipeline/analyze.py
"""
分析管线入口。

提供 `run_analysis()`：输入仓库路径并返回分析结果 (list of IntegratedRoleResult)。
"""

import os
from typing import List, Optional

from raacs.adapters.pydeps import PydepsExtractor
from raacs.core.ast import CodeRoleClassifier, SymbolCollector, ProjectSymbolTable, RolePropagator
from raacs.core.graph import DependencyGraphAnalyzer
from raacs.core.fusion import IntegratedRoleAnalyzer, IntegratedRoleResult

def run_analysis(project_root: str, debug: bool = False) -> List[IntegratedRoleResult]:
    """
    执行完整的 RAACS 分析流程：
    1. 解析依赖图 (pydeps)
    2. 执行 AST 分析和角色传播
    3. 执行图结构分析
    4. 三层信号融合，生成最终角色

    Returns:
        集成角色分析结果列表。
    """
    project_root = os.path.abspath(project_root)

    # 1) 调用 pydeps 生成依赖图
    dep_map = PydepsExtractor.extract(project_root, debug=debug)

    # 2) 第一次遍历：AST 分析 + 符号收集
    symbol_collector = SymbolCollector(project_root)
    symbol_collector.collect()

    classifier = CodeRoleClassifier()
    file_analyses = classifier.analyze_project(project_root)

    # 3) 第二次遍历：角色传播
    propagator = RolePropagator(symbol_collector.project_symbol_table)
    propagator.propagate()

    # 4) 图分析
    graph_analyzer = DependencyGraphAnalyzer(dep_map) if dep_map else None

    # 5) 融合
    analyzer = IntegratedRoleAnalyzer(
        file_analyses=file_analyses,
        symbol_table=symbol_collector.project_symbol_table,
        graph_analyzer=graph_analyzer
    )
    final_results: List[IntegratedRoleResult] = analyzer.integrate()

    return final_results