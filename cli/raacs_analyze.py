# cli/raacs_analyze.py
"""
RAACS 命令行接口 - 代码角色分析与可视化。

用法:
    python -m cli.raacs_analyze <repo_root> [options]

示例:
    # 仅分析，输出 JSON
    python -m cli.raacs_analyze ./my_project --out analysis.json

    # 分析并生成 HTML 可视化
    python -m cli.raacs_analyze ./my_project --viz graph.html

    # 仅生成依赖图可视化（不运行完整分析）
    python -m cli.raacs_analyze ./my_project --viz-only deps.html
"""
import argparse
import json
import os
from pathlib import Path
from enum import Enum
from dataclasses import asdict

from raacs.pipeline.analyze import run_analysis


def serialize_result(result):
    """将 IntegratedRoleResult 序列化为可 JSON 化的字典"""
    data = asdict(result)
    # 转换枚举值为字符串
    for key, value in data.items():
        if isinstance(value, Enum):
            data[key] = value.value
    return data


def generate_visualization(repo_root: str, output_path: str, role_results: dict = None,
                          debug: bool = False, physics: bool = True,
                          max_nodes: int = None, min_degree: int = 0,
                          exclude_roles: list = None, layout: str = "hierarchical"):
    """生成 HTML 可视化

    Args:
        repo_root: 项目根目录
        output_path: 输出 HTML 文件路径
        role_results: 角色分析结果字典（可选）
        debug: 调试模式
        physics: 是否启用物理模拟（禁用后打开更快，但布局固定）
        max_nodes: 最大节点数（按度数重要性排序选取）
        min_degree: 最小度数阈值
        exclude_roles: 排除的角色列表
        layout: 布局方式 - "hierarchical"(层次), "physics"(力导向), "role"(按角色分组)
    """
    from raacs.adapters.import_scanner import StaticImportScanner

    print(f"[RAACS] 正在生成可视化...")

    # 扫描依赖
    scanner = StaticImportScanner(repo_root, debug=debug)
    dep_map = scanner.scan()

    if not dep_map:
        print("[RAACS] 警告: 未找到模块依赖")
        return None

    print(f"[RAACS] 扫描到 {len(dep_map)} 个模块")

    # 尝试导入可视化模块
    try:
        from raacs.adapters.viz import RoleGraphVisualizer
    except ImportError as e:
        print(f"[RAACS] 错误: 缺少可视化依赖")
        print(f"[RAACS] 请安装: pip install pyvis")
        return None

    # 生成可视化
    visualizer = RoleGraphVisualizer(dep_map, role_results)

    # 从路径获取项目名作为标题
    project_name = os.path.basename(os.path.abspath(repo_root))
    title = f"RAACS - {project_name}"

    try:
        return visualizer.generate_html(
            output_path, 
            title=title, 
            physics=physics,
            max_nodes=max_nodes,
            min_degree=min_degree,
            exclude_roles=exclude_roles,
            layout=layout
        )
    except ImportError as e:
        print(f"[RAACS] 错误: 缺少可视化依赖")
        print(f"[RAACS] 请安装: pip install pyvis")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="RAACS 代码角色分析 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s ./my_project                          # 分析并输出 analysis.json
  %(prog)s ./my_project --out result.json        # 指定输出文件
  %(prog)s ./my_project --viz graph.html         # 分析并生成可视化
  %(prog)s ./my_project --viz-only deps.html     # 仅生成依赖图可视化
  %(prog)s ./my_project --debug                  # 调试模式
        """
    )
    parser.add_argument("repo_root", help="待分析的代码仓库根目录")
    parser.add_argument("--out", default="analysis.json", help="分析结果 JSON 输出路径 (默认: analysis.json)")
    parser.add_argument("--viz", metavar="HTML_PATH", help="生成交互式 HTML 依赖图可视化")
    parser.add_argument("--viz-only", metavar="HTML_PATH", help="仅生成依赖图可视化（跳过完整角色分析）")
    parser.add_argument("--no-physics", action="store_true", help="禁用物理模拟（HTML打开更快，但布局固定）")
    parser.add_argument("--max-nodes", type=int, metavar="N", help="可视化最大节点数（按度数重要性选取）")
    parser.add_argument("--min-degree", type=int, default=0, metavar="N", help="可视化最小度数阈值（默认: 0）")
    parser.add_argument("--exclude-roles", type=str, metavar="ROLES", 
                       help="排除的角色列表（逗号分隔，如: TEST,NAMESPACE）")
    parser.add_argument("--layout", type=str, default="hierarchical",
                       choices=["hierarchical", "physics", "role"],
                       help="可视化布局方式: hierarchical(层次布局,默认), physics(力导向), role(按角色分组)")
    parser.add_argument("--debug", action="store_true", help="调试模式，输出详细日志")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)

    if not os.path.isdir(repo_root):
        print(f"[RAACS] 错误: 目录不存在: {repo_root}")
        return 1

    # 解析排除角色
    exclude_roles = None
    if args.exclude_roles:
        exclude_roles = [r.strip().upper() for r in args.exclude_roles.split(',')]

    # 仅可视化模式
    if args.viz_only:
        physics = not args.no_physics
        result = generate_visualization(
            repo_root, args.viz_only, 
            debug=args.debug, physics=physics,
            max_nodes=args.max_nodes, 
            min_degree=args.min_degree,
            exclude_roles=exclude_roles,
            layout=args.layout
        )
        if result:
            print(f"[RAACS] 可视化已生成: {result}")
        return 0 if result else 1

    # 完整分析模式
    print(f"[RAACS] 开始分析: {repo_root}")
    results = run_analysis(repo_root, debug=args.debug)

    if not results:
        print("[RAACS] 警告: 未找到可分析的文件")
        return 1

    # 序列化结果
    data = [serialize_result(r) for r in results]

    # 保存 JSON
    output_path = Path(args.out)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[RAACS] 分析完成，结果已写入 {output_path}")

    # 统计信息
    role_counts = {}
    for r in results:
        role = r.final_role.value if hasattr(r.final_role, 'value') else str(r.final_role)
        role_counts[role] = role_counts.get(role, 0) + 1

    print(f"[RAACS] 角色分布: {role_counts}")

    # 生成可视化（如果请求）
    if args.viz:
        # 构建角色结果字典（用于可视化着色）
        role_results_dict = {r.file_path: r for r in results}
        physics = not args.no_physics
        result = generate_visualization(
            repo_root, args.viz, role_results_dict,
            debug=args.debug, physics=physics,
            max_nodes=args.max_nodes,
            min_degree=args.min_degree,
            exclude_roles=exclude_roles,
            layout=args.layout
        )
        if result:
            print(f"[RAACS] 可视化已生成: {result}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
