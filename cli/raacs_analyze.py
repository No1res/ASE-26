# cli/raacs_analyze.py
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(description="RAACS 分析 CLI")
    parser.add_argument("repo_root", help="待分析的代码仓库根目录")
    parser.add_argument("--out", default="analysis.json", help="分析结果输出路径")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    results = run_analysis(args.repo_root, debug=args.debug)
    data = [serialize_result(r) for r in results]

    output_path = Path(args.out)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[RAACS] 分析完成，结果已写入 {output_path}")


if __name__ == "__main__":
    main()
