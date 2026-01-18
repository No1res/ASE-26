# cli/raacs_analyze.py
import argparse
import json
from pathlib import Path

from raacs.pipeline.analyze import run_analysis

def main():
    parser = argparse.ArgumentParser(description="RAACS 分析 CLI")
    parser.add_argument("repo_root", help="待分析的代码仓库根目录")
    parser.add_argument("--out", default="analysis.json", help="分析结果输出路径")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    results = run_analysis(args.repo_root, debug=args.debug)
    data = [r.__dict__ for r in results]

    output_path = Path(args.out)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[RAACS] 分析完成，结果已写入 {output_path}")

if __name__ == "__main__":
    main()