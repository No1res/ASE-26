# cli/raacs_context.py
"""
上下文构建 CLI。

TODO: 实现基于 RAACS 的上下文检索与压缩。
"""
import argparse

def main():
    parser = argparse.ArgumentParser(description="RAACS 上下文构建 CLI")
    parser.add_argument("repo_root", help="代码仓库根目录")
    parser.add_argument("--budget", type=int, default=4096, help="Token 预算")
    args = parser.parse_args()

    # TODO: 调用 diffusion 和 packing 构造上下文
    print("[raacs_context] 上下文构建尚未实现。")

if __name__ == "__main__":
    main()