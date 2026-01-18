# raacs/adapters/pydeps.py
"""
pydeps 适配器：调用 pydeps 生成依赖图，返回原始依赖映射。
"""

import json
import os
import subprocess
from typing import Optional, Dict

class PydepsExtractor:
    @staticmethod
    def extract(project_root: str, debug: bool = False) -> Optional[Dict]:
        """
        调用 pydeps 并解析输出。

        Args:
            project_root: 仓库根目录
            debug: 是否打印调试信息

        Returns:
            解析后的依赖图（dict），失败返回 None。
        """
        project_root = os.path.abspath(project_root)
        project_name = os.path.basename(project_root)

        # 1. 检查 pydeps 是否安装
        try:
            subprocess.run(["pydeps", "--version"], capture_output=True, check=True, text=True)
        except Exception:
            if debug:
                print("[PydepsExtractor] pydeps not available.")
            return None

        # 2. 调用 pydeps
        cmd = [
            "pydeps",
            project_name,
            "--show-deps",
            "--no-show",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(project_root))
            if not result.stdout.strip().startswith("{"):
                if debug:
                    print("[PydepsExtractor] pydeps returned no JSON.")
                return None
            dep_map = json.loads(result.stdout)
            return dep_map
        except Exception as e:
            if debug:
                print("[PydepsExtractor] failed:", e)
            return None