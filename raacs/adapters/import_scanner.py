# raacs/adapters/import_scanner.py
"""
轻量级静态 Import 扫描器 - 基于 AST，无需运行时环境。

替代 pydeps，实现纯静态的依赖图生成。
不需要安装被分析项目的依赖，只需要源代码即可。
"""

import ast
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path


@dataclass
class ModuleInfo:
    """模块信息"""
    name: str                          # 模块名 (如 cinder.volume.api)
    path: Optional[str]                # 文件路径
    imports: List[str] = field(default_factory=list)  # 导入的模块列表
    is_package: bool = False           # 是否是包 (__init__.py)

    # 详细的导入信息
    import_details: List[Dict] = field(default_factory=list)
    # 每项包含: {'module': str, 'names': List[str], 'level': int, 'line': int}


class ImportVisitor(ast.NodeVisitor):
    """AST 访问器，提取所有 import 语句"""

    def __init__(self, current_module: str, package_root: str):
        self.current_module = current_module
        self.package_root = package_root
        self.imports: List[str] = []
        self.import_details: List[Dict] = []

    def visit_Import(self, node: ast.Import):
        """处理 import xxx 语句"""
        for alias in node.names:
            module_name = alias.name
            self.imports.append(module_name)
            self.import_details.append({
                'module': module_name,
                'names': [alias.asname or alias.name],
                'level': 0,  # 绝对导入
                'line': node.lineno,
                'type': 'import'
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """处理 from xxx import yyy 语句"""
        module = node.module or ''
        level = node.level  # 相对导入级别 (0=绝对, 1=当前包, 2=上级包...)

        # 解析相对导入为绝对路径
        if level > 0:
            resolved_module = self._resolve_relative_import(module, level)
        else:
            resolved_module = module

        if resolved_module:
            self.imports.append(resolved_module)

            # 也记录具体导入的名称（用于更精细的分析）
            names = [alias.name for alias in node.names]
            self.import_details.append({
                'module': resolved_module,
                'names': names,
                'level': level,
                'line': node.lineno,
                'type': 'from_import',
                'original_module': module  # 保留原始写法
            })

            # 如果是 from x import y，y 可能是子模块
            # 尝试解析为子模块（如 from cinder import volume -> cinder.volume）
            for alias in node.names:
                if alias.name != '*':
                    potential_submodule = f"{resolved_module}.{alias.name}"
                    # 这个会在后处理中验证是否真的是模块
                    self.import_details[-1].setdefault('potential_submodules', []).append(potential_submodule)

        self.generic_visit(node)

    def _resolve_relative_import(self, module: str, level: int) -> Optional[str]:
        """解析相对导入为绝对模块名"""
        parts = self.current_module.split('.')

        # level=1 表示当前包，level=2 表示上级包...
        if level > len(parts):
            return None  # 相对导入超出包边界

        # 去掉尾部的 level-1 个部分（因为当前模块本身算一级）
        base_parts = parts[:len(parts) - level + 1]

        if module:
            return '.'.join(base_parts[:-1]) + '.' + module if base_parts else module
        else:
            return '.'.join(base_parts[:-1]) if len(base_parts) > 1 else (base_parts[0] if base_parts else None)


class StaticImportScanner:
    """
    静态 Import 扫描器

    纯静态分析，不执行任何代码，不需要安装依赖。
    """

    def __init__(self, project_root: str, debug: bool = False):
        self.project_root = os.path.abspath(project_root)
        self.debug = debug

        # 扫描结果
        self.modules: Dict[str, ModuleInfo] = {}

        # 项目内的包名（用于区分内部/外部依赖）
        self.internal_packages: Set[str] = set()

    def scan(self) -> Dict[str, Dict]:
        """
        扫描项目，返回 pydeps 兼容格式的依赖图

        Returns:
            {
                "module.name": {
                    "path": "/path/to/file.py",
                    "imports": ["other.module", ...],
                    "imported_by": ["caller.module", ...]  # 可选
                },
                ...
            }
        """
        if self.debug:
            print(f"[ImportScanner] Scanning project: {self.project_root}")

        # 1. 发现所有 Python 包和模块
        self._discover_modules()

        if self.debug:
            print(f"[ImportScanner] Found {len(self.modules)} modules")
            print(f"[ImportScanner] Internal packages: {self.internal_packages}")

        # 2. 解析每个模块的 imports
        self._parse_imports()

        # 3. 过滤和规范化（只保留内部依赖）
        self._normalize_imports()

        # 4. 转换为输出格式
        return self._to_output_format()

    def _discover_modules(self):
        """发现项目中的所有 Python 模块"""
        # 首先找到顶级包
        for item in os.listdir(self.project_root):
            item_path = os.path.join(self.project_root, item)
            if os.path.isdir(item_path):
                init_file = os.path.join(item_path, '__init__.py')
                if os.path.exists(init_file) and item.isidentifier():
                    self.internal_packages.add(item)

        if not self.internal_packages:
            # 如果没有找到包，把整个目录当作一个扁平模块集合
            if self.debug:
                print("[ImportScanner] No packages found, scanning flat structure")
            self._scan_flat_modules()
            return

        # 递归扫描每个包
        for package in self.internal_packages:
            package_path = os.path.join(self.project_root, package)
            self._scan_package(package, package_path)

    def _scan_flat_modules(self):
        """扫描扁平结构的模块（没有 __init__.py 的情况）"""
        for item in os.listdir(self.project_root):
            if item.endswith('.py') and not item.startswith('_'):
                module_name = item[:-3]  # 去掉 .py
                if module_name.isidentifier():
                    file_path = os.path.join(self.project_root, item)
                    self.modules[module_name] = ModuleInfo(
                        name=module_name,
                        path=file_path,
                        is_package=False
                    )
                    self.internal_packages.add(module_name)

    def _scan_package(self, package_name: str, package_path: str):
        """递归扫描一个包（支持隐式命名空间包）"""
        for root, dirs, files in os.walk(package_path):
            # 计算相对于项目根的路径
            rel_path = os.path.relpath(root, self.project_root)
            current_module = rel_path.replace(os.sep, '.')

            # 检查是否有 Python 文件
            has_py_files = any(f.endswith('.py') for f in files)

            # 过滤目录：
            # 1. 必须是有效的标识符
            # 2. 不能以下划线开头（除非是 __pycache__ 等）
            # 3. 必须包含 __init__.py 或者包含 .py 文件（隐式命名空间包）
            filtered_dirs = []
            for d in dirs:
                if not d.isidentifier():
                    continue
                if d.startswith('_'):
                    continue
                subdir_path = os.path.join(root, d)
                has_init = os.path.exists(os.path.join(subdir_path, '__init__.py'))
                has_subpy = any(f.endswith('.py') for f in os.listdir(subdir_path)
                               if os.path.isfile(os.path.join(subdir_path, f)))
                if has_init or has_subpy:
                    filtered_dirs.append(d)
            dirs[:] = filtered_dirs

            for file in files:
                if not file.endswith('.py'):
                    continue
                if file.startswith('_') and file != '__init__.py':
                    continue

                file_path = os.path.join(root, file)

                if file == '__init__.py':
                    module_name = current_module
                    is_package = True
                else:
                    module_name = f"{current_module}.{file[:-3]}"
                    is_package = False

                # 验证模块名
                if not all(part.isidentifier() for part in module_name.split('.')):
                    continue

                self.modules[module_name] = ModuleInfo(
                    name=module_name,
                    path=file_path,
                    is_package=is_package
                )

    def _parse_imports(self):
        """解析所有模块的 import 语句"""
        for module_name, info in self.modules.items():
            if not info.path or not os.path.exists(info.path):
                continue

            try:
                with open(info.path, 'r', encoding='utf-8', errors='ignore') as f:
                    source = f.read()

                # 抑制 SyntaxWarning（被分析代码中的无效转义序列等）
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SyntaxWarning)
                    tree = ast.parse(source, filename=info.path)
                
                visitor = ImportVisitor(module_name, self.project_root)
                visitor.visit(tree)

                info.imports = visitor.imports
                info.import_details = visitor.import_details

            except SyntaxError as e:
                if self.debug:
                    print(f"[ImportScanner] Syntax error in {info.path}: {e}")
            except Exception as e:
                if self.debug:
                    print(f"[ImportScanner] Error parsing {info.path}: {e}")

    def _normalize_imports(self):
        """规范化导入：解析子模块引用，过滤外部依赖"""
        all_module_names = set(self.modules.keys())

        for module_name, info in self.modules.items():
            normalized_imports = []

            for imp in info.imports:
                # 检查是否是内部模块
                if self._is_internal_import(imp, all_module_names):
                    # 找到最具体的匹配模块
                    matched = self._find_matching_module(imp, all_module_names)
                    if matched and matched not in normalized_imports:
                        normalized_imports.append(matched)

            # 同时检查 potential_submodules
            for detail in info.import_details:
                for potential in detail.get('potential_submodules', []):
                    if potential in all_module_names and potential not in normalized_imports:
                        normalized_imports.append(potential)

            info.imports = normalized_imports

    def _is_internal_import(self, import_name: str, all_modules: Set[str]) -> bool:
        """检查是否是内部导入"""
        # 直接匹配
        if import_name in all_modules:
            return True

        # 检查是否是内部包的前缀
        first_part = import_name.split('.')[0]
        if first_part in self.internal_packages:
            return True

        # 检查是否是某个模块的前缀
        for module in all_modules:
            if module.startswith(import_name + '.'):
                return True

        return False

    def _find_matching_module(self, import_name: str, all_modules: Set[str]) -> Optional[str]:
        """找到最匹配的内部模块"""
        # 精确匹配
        if import_name in all_modules:
            return import_name

        # 查找最长前缀匹配
        best_match = None
        for module in all_modules:
            if import_name.startswith(module + '.') or module.startswith(import_name + '.'):
                if import_name.startswith(module):
                    if best_match is None or len(module) > len(best_match):
                        best_match = module

        # 如果没有前缀匹配，检查是否是包导入（import_name 是某个模块的前缀）
        if best_match is None:
            for module in all_modules:
                if module.startswith(import_name + '.'):
                    # 返回 import_name 对应的包（如果存在）
                    if import_name in all_modules:
                        return import_name
                    # 否则返回第一个匹配的子模块
                    return module

        return best_match

    def _to_output_format(self) -> Dict[str, Dict]:
        """转换为 pydeps 兼容的输出格式"""
        output = {}

        for module_name, info in self.modules.items():
            output[module_name] = {
                'path': info.path,
                'imports': info.imports,
            }

        # 计算 imported_by（反向依赖）
        for module_name, data in output.items():
            imported_by = []
            for other_name, other_data in output.items():
                if module_name in other_data.get('imports', []):
                    imported_by.append(other_name)
            data['imported_by'] = imported_by

        return output


def scan_imports(project_root: str, debug: bool = False) -> Dict[str, Dict]:
    """
    便捷函数：扫描项目的 import 依赖

    Args:
        project_root: 项目根目录
        debug: 调试模式

    Returns:
        依赖图字典，格式与 pydeps 兼容
    """
    scanner = StaticImportScanner(project_root, debug=debug)
    return scanner.scan()


__all__ = [
    "StaticImportScanner",
    "scan_imports",
    "ModuleInfo",
]
