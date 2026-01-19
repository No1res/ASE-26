import ast
import os
import argparse
from collections import defaultdict
from typing import Set, List, Dict, Any

class CodeRoleClassifier:
    """
    RAACS 2.0: 基于 AST 语义指纹的代码角色分类器
    """
    
    # 角色定义
    ROLE_TEST = "TEST"             # 验证层
    ROLE_NAMESPACE = "NAMESPACE"   # 命名空间层 (纯净 __init__)
    ROLE_INTERFACE = "INTERFACE"   # 抽象契约层
    ROLE_SCHEMA = "SCHEMA"         # 数据契约层
    ROLE_ADAPTER = "ADAPTER"       # 适配/入口层 (Views/API)
    ROLE_CONFIG = "CONFIG"         # 配置层
    ROLE_SCRIPT = "SCRIPT"         # 脚本层
    ROLE_LOGIC = "LOGIC"           # 核心逻辑层 (兜底)
    ROLE_UNKNOWN = "UNKNOWN"

    # 框架特征指纹库
    FRAMEWORK_SIGNATURES = {
        'imports': {
            'django', 'flask', 'fastapi', 'rest_framework', 'starlette', 'aiohttp',
            'tornado', 'bottle', 'falcon', 'pydantic', 'sqlalchemy'
        },
        'decorators': {
            'route', 'get', 'post', 'put', 'delete', 'patch',  # Flask/FastAPI
            'login_required', 'require_http_methods', 'api_view', # Django/DRF
            'task', 'shared_task' # Celery
        }
    }

    def analyze_file(self, file_path: str) -> str:
        """分析单个文件"""
        if not os.path.exists(file_path):
            return self.ROLE_UNKNOWN
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self.analyze_source(source_code, filename=os.path.basename(file_path))
        except Exception as e:
            # print(f"[Debug] Parse Error: {e}")
            return self.ROLE_UNKNOWN

    def analyze_source(self, source_code: str, filename: str = "") -> str:
        """
        核心分析流水线
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return self.ROLE_UNKNOWN

        # 1. 提取 AST 语义特征
        features = self._extract_features(tree)
        features['filename'] = filename.lower()
        
        # 2. 执行瀑布式决策树
        return self._apply_heuristics(features)

    def _extract_features(self, tree: ast.AST) -> Dict[str, Any]:
        """收集 AST 统计信息与语义指纹"""
        f = defaultdict(int)
        f['framework_imports'] = set()
        f['framework_decorators'] = set()
        f['base_classes'] = set()
        
        # 纯度检查标记
        has_logic_stmt = False  # 是否包含 if/for/while/try
        has_func_def = False    # 是否包含函数定义
        has_class_def = False   # 是否包含类定义

        for node in ast.walk(tree):
            # --- 依赖分析 ---
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = ""
                if isinstance(node, ast.Import):
                    for n in node.names: module_name = n.name
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                
                # 提取顶级包名 (e.g., django.db -> django)
                top_pkg = module_name.split('.')[0]
                if top_pkg in self.FRAMEWORK_SIGNATURES['imports']:
                    f['framework_imports'].add(top_pkg)

            # --- 类定义分析 ---
            elif isinstance(node, ast.ClassDef):
                has_class_def = True
                f['class_count'] += 1
                
                # 收集基类
                for base in node.bases:
                    base_name = self._get_name(base)
                    f['base_classes'].add(base_name)
                    
                    # 特殊基类检测
                    if 'Test' in base_name or 'TestCase' in base_name: f['is_test_class'] += 1
                    if 'Exception' in base_name or 'Error' in base_name: f['is_exception_class'] += 1
                    if base_name in ['ABC', 'Protocol']: f['is_abstract_class'] += 1
                    if base_name in ['BaseModel', 'Model']: f['is_model_class'] += 1

                # 收集装饰器
                for deco in node.decorator_list:
                    deco_name = self._get_name(deco)
                    f['framework_decorators'].add(deco_name)
                    if deco_name == 'dataclass': f['is_dataclass'] += 1

            # --- 函数定义分析 ---
            elif isinstance(node, ast.FunctionDef):
                has_func_def = True
                f['function_count'] += 1
                if node.name.startswith('test_'): f['test_func_count'] += 1
                if self._is_empty_body(node.body): f['empty_func_count'] += 1
                
                # 检查装饰器
                for deco in node.decorator_list:
                    deco_name = self._get_name(deco)
                    f['framework_decorators'].add(deco_name)
                    if 'abstract' in deco_name: f['abstract_method_count'] += 1
                    
                    # 检查框架装饰器 (e.g. @login_required)
                    if deco_name in self.FRAMEWORK_SIGNATURES['decorators']:
                        f['has_framework_deco'] += 1

            # --- 逻辑控制流 ---
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                # 排除 if __name__ == "__main__"
                if isinstance(node, ast.If) and self._is_main_check(node):
                    f['has_main_entry'] += 1
                else:
                    has_logic_stmt = True
                    f['control_flow_count'] += 1

            # --- 语句特征 ---
            elif isinstance(node, ast.Assert):
                f['assert_count'] += 1
            elif isinstance(node, ast.Raise):
                f['raise_count'] += 1
                if self._is_not_implemented_raise(node): f['not_implemented_count'] += 1
            
            # --- 赋值特征 ---
            elif isinstance(node, ast.Assign):
                f['assign_count'] += 1
                # 检查大写变量 (常量特征)
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        f['upper_case_assign_count'] += 1
            elif isinstance(node, ast.AnnAssign):
                f['type_assign_count'] += 1

        f['has_logic_stmt'] = has_logic_stmt
        f['has_def'] = has_func_def or has_class_def
        return f

    def _apply_heuristics(self, f: Dict[str, Any]) -> str:
        """
        决策逻辑 (Waterfall Priority)
        """
        fname = f['filename']

        # 1. TEST (验证层) - 优先级最高
        # 特征：文件名含test，或包含 unittest/pytest 特征
        if ('test' in fname and f['function_count'] > 0) or \
           f['test_func_count'] > 0 or f['is_test_class'] > 0:
            # 必须包含断言或测试类，防止误判普通含有 test 字符的逻辑文件
            if f['assert_count'] > 0 or f['is_test_class'] > 0 or f['test_func_count'] > 0:
                return self.ROLE_TEST

        # 2. SCRIPT (脚本层)
        # 特征：显式的入口检查
        if f['has_main_entry'] > 0:
            return self.ROLE_SCRIPT

        # 3. NAMESPACE (命名空间层)
        # 特征：必须是 __init__.py 且极其纯净 (无逻辑，无定义)
        if fname == '__init__.py':
            if not f['has_logic_stmt'] and not f['has_def']:
                return self.ROLE_NAMESPACE
            else:
                # 即使是 __init__.py，如果有逻辑，也属于 LOGIC/ADAPTER
                # 暂存一个标记，让它流向后续判断，如果后续没命中 ADAPTER，则归为 LOGIC
                pass 

        # 4. CONFIG (配置层)
        # 特征：无函数无类，主要是赋值，且大写变量占比高
        if not f['has_def'] and f['assign_count'] > 0:
            # 或者是特定的配置文件名
            if any(n in fname for n in ['settings', 'config', 'urls', 'constants']):
                return self.ROLE_CONFIG
            # 或者大写赋值比例高
            if f['upper_case_assign_count'] / f['assign_count'] > 0.3:
                return self.ROLE_CONFIG

        # 5. SCHEMA (Exception 特例)
        # 特征：全部是异常类定义
        if f['class_count'] > 0 and f['class_count'] == f['is_exception_class']:
            return self.ROLE_SCHEMA

        # 6. INTERFACE (抽象契约层)
        # 特征：抽象方法/空方法占比高，或继承 ABC
        total_methods = f['function_count']
        if total_methods > 0:
            abstract_indicators = f['abstract_method_count'] + f['empty_func_count'] + f['not_implemented_count']
            if f['is_abstract_class'] > 0 or (abstract_indicators / total_methods > 0.7):
                return self.ROLE_INTERFACE

        # 7. ADAPTER (适配层)
        # 特征：引用了框架 (Django/Flask)，使用了路由装饰器，或文件名暗示
        is_web_file = any(n in fname for n in ['view', 'api', 'controller', 'handler', 'schema', 'serializer'])
        has_framework_signal = (len(f['framework_imports']) > 0 or f['has_framework_deco'] > 0)
        
        if is_web_file or has_framework_signal:
            # 只有在包含定义或逻辑时才算 Adapter，否则可能是空文件
            if f['has_def'] or f['has_logic_stmt']:
                return self.ROLE_ADAPTER

        # 8. SCHEMA (数据契约层)
        # 特征：有类，逻辑密度低，使用了 dataclass/pydantic 或大量类型注解
        if f['class_count'] > 0:
            logic_density = f['control_flow_count'] / (f['function_count'] + 1)
            is_data_heavy = (f['is_dataclass'] > 0 or 
                             f['is_model_class'] > 0 or 
                             f['type_assign_count'] > f['function_count'])
            
            if is_data_heavy and logic_density < 1.5:
                return self.ROLE_SCHEMA
            
            # 即使没有显式 Data 特征，如果几乎没有逻辑，也倾向于 Schema
            if logic_density < 0.5:
                return self.ROLE_SCHEMA

        # 9. LOGIC / UTIL (兜底)
        # 区分 Util 和 Logic 的边界很模糊，这里做一个简单的区分：
        # 如果只有函数且没有类 -> UTIL (倾向于无状态)
        # 如果有类 -> LOGIC (倾向于有状态 Core)
        if f['function_count'] > 0 and f['class_count'] == 0:
            # 如果逻辑密度极高，可能还是 Logic，但 Util 是个安全的压缩假设
            return self.ROLE_LOGIC # 修正：为了安全，统一归为 LOGIC (UTIL 可以看作简单的 LOGIC)
            # 或者：return "UTIL"

        return self.ROLE_LOGIC

    # --- AST Helpers ---
    def _get_name(self, node):
        """安全获取节点名称 (Class/Func/Decorator)"""
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute): return node.attr
        if isinstance(node, ast.Call): return self._get_name(node.func)
        return ""

    def _is_empty_body(self, body):
        """判断函数体是否为空 (pass / ... / docstring only)"""
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue # Skip docstring
            if isinstance(stmt, (ast.Pass, ast.Ellipsis)):
                continue
            return False
        return True

    def _is_main_check(self, node: ast.If):
        """检测 if __name__ == '__main__'"""
        try:
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__": return True
        except: pass
        return False

    def _is_not_implemented_raise(self, node: ast.Raise):
        """检测 raise NotImplementedError"""
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            return 'NotImplemented' in node.exc.func.id
        if isinstance(node.exc, ast.Name):
            return 'NotImplemented' in node.exc.id
        return False

# --- CLI Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAACS Code Role Classifier")
    parser.add_argument("path", help="Path to analyze")
    args = parser.parse_args()

    classifier = CodeRoleClassifier()
    
    # 颜色代码
    COLORS = {
        'TEST': '\033[90m',       # Grey
        'NAMESPACE': '\033[36m',  # Cyan
        'INTERFACE': '\033[35m',  # Magenta
        'SCHEMA': '\033[34m',     # Blue
        'ADAPTER': '\033[33m',    # Yellow
        'CONFIG': '\033[37m',     # White
        'SCRIPT': '\033[31m',     # Red
        'LOGIC': '\033[32m',      # Green
        'UNKNOWN': '\033[0m'
    }
    RESET = '\033[0m'

    target = os.path.abspath(args.path)
    
    if os.path.isfile(target):
        r = classifier.analyze_file(target)
        print(f"File: {os.path.basename(target)}")
        print(f"Role: {COLORS.get(r, '')}{r}{RESET}")
    else:
        print(f"{'Filename':<50} | {'Role':<15}")
        print("-" * 70)
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if f.endswith('.py'):
                    full_path = os.path.join(root, f)
                    role = classifier.analyze_file(full_path)
                    rel_path = os.path.relpath(full_path, target)
                    c = COLORS.get(role, '')
                    print(f"{rel_path:<50} | {c}{role}{RESET}")