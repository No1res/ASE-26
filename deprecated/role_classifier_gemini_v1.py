import ast
import os
import argparse
from collections import defaultdict, Counter
from typing import Set, List, Dict, Any, Optional
from dataclasses import dataclass, field

# --- Data Structures ---

@dataclass
class RoleScore:
    """角色评分卡"""
    scores: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def add(self, role: str, points: float, reason: str = ""):
        self.scores[role] += points
        # Debug usage: print(f"  [+] {role} +{points} ({reason})")

    def top_role(self) -> str:
        if not self.scores:
            return "UNKNOWN"
        # 按分数降序排列
        sorted_roles = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        top_role, score = sorted_roles[0]
        # 设定最小阈值，防止噪音
        if score < 1.0: return "UNKNOWN"
        return top_role

    def get_confidence(self) -> float:
        if not self.scores: return 0.0
        total = sum(self.scores.values())
        top = sorted(self.scores.values(), reverse=True)[0]
        return round(top / (total + 0.01), 2)

@dataclass
class SymbolInfo:
    """细粒度符号信息 (类/函数)"""
    name: str
    type: str  # 'class' or 'function'
    role: str
    start_line: int
    end_line: int

@dataclass
class FileAnalysisResult:
    """文件级分析结果"""
    path: str
    main_role: str
    confidence: float
    symbols: List[SymbolInfo]
    scores: Dict[str, float]

class CodeRoleClassifier:
    """
    RAACS 2.1: 基于加权评分与细粒度分析的代码角色分类器
    """
    
    # 角色定义
    ROLE_TEST = "TEST"
    ROLE_INTERFACE = "INTERFACE"   # 抽象/协议
    ROLE_SCHEMA = "SCHEMA"         # 数据定义
    ROLE_ADAPTER = "ADAPTER"       # 框架入口/API
    ROLE_CONFIG = "CONFIG"
    ROLE_SCRIPT = "SCRIPT"
    ROLE_UTIL = "UTIL"             # 新增：工具/无状态函数
    ROLE_LOGIC = "LOGIC"           # 核心业务逻辑
    ROLE_UNKNOWN = "UNKNOWN"

    # 特征关键词库 (文件名/路径)
    PATH_KEYWORDS = {
        ROLE_TEST: ['test', 'spec', 'mock', 'fixtures'],
        ROLE_CONFIG: ['config', 'setting', 'conf', 'env', 'url', 'registry'],
        ROLE_ADAPTER: ['view', 'api', 'controller', 'handler', 'route', 'endpoint', 'serializer'],
        ROLE_SCHEMA: ['schema', 'model', 'dto', 'entity', 'type', 'interface', 'protocol'],
        ROLE_UTIL: ['util', 'helper', 'common', 'lib', 'tool', 'formatter'],
        ROLE_SCRIPT: ['script', 'cmd', 'cli', 'manage', 'run']
    }

    # 框架特征指纹
    FRAMEWORK_SIGNATURES = {
        'web_adapter': {
            'imports': {'django', 'flask', 'fastapi', 'starlette', 'aiohttp', 'tornado', 'rest_framework'},
            'decorators': {'route', 'get', 'post', 'api_view', 'action'}
        },
        'schema_def': {
            'imports': {'pydantic', 'attrs', 'marshmallow', 'typing'},
            'decorators': {'dataclass', 'attr'},
            'classes': {'BaseModel', 'Schema', 'Model'}
        },
        'test_framework': {
            'imports': {'pytest', 'unittest', 'mock', 'faker'},
            'classes': {'TestCase'}
        }
    }

    def analyze_file(self, file_path: str) -> FileAnalysisResult:
        if not os.path.exists(file_path):
            return self._empty_result(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self._analyze_content(source_code, file_path)
        except Exception as e:
            # 记录错误但不中断流，方便调试
            # logging.error(f"Error parsing {file_path}: {e}")
            return self._empty_result(file_path)

    def _analyze_content(self, source_code: str, file_path: str) -> FileAnalysisResult:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return self._empty_result(file_path)

        scorer = RoleScore()
        visitor = FeatureVisitor()
        visitor.visit(tree)

        # === 1. 路径与文件名评分 (Path Heuristics) ===
        self._score_path(scorer, file_path)

        # === 2. 导入依赖评分 (Import Heuristics) ===
        self._score_imports(scorer, visitor.imports)

        # === 3. 全局结构评分 (Global Structure) ===
        if visitor.has_main_block:
            scorer.add(self.ROLE_SCRIPT, 5.0, "has_main_block")
        
        # 纯赋值判断 (Config)
        if not visitor.definitions and visitor.assign_count > 0:
            if visitor.upper_assign_count / (visitor.assign_count + 1) > 0.4:
                scorer.add(self.ROLE_CONFIG, 4.0, "constants_only")

        # === 4. 细粒度符号分析 (Symbol Analysis) ===
        symbols = []
        
        # 分析每一个类
        for class_node in visitor.classes:
            sym_role = self._analyze_class_role(class_node, visitor.imports)
            symbols.append(SymbolInfo(
                name=class_node.name, type='class', role=sym_role,
                start_line=class_node.lineno, end_line=class_node.end_lineno
            ))
            # 汇总分数：符号的角色向文件级角色投票
            scorer.add(sym_role, 2.0, f"contains_{sym_role}_class")

        # 分析每一个函数
        for func_node in visitor.functions:
            sym_role = self._analyze_function_role(func_node, visitor.imports)
            symbols.append(SymbolInfo(
                name=func_node.name, type='function', role=sym_role,
                start_line=func_node.lineno, end_line=func_node.end_lineno
            ))
            # 汇总分数
            scorer.add(sym_role, 1.0, f"contains_{sym_role}_func")
        
        # === 5. 兜底逻辑 ===
        # 如果符号都是 UTIL，文件就是 UTIL
        # 如果文件主要是 LOGIC，但没有显式特征，会被 LOGIC 捕获
        
        final_role = scorer.top_role()
        
        # 处理 __init__.py 特例
        if os.path.basename(file_path) == '__init__.py':
            if final_role == 'UNKNOWN' or final_role == 'UTIL':
                final_role = 'NAMESPACE'  # 纯净命名空间

        return FileAnalysisResult(
            path=file_path,
            main_role=final_role,
            confidence=scorer.get_confidence(),
            symbols=symbols,
            scores=dict(scorer.scores)
        )

    # --- Scoring Logic ---

    def _score_path(self, scorer: RoleScore, path: str):
        parts = path.lower().replace('\\', '/').split('/')
        fname = parts[-1]
        
        for role, keywords in self.PATH_KEYWORDS.items():
            # 检查文件名
            for kw in keywords:
                if kw in fname:
                    weight = 3.0 if role == self.ROLE_TEST else 2.0
                    scorer.add(role, weight, f"filename_has_{kw}")
            
            # 检查路径 (例如 tests/ folder)
            # 排除文件名本身，只看目录
            for parent in parts[:-1]:
                if parent in keywords:
                     scorer.add(role, 1.5, f"path_has_{parent}")

    def _score_imports(self, scorer: RoleScore, imports: Set[str]):
        # Test
        if imports & self.FRAMEWORK_SIGNATURES['test_framework']['imports']:
            scorer.add(self.ROLE_TEST, 3.0, "test_imports")
        
        # Schema
        if imports & self.FRAMEWORK_SIGNATURES['schema_def']['imports']:
            scorer.add(self.ROLE_SCHEMA, 2.0, "schema_imports")
            
        # Adapter / Web
        if imports & self.FRAMEWORK_SIGNATURES['web_adapter']['imports']:
            scorer.add(self.ROLE_ADAPTER, 2.0, "web_framework_imports")

    # --- Symbol Level Logic ---

    def _analyze_class_role(self, node: ast.ClassDef, file_imports: Set[str]) -> str:
        """推断单个类的角色"""
        bases = [self._get_name(b) for b in node.bases]
        decos = [self._get_name(d) for d in node.decorator_list]
        
        # 1. Test Class
        if 'TestCase' in bases or any('Test' in b for b in bases) or node.name.startswith('Test'):
            return self.ROLE_TEST
        
        # 2. Schema / Model
        if 'BaseModel' in bases or 'dataclass' in decos:
            return self.ROLE_SCHEMA
        if any(imp in file_imports for imp in ['pydantic', 'marshmallow']):
            if not self._has_complex_logic(node): # 如果逻辑很简单，大概率是 Schema
                return self.ROLE_SCHEMA
        
        # 3. Exception
        if any('Exception' in b or 'Error' in b for b in bases):
            return self.ROLE_SCHEMA # 或者是单独的 EXCEPTION 角色，这里归为定义类

        # 4. Interface
        if 'ABC' in bases or 'Protocol' in bases:
            return self.ROLE_INTERFACE
        
        # 5. Adapter (View)
        if any(d in self.FRAMEWORK_SIGNATURES['web_adapter']['decorators'] for d in decos):
            return self.ROLE_ADAPTER
        
        return self.ROLE_LOGIC

    def _analyze_function_role(self, node: ast.FunctionDef, file_imports: Set[str]) -> str:
        """推断单个函数的角色"""
        decos = [self._get_name(d) for d in node.decorator_list]
        
        # 1. Test Function
        if node.name.startswith('test_'):
            return self.ROLE_TEST
        
        # 2. Adapter (Route Handler)
        if any(d in self.FRAMEWORK_SIGNATURES['web_adapter']['decorators'] for d in decos):
            return self.ROLE_ADAPTER
        
        # 3. Interface (Abstract)
        if any('abstract' in d for d in decos) or self._is_empty_body(node.body):
            return self.ROLE_INTERFACE
        
        # 4. Util vs Logic
        # Util 通常是静态的、独立的、无状态的。
        # 如果函数体很短，且没有复杂的 class 实例化或 self 调用，倾向于 UTIL
        complexity = self._calc_complexity(node)
        if complexity < 3 and 'self' not in [a.arg for a in node.args.args]:
            return self.ROLE_UTIL
            
        return self.ROLE_LOGIC

    # --- Helpers ---

    def _get_name(self, node):
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute): return node.attr
        if isinstance(node, ast.Call): return self._get_name(node.func)
        return ""

    def _is_empty_body(self, body):
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant): continue # docstring
            if isinstance(stmt, (ast.Pass, ast.Ellipsis)): continue
            if isinstance(stmt, ast.Raise) and 'NotImplemented' in ast.dump(stmt): continue
            return False
        return True

    def _has_complex_logic(self, node: ast.AST) -> bool:
        """粗略判断是否有逻辑控制流"""
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                return True
        return False
    
    def _calc_complexity(self, node: ast.AST) -> int:
        """简化的 McCabe 复杂度"""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                count += 1
        return count

    def _empty_result(self, path):
        return FileAnalysisResult(path, self.ROLE_UNKNOWN, 0.0, [], {})

class FeatureVisitor(ast.NodeVisitor):
    """AST 访问者：收集全局信息和顶层定义"""
    def __init__(self):
        self.imports = set()
        self.classes = []
        self.functions = []
        self.assign_count = 0
        self.upper_assign_count = 0
        self.definitions = False
        self.has_main_block = False

    def visit_Import(self, node):
        for n in node.names:
            self.imports.add(n.name.split('.')[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.definitions = True
        self.classes.append(node)
        # 不再 generic_visit 内部，避免重复计算内部函数
    
    def visit_FunctionDef(self, node):
        # 这里的 visit 只捕获顶层函数
        self.definitions = True
        self.functions.append(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Assign(self, node):
        self.assign_count += 1
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                self.upper_assign_count += 1
    
    def visit_If(self, node):
        # Check if __name__ == "__main__"
        try:
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    self.has_main_block = True
        except: pass
        self.generic_visit(node)

# --- CLI Interface ---
if __name__ == "__main__":
    import sys
    import json
    
    # 简单的 JSON Encoder 处理 dataclass
    class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if hasattr(o, '__dict__'): return o.__dict__
            return super().default(o)

    path_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    classifier = CodeRoleClassifier()

    if os.path.isfile(path_arg):
        res = classifier.analyze_file(path_arg)
        print(json.dumps(res, cls=EnhancedJSONEncoder, indent=2))
    else:
        for root, _, files in os.walk(path_arg):
            for f in files:
                if f.endswith('.py'):
                    full_path = os.path.join(root, f)
                    res = classifier.analyze_file(full_path)
                    print(f"[{res.main_role}] {f} (Conf: {res.confidence})")
                    # 可选：打印内部符号详情
                    # for s in res.symbols:
                    #     print(f"  - {s.type} {s.name}: {s.role}")