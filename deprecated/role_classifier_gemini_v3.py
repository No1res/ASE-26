import ast
import os
import argparse
from collections import defaultdict
from typing import Set, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

class Role(Enum):
    TEST = "TEST"             # 验证层
    NAMESPACE = "NAMESPACE"   # 命名空间层
    INTERFACE = "INTERFACE"   # 抽象契约层
    SCHEMA = "SCHEMA"         # 数据契约层
    ADAPTER = "ADAPTER"       # 适配/入口层
    CONFIG = "CONFIG"         # 配置层
    SCRIPT = "SCRIPT"         # 脚本层
    UTIL = "UTIL"             # 无状态工具层
    LOGIC = "LOGIC"           # 核心逻辑层
    UNKNOWN = "UNKNOWN"

@dataclass
class ClassifierConfig:
    """分类器阈值配置"""
    # 结构模式阈值
    ABSTRACTION_THRESHOLD: float = 0.65       # 抽象方法比例阈值
    DATA_RATIO_THRESHOLD: float = 0.60        # 数据字段比例阈值
    ORCHESTRATION_CALLS: int = 3              # 编排函数最少调用数
    COMPUTATION_DENSITY: float = 2.0          # 计算密度阈值 (LOGIC vs UTIL)
    
    # 权重配置 (混合策略)
    WEIGHT_FRAMEWORK: float = 4.0             # 框架指纹权重 (强信号)
    WEIGHT_PATTERN: float = 2.5               # 结构模式权重 (中信号)
    WEIGHT_PATH: float = 1.5                  # 路径暗示权重 (弱信号)
    
    # 命名关键词增强
    IO_KEYWORDS: Set[str] = field(default_factory=lambda: {
        'request', 'response', 'req', 'resp', 'ctx', 'context', 'session', 'dto', 'payload', 'event'
    })

@dataclass
class StructuralPattern:
    """结构模式特征"""
    is_io_boundary: bool = False          
    is_data_transformer: bool = False     
    is_orchestrator: bool = False         
    abstraction_ratio: float = 0.0        
    data_definition_ratio: float = 0.0    
    assertion_density: float = 0.0        
    fixture_ratio: float = 0.0            
    is_pure_declaration: bool = False     
    constant_ratio: float = 0.0           
    has_entry_point: bool = False         
    imperative_ratio: float = 0.0         
    computation_intensity: float = 0.0    

@dataclass
class RoleScore:
    """角色打分结果"""
    primary_role: Role
    primary_score: float
    secondary_roles: List[Tuple[Role, float]] = field(default_factory=list)
    matches: List[str] = field(default_factory=list)

@dataclass
class FileAnalysis:
    file_path: str
    role_score: RoleScore
    pattern: Optional[StructuralPattern] = None
    features: Dict[str, Any] = field(default_factory=dict)

class CodeRoleClassifier:
    """
    RAACS 3.2: Hybrid Code Role Classifier (Fixed False Positives)
    修正了 'test_' 前缀函数的误判逻辑
    """
    
    FRAMEWORK_SIGNATURES = {
        'web_adapter': {
            'imports': {'django', 'flask', 'fastapi', 'starlette', 'aiohttp', 'tornado', 'rest_framework', 'drf'},
            'decorators': {'route', 'get', 'post', 'put', 'delete', 'api_view', 'action', 'task'}
        },
        'schema_def': {
            'imports': {'pydantic', 'attrs', 'marshmallow', 'sqlalchemy', 'mongoengine'},
            'classes': {'BaseModel', 'Schema', 'Model', 'Entity'},
            'decorators': {'dataclass', 'attr'}
        },
        'test_framework': {
            'imports': {'pytest', 'unittest', 'mock', 'hypothesis'},
            'classes': {'TestCase'},
            'decorators': {'fixture', 'mock', 'patch'}
        }
    }

    PATH_HINTS = {
        Role.TEST: ['test', 'spec', 'mock', 'fixtures'],
        Role.CONFIG: ['config', 'setting', 'conf', 'env'],
        Role.SCRIPT: ['script', 'cmd', 'cli', 'manage', 'bin'],
        Role.ADAPTER: ['view', 'api', 'controller', 'handler', 'route', 'endpoint'],
        Role.SCHEMA: ['schema', 'model', 'dto', 'entity', 'type'],
        Role.UTIL: ['util', 'helper', 'common', 'lib']
    }

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()

    def analyze_file(self, file_path: str) -> FileAnalysis:
        if not os.path.exists(file_path):
            return self._empty_result(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self.analyze_source(source_code, file_path=file_path)
        except Exception:
            return self._empty_result(file_path)

    def analyze_source(self, source_code: str, file_path: str = "") -> FileAnalysis:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return self._empty_result(file_path)

        features = self._extract_features(tree, file_path)
        pattern = self._analyze_structural_pattern(tree, features)
        role_score = self._calculate_hybrid_score(pattern, features, file_path)
        
        return FileAnalysis(
            file_path=file_path,
            role_score=role_score,
            pattern=pattern,
            features=features
        )

    def _extract_features(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """提取基础统计特征 + 框架信号"""
        f = defaultdict(int)
        f['file_path'] = file_path
        f['filename'] = os.path.basename(file_path).lower() if file_path else ""
        
        f['imports'] = set()
        f['decorators'] = set()
        f['base_classes'] = set()
        
        f['sig_web_adapter'] = 0
        f['sig_schema'] = 0
        f['sig_test'] = 0
        
        # 判断文件名是否暗示测试文件
        is_test_file_path = any(hint in f['filename'] for hint in ['test', 'spec'])

        for node in ast.walk(tree):
            # Imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.Import):
                    for n in node.names: module = n.name
                else:
                    module = node.module or ""
                
                top_pkg = module.split('.')[0]
                f['imports'].add(top_pkg)
                
                if top_pkg in self.FRAMEWORK_SIGNATURES['web_adapter']['imports']: f['sig_web_adapter'] += 1
                if top_pkg in self.FRAMEWORK_SIGNATURES['schema_def']['imports']: f['sig_schema'] += 1
                if top_pkg in self.FRAMEWORK_SIGNATURES['test_framework']['imports']: f['sig_test'] += 1

            # Classes
            elif isinstance(node, ast.ClassDef):
                f['class_count'] += 1
                for base in node.bases:
                    name = self._get_name(base)
                    f['base_classes'].add(name)
                    if name in self.FRAMEWORK_SIGNATURES['schema_def']['classes']: f['sig_schema'] += 1
                    if name in self.FRAMEWORK_SIGNATURES['test_framework']['classes']: f['sig_test'] += 1
                
                for deco in node.decorator_list:
                    name = self._get_name(deco)
                    f['decorators'].add(name)
                    if name in self.FRAMEWORK_SIGNATURES['schema_def']['decorators']: f['sig_schema'] += 1

            # Functions (Corrected Logic)
            elif isinstance(node, ast.FunctionDef):
                f['function_count'] += 1
                
                # --- FIX: 上下文感知的测试函数判定 ---
                if node.name.startswith('test_'):
                    # 1. 如果在测试文件里，名字以 test_ 开头 -> 强信号
                    # 2. 如果不在测试文件里，必须包含断言(Assert) -> 强信号
                    # 3. 否则 (如在 utils.py 里的 test_connection) -> 忽略，防止误判
                    
                    has_assert = any(isinstance(n, ast.Assert) for n in ast.walk(node))
                    
                    if is_test_file_path or has_assert:
                        f['sig_test'] += 1
                # ------------------------------------
                
                for deco in node.decorator_list:
                    name = self._get_name(deco)
                    f['decorators'].add(name)
                    if any(s in name for s in self.FRAMEWORK_SIGNATURES['web_adapter']['decorators']):
                        f['sig_web_adapter'] += 1
                    if any(s in name for s in self.FRAMEWORK_SIGNATURES['test_framework']['decorators']):
                        f['sig_test'] += 1

            # Statements
            elif isinstance(node, ast.Assign): f['assign_count'] += 1
            elif isinstance(node, ast.Expr): f['expr_count'] += 1
            elif isinstance(node, (ast.If, ast.For, ast.While)): 
                if isinstance(node, ast.If) and self._is_main_check(node):
                    f['has_main_entry'] = True
                else:
                    f['control_flow_count'] += 1
            elif isinstance(node, ast.Call): f['call_count'] += 1

        f['is_pure_init'] = (f['filename'] == '__init__.py' and f['function_count'] == 0 and f['class_count'] == 0)
        
        return f

    def _analyze_structural_pattern(self, tree: ast.AST, features: Dict) -> StructuralPattern:
        p = StructuralPattern()
        
        total_funcs = features['function_count']
        total_classes = features['class_count']
        
        abstract_funcs = 0
        data_classes = 0
        io_funcs = 0
        orchestrations = 0
        assertions = 0
        constants = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._is_io_boundary_func(node): io_funcs += 1
                if self._is_orchestrator(node): orchestrations += 1
                if self._is_empty_body(node.body) or self._has_not_implemented(node):
                    abstract_funcs += 1
                assertions += sum(1 for n in ast.walk(node) if isinstance(n, ast.Assert))

            elif isinstance(node, ast.ClassDef):
                fields = sum(1 for n in node.body if isinstance(n, ast.AnnAssign))
                methods = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                if fields > methods or (methods == 0 and fields > 0):
                    data_classes += 1
                if any('Exception' in self._get_name(b) for b in node.bases):
                    data_classes += 1

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants += 1

        if total_funcs > 0:
            p.abstraction_ratio = abstract_funcs / total_funcs
            p.assertion_density = assertions / total_funcs
            p.is_orchestrator = (orchestrations / total_funcs) > 0.3
            p.computation_intensity = features['control_flow_count'] / total_funcs
        
        if total_classes > 0:
            p.data_definition_ratio = data_classes / total_classes
        
        p.is_io_boundary = (io_funcs > 0) or (features['sig_web_adapter'] > 0)
        p.is_pure_declaration = (total_funcs == 0 and total_classes == 0 and features['assign_count'] > 0)
        if features['assign_count'] > 0:
            p.constant_ratio = constants / features['assign_count']
            
        p.has_entry_point = features.get('has_main_entry', False)
        
        return p

    def _calculate_hybrid_score(self, p: StructuralPattern, f: Dict, file_path: str) -> RoleScore:
        scores = defaultdict(float)
        reasons = defaultdict(list)
        
        def add_score(role, points, reason):
            scores[role] += points
            reasons[role].append(reason)

        # 1. Framework Signals
        if f['sig_test'] > 0:
            add_score(Role.TEST, self.config.WEIGHT_FRAMEWORK, "Framework: Test imports/classes/asserts")
        if f['sig_web_adapter'] > 0:
            add_score(Role.ADAPTER, self.config.WEIGHT_FRAMEWORK, "Framework: Web Adapter signals")
        if f['sig_schema'] > 0:
            add_score(Role.SCHEMA, self.config.WEIGHT_FRAMEWORK, "Framework: Schema/ORM signals")

        # 2. Structural Patterns
        if p.assertion_density > 0.5:
            add_score(Role.TEST, self.config.WEIGHT_PATTERN, f"Pattern: High assertion density")
            
        if p.data_definition_ratio > self.config.DATA_RATIO_THRESHOLD:
            add_score(Role.SCHEMA, self.config.WEIGHT_PATTERN, "Pattern: High data class ratio")
        elif p.data_definition_ratio > 0.3 and p.computation_intensity < 0.5:
            add_score(Role.SCHEMA, 1.5, "Pattern: Moderate data definitions")
            
        if p.abstraction_ratio > self.config.ABSTRACTION_THRESHOLD:
            add_score(Role.INTERFACE, self.config.WEIGHT_PATTERN, "Pattern: High abstraction ratio")
            
        if p.is_io_boundary and f['sig_web_adapter'] == 0:
            add_score(Role.ADAPTER, self.config.WEIGHT_PATTERN, "Pattern: IO Boundary detected")
        if p.is_orchestrator:
            add_score(Role.ADAPTER, 1.5, "Pattern: Orchestrator logic")
            
        if p.is_pure_declaration or p.constant_ratio > 0.5:
            add_score(Role.CONFIG, self.config.WEIGHT_PATTERN, "Pattern: Declaration/Constants")
            
        if p.has_entry_point:
            add_score(Role.SCRIPT, self.config.WEIGHT_PATTERN, "Pattern: Main Entry Point")
            
        if f['function_count'] > 0 and f['class_count'] == 0:
            if p.computation_intensity < self.config.COMPUTATION_DENSITY:
                add_score(Role.UTIL, self.config.WEIGHT_PATTERN, "Pattern: Low complexity functions")
            else:
                add_score(Role.LOGIC, 1.0, "Pattern: High complexity functions")
        
        if f['function_count'] > 0 or f['class_count'] > 0:
            add_score(Role.LOGIC, 1.0, "Base: Contains logic")

        # 3. Path Hints
        for role, keywords in self.PATH_HINTS.items():
            path_lower = f['file_path'].lower()
            if any(kw in path_lower for kw in keywords):
                add_score(role, self.config.WEIGHT_PATH, f"Path: Matches '{keywords[0]}...'")

        if f['is_pure_init']:
            add_score(Role.NAMESPACE, 10.0, "Special: Pure __init__")

        if not scores:
            return RoleScore(Role.UNKNOWN, 0.0, matches=["No clear signals"])
            
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_role, primary_score = sorted_scores[0]
        
        secondary = []
        if len(sorted_scores) > 1:
            threshold = primary_score * 0.4
            secondary = [(r, s) for r, s in sorted_scores[1:] if s > threshold][:2]
            
        return RoleScore(
            primary_role=primary_role,
            primary_score=primary_score,
            secondary_roles=secondary,
            matches=reasons[primary_role]
        )

    def _is_io_boundary_func(self, node: ast.FunctionDef) -> bool:
        param_names = {arg.arg.lower() for arg in node.args.args}
        if not param_names.isdisjoint(self.config.IO_KEYWORDS): return True
        if node.returns:
            ret_name = self._get_name(node.returns).lower()
            if any(k in ret_name for k in ['response', 'json', 'dict', 'schema']): return True
        return False

    def _is_orchestrator(self, node: ast.FunctionDef) -> bool:
        calls = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call))
        control = sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While)))
        return calls >= self.config.ORCHESTRATION_CALLS and control <= 1

    def _get_name(self, node):
        if isinstance(node, ast.Name): return node.id
        if isinstance(node, ast.Attribute): return node.attr
        if isinstance(node, ast.Call): return self._get_name(node.func)
        return ""

    def _is_empty_body(self, body):
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant): continue
            if isinstance(stmt, (ast.Pass, ast.Ellipsis)): continue
            return False
        return True

    def _has_not_implemented(self, node):
        for n in ast.walk(node):
            if isinstance(n, ast.Raise):
                exc_name = self._get_name(n.exc) if isinstance(n.exc, ast.Name) else self._get_name(n.exc.func) if isinstance(n.exc, ast.Call) else ""
                if 'NotImplemented' in exc_name: return True
        return False

    def _is_main_check(self, node: ast.If):
        try:
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__": return True
        except: pass
        return False
    
    def _empty_result(self, path):
        return FileAnalysis(path, RoleScore(Role.UNKNOWN, 0.0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    classifier = CodeRoleClassifier()
    
    COLORS = {
        Role.TEST: '\033[90m', Role.NAMESPACE: '\033[36m',
        Role.INTERFACE: '\033[35m', Role.SCHEMA: '\033[34m',
        Role.ADAPTER: '\033[33m', Role.CONFIG: '\033[37m',
        Role.SCRIPT: '\033[31m', Role.UTIL: '\033[96m',
        Role.LOGIC: '\033[32m', Role.UNKNOWN: '\033[0m'
    }
    RESET = '\033[0m'
    
    target = os.path.abspath(args.path)
    if os.path.isfile(target):
        r = classifier.analyze_file(target)
        s = r.role_score
        print(f"File: {os.path.basename(target)}")
        print(f"Role: {COLORS[s.primary_role]}{s.primary_role.value}{RESET} (Score: {s.primary_score:.1f})")
        if args.detail:
            print("Matches:")
            for m in s.matches: print(f"  - {m}")
    else:
        print("-" * 80)
        print(f"{'File':<60} | {'Role':<10} | {'Reason (Top Match)'}")
        print("-" * 80)
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if f.endswith('.py'):
                    full_path = os.path.join(root, f)
                    r = classifier.analyze_file(full_path)
                    s = r.role_score
                    reason = s.matches[0] if s.matches else ""
                    rel_path = os.path.relpath(full_path, target)
                    if len(rel_path) > 60: rel_path = "..." + rel_path[-57:]
                    print(f"{rel_path:<60} | {COLORS[s.primary_role]}{s.primary_role.value:<10}{RESET} | {reason}")