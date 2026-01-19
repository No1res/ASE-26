import ast
import os
import argparse
from collections import defaultdict
from typing import Set, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

class Role(Enum):
    """代码角色枚举"""
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
class RoleScore:
    """角色打分结果"""
    primary_role: Role
    primary_score: float
    secondary_roles: List[Tuple[Role, float]] = field(default_factory=list)
    
    def get_all_roles(self) -> List[Role]:
        """获取所有角色（主+次）"""
        roles = [self.primary_role]
        roles.extend([r for r, _ in self.secondary_roles])
        return roles

@dataclass
class EntityRole:
    """实体级角色（类/函数）"""
    name: str
    type: str  # 'class' or 'function'
    role: Role
    score: float
    lineno: int

@dataclass
class FileAnalysis:
    """文件分析结果"""
    file_path: str
    role_score: RoleScore
    entity_roles: List[EntityRole] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)

class CodeRoleClassifier:
    """
    RAACS 3.0: 多标签、细粒度代码角色分类器
    """
    
    # 框架特征指纹库
    FRAMEWORK_SIGNATURES = {
        'imports': {
            'django', 'flask', 'fastapi', 'rest_framework', 'starlette', 'aiohttp',
            'tornado', 'bottle', 'falcon', 'pydantic', 'sqlalchemy', 'pytest', 'unittest'
        },
        'decorators': {
            'route', 'get', 'post', 'put', 'delete', 'patch',
            'login_required', 'require_http_methods', 'api_view',
            'task', 'shared_task', 'pytest.fixture', 'fixture'
        },
        'test_base_classes': {
            'TestCase', 'unittest.TestCase', 'TestSuite', 'IsolatedAsyncioTestCase'
        }
    }
    
    # 路径模式识别
    PATH_PATTERNS = {
        Role.TEST: ['test', 'tests', 'testing', '__tests__', 'spec'],
        Role.CONFIG: ['config', 'conf', 'settings', 'constants'],
        Role.SCRIPT: ['scripts', 'bin', 'tools', 'management/commands'],
        Role.ADAPTER: ['views', 'api', 'controllers', 'handlers', 'routes'],
        Role.SCHEMA: ['models', 'schemas', 'entities', 'serializers'],
        Role.UTIL: ['utils', 'helpers', 'common', 'lib']
    }
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """分析单个文件，返回完整分析结果"""
        if not os.path.exists(file_path):
            return FileAnalysis(
                file_path=file_path,
                role_score=RoleScore(Role.UNKNOWN, 0.0)
            )
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self.analyze_source(source_code, file_path=file_path)
        except SyntaxError as e:
            if self.debug:
                print(f"[SyntaxError] {file_path}: {e}")
            return FileAnalysis(
                file_path=file_path,
                role_score=RoleScore(Role.UNKNOWN, 0.0)
            )
        except Exception as e:
            if self.debug:
                print(f"[Error] {file_path}: {e}")
            return FileAnalysis(
                file_path=file_path,
                role_score=RoleScore(Role.UNKNOWN, 0.0)
            )

    def analyze_source(self, source_code: str, file_path: str = "") -> FileAnalysis:
        """核心分析流水线"""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return FileAnalysis(
                file_path=file_path,
                role_score=RoleScore(Role.UNKNOWN, 0.0)
            )

        # 1. 提取 AST 特征（文件级 + 实体级）
        features = self._extract_features(tree, file_path)
        entity_roles = self._analyze_entities(tree, features)
        
        # 2. 执行打分决策
        role_score = self._score_roles(features)
        
        return FileAnalysis(
            file_path=file_path,
            role_score=role_score,
            entity_roles=entity_roles,
            features=features
        )

    def _extract_features(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """收集 AST 统计信息与语义指纹"""
        f = defaultdict(int)
        f['framework_imports'] = set()
        f['framework_decorators'] = set()
        f['base_classes'] = set()
        f['all_imports'] = []  # 保存所有import信息
        
        # 路径信息
        f['file_path'] = file_path
        f['filename'] = os.path.basename(file_path).lower() if file_path else ""
        f['dir_path'] = os.path.dirname(file_path).lower() if file_path else ""
        
        # 纯度检查标记
        has_logic_stmt = False
        has_func_def = False
        has_class_def = False

        for node in ast.walk(tree):
            # --- 依赖分析（修复多import bug）---
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    f['all_imports'].append(module_name)
                    top_pkg = module_name.split('.')[0]
                    if top_pkg in self.FRAMEWORK_SIGNATURES['imports']:
                        f['framework_imports'].add(top_pkg)
                        
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                f['all_imports'].append(module_name)
                top_pkg = module_name.split('.')[0] if module_name else ""
                if top_pkg in self.FRAMEWORK_SIGNATURES['imports']:
                    f['framework_imports'].add(top_pkg)

            # --- 类定义分析 ---
            elif isinstance(node, ast.ClassDef):
                has_class_def = True
                f['class_count'] += 1
                
                for base in node.bases:
                    base_name = self._get_name(base)
                    f['base_classes'].add(base_name)
                    
                    if 'Test' in base_name or 'TestCase' in base_name:
                        f['is_test_class'] += 1
                    if 'Exception' in base_name or 'Error' in base_name:
                        f['is_exception_class'] += 1
                    if base_name in ['ABC', 'Protocol']:
                        f['is_abstract_class'] += 1
                    if base_name in ['BaseModel', 'Model', 'Schema']:
                        f['is_model_class'] += 1

                for deco in node.decorator_list:
                    deco_name = self._get_name(deco)
                    f['framework_decorators'].add(deco_name)
                    if deco_name == 'dataclass':
                        f['is_dataclass'] += 1

            # --- 函数定义分析 ---
            elif isinstance(node, ast.FunctionDef):
                has_func_def = True
                f['function_count'] += 1
                if node.name.startswith('test_'):
                    f['test_func_count'] += 1
                if self._is_empty_body(node.body):
                    f['empty_func_count'] += 1
                
                for deco in node.decorator_list:
                    deco_name = self._get_name(deco)
                    f['framework_decorators'].add(deco_name)
                    if 'abstract' in deco_name:
                        f['abstract_method_count'] += 1
                    if deco_name in self.FRAMEWORK_SIGNATURES['decorators']:
                        f['has_framework_deco'] += 1

            # --- 逻辑控制流 ---
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
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
                if self._is_not_implemented_raise(node):
                    f['not_implemented_count'] += 1
            
            # --- 赋值特征 ---
            elif isinstance(node, ast.Assign):
                f['assign_count'] += 1
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        f['upper_case_assign_count'] += 1
            elif isinstance(node, ast.AnnAssign):
                f['type_assign_count'] += 1

        f['has_logic_stmt'] = has_logic_stmt
        f['has_def'] = has_func_def or has_class_def
        f['path_signals'] = self._extract_path_signals(file_path)
        
        return f

    def _extract_path_signals(self, file_path: str) -> Dict[Role, float]:
        """从路径中提取角色信号"""
        signals = {}
        if not file_path:
            return signals
        
        path_lower = file_path.lower()
        for role, patterns in self.PATH_PATTERNS.items():
            for pattern in patterns:
                if f'/{pattern}/' in path_lower or path_lower.startswith(pattern + '/'):
                    signals[role] = signals.get(role, 0) + 1.0
        
        return signals

    def _analyze_entities(self, tree: ast.AST, features: Dict) -> List[EntityRole]:
        """分析类和函数级别的角色"""
        entities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                role, score = self._classify_class(node, features)
                entities.append(EntityRole(
                    name=node.name,
                    type='class',
                    role=role,
                    score=score,
                    lineno=node.lineno
                ))
            elif isinstance(node, ast.FunctionDef):
                # 只分析模块级函数，不分析类方法
                if self._is_module_level(tree, node):
                    role, score = self._classify_function(node, features)
                    entities.append(EntityRole(
                        name=node.name,
                        type='function',
                        role=role,
                        score=score,
                        lineno=node.lineno
                    ))
        
        return entities

    def _classify_class(self, node: ast.ClassDef, features: Dict) -> Tuple[Role, float]:
        """分类单个类"""
        scores = defaultdict(float)
        
        # 检查基类
        for base in node.bases:
            base_name = self._get_name(base)
            if 'Test' in base_name or 'TestCase' in base_name:
                scores[Role.TEST] += 3.0
            if 'Exception' in base_name or 'Error' in base_name:
                scores[Role.SCHEMA] += 2.5
            if base_name in ['ABC', 'Protocol']:
                scores[Role.INTERFACE] += 2.5
            if base_name in ['BaseModel', 'Model', 'Schema']:
                scores[Role.SCHEMA] += 2.0
        
        # 检查装饰器
        for deco in node.decorator_list:
            if self._get_name(deco) == 'dataclass':
                scores[Role.SCHEMA] += 2.0
        
        # 分析方法
        method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
        empty_methods = sum(1 for n in node.body 
                           if isinstance(n, ast.FunctionDef) and self._is_empty_body(n.body))
        
        if method_count > 0 and empty_methods / method_count > 0.7:
            scores[Role.INTERFACE] += 1.5
        
        # 默认逻辑
        if not scores:
            scores[Role.LOGIC] = 1.0
        
        best_role = max(scores.items(), key=lambda x: x[1])
        return best_role

    def _classify_function(self, node: ast.FunctionDef, features: Dict) -> Tuple[Role, float]:
        """分类单个函数"""
        scores = defaultdict(float)
        
        # 测试函数
        if node.name.startswith('test_'):
            scores[Role.TEST] += 3.0
        
        # 空函数或抽象方法
        if self._is_empty_body(node.body):
            scores[Role.INTERFACE] += 2.0
        
        # 路由装饰器
        for deco in node.decorator_list:
            deco_name = self._get_name(deco)
            if deco_name in self.FRAMEWORK_SIGNATURES['decorators']:
                scores[Role.ADAPTER] += 2.0
        
        # 分析函数体复杂度
        control_flow = sum(1 for n in ast.walk(node) 
                          if isinstance(n, (ast.If, ast.For, ast.While, ast.Try)))
        
        if control_flow == 0 and len(node.body) < 5:
            scores[Role.UTIL] += 1.5
        elif control_flow > 3:
            scores[Role.LOGIC] += 1.5
        
        if not scores:
            scores[Role.UTIL if features.get('class_count', 0) == 0 else Role.LOGIC] = 1.0
        
        best_role = max(scores.items(), key=lambda x: x[1])
        return best_role

    def _is_module_level(self, tree: ast.AST, func_node: ast.FunctionDef) -> bool:
        """判断函数是否为模块级（非类方法）"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in ast.walk(node):
                    return False
        return True

    def _score_roles(self, f: Dict[str, Any]) -> RoleScore:
        """打分机制：为每个角色计算分数，支持多标签"""
        scores = defaultdict(float)
        
        # 1. TEST 角色打分
        if f['test_func_count'] > 0:
            scores[Role.TEST] += 3.0
        if f['is_test_class'] > 0:
            scores[Role.TEST] += 3.0
        if f['assert_count'] > 0 and 'test' in f['filename']:
            scores[Role.TEST] += 2.0
        if Role.TEST in f.get('path_signals', {}):
            scores[Role.TEST] += f['path_signals'][Role.TEST] * 2.0
        
        # 2. SCRIPT 角色打分
        if f['has_main_entry'] > 0:
            scores[Role.SCRIPT] += 3.0
        if Role.SCRIPT in f.get('path_signals', {}):
            scores[Role.SCRIPT] += 1.5
        
        # 3. NAMESPACE 角色打分
        if f['filename'] == '__init__.py':
            if not f['has_logic_stmt'] and not f['has_def']:
                scores[Role.NAMESPACE] += 5.0
            else:
                scores[Role.NAMESPACE] += 1.0  # 有定义的__init__也保留一定分数
        
        # 4. CONFIG 角色打分
        if not f['has_def'] and f['assign_count'] > 0:
            if any(n in f['filename'] for n in ['settings', 'config', 'constants']):
                scores[Role.CONFIG] += 3.0
            if f['assign_count'] > 0 and f['upper_case_assign_count'] / f['assign_count'] > 0.3:
                scores[Role.CONFIG] += 2.0
        if Role.CONFIG in f.get('path_signals', {}):
            scores[Role.CONFIG] += 2.0
        
        # 5. INTERFACE 角色打分
        if f['function_count'] > 0:
            abstract_ratio = (f['abstract_method_count'] + f['empty_func_count'] + 
                            f['not_implemented_count']) / f['function_count']
            scores[Role.INTERFACE] += abstract_ratio * 3.0
        if f['is_abstract_class'] > 0:
            scores[Role.INTERFACE] += 2.5
        
        # 6. SCHEMA 角色打分
        if f['class_count'] > 0:
            if f['class_count'] == f['is_exception_class']:
                scores[Role.SCHEMA] += 3.0
            if f['is_dataclass'] > 0 or f['is_model_class'] > 0:
                scores[Role.SCHEMA] += 2.5
            if f['type_assign_count'] > f['function_count']:
                scores[Role.SCHEMA] += 1.5
            
            # 逻辑密度低 -> Schema倾向
            if f['function_count'] > 0:
                logic_density = f['control_flow_count'] / f['function_count']
                if logic_density < 0.5:
                    scores[Role.SCHEMA] += 1.5
        if Role.SCHEMA in f.get('path_signals', {}):
            scores[Role.SCHEMA] += 1.5
        
        # 7. ADAPTER 角色打分
        is_web_file = any(n in f['filename'] for n in 
                         ['view', 'api', 'controller', 'handler', 'serializer'])
        if is_web_file:
            scores[Role.ADAPTER] += 2.0
        if len(f['framework_imports']) > 0:
            scores[Role.ADAPTER] += 1.5
        if f['has_framework_deco'] > 0:
            scores[Role.ADAPTER] += 2.0
        if Role.ADAPTER in f.get('path_signals', {}):
            scores[Role.ADAPTER] += 2.0
        
        # 8. UTIL 角色打分
        if f['function_count'] > 0 and f['class_count'] == 0:
            # 无状态函数集合
            avg_complexity = f['control_flow_count'] / f['function_count'] if f['function_count'] > 0 else 0
            if avg_complexity < 2:
                scores[Role.UTIL] += 2.0
            if any(n in f['filename'] for n in ['util', 'helper', 'common']):
                scores[Role.UTIL] += 2.0
        if Role.UTIL in f.get('path_signals', {}):
            scores[Role.UTIL] += 2.0
        
        # 9. LOGIC 角色打分（兜底）
        if f['has_def'] or f['has_logic_stmt']:
            scores[Role.LOGIC] += 1.0
        if f['class_count'] > 0:
            scores[Role.LOGIC] += 0.5
        
        # 选出主角色和次要角色
        if not scores:
            return RoleScore(Role.UNKNOWN, 0.0)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_role, primary_score = sorted_scores[0]
        
        # 次要角色：分数 > 主角色的50%
        threshold = primary_score * 0.5
        secondary_roles = [(role, score) for role, score in sorted_scores[1:] 
                          if score > threshold]
        
        return RoleScore(
            primary_role=primary_role,
            primary_score=primary_score,
            secondary_roles=secondary_roles[:2]  # 最多保留2个次要角色
        )

    # --- AST Helpers ---
    def _get_name(self, node):
        """安全获取节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return self._get_name(node.func)
        return ""

    def _is_empty_body(self, body):
        """判断函数体是否为空"""
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt, (ast.Pass, ast.Ellipsis)):
                continue
            return False
        return True

    def _is_main_check(self, node: ast.If):
        """检测 if __name__ == '__main__'"""
        try:
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    return True
        except:
            pass
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
    parser = argparse.ArgumentParser(description="RAACS 3.0 Code Role Classifier")
    parser.add_argument("path", help="Path to analyze")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--entities", action="store_true", help="Show entity-level roles")
    args = parser.parse_args()

    classifier = CodeRoleClassifier(debug=args.debug)
    
    COLORS = {
        Role.TEST: '\033[90m',
        Role.NAMESPACE: '\033[36m',
        Role.INTERFACE: '\033[35m',
        Role.SCHEMA: '\033[34m',
        Role.ADAPTER: '\033[33m',
        Role.CONFIG: '\033[37m',
        Role.SCRIPT: '\033[31m',
        Role.UTIL: '\033[96m',
        Role.LOGIC: '\033[32m',
        Role.UNKNOWN: '\033[0m'
    }
    RESET = '\033[0m'

    target = os.path.abspath(args.path)
    
    if os.path.isfile(target):
        result = classifier.analyze_file(target)
        print(f"File: {os.path.basename(target)}")
        print(f"Primary Role: {COLORS[result.role_score.primary_role]}{result.role_score.primary_role.value}{RESET} ({result.role_score.primary_score:.2f})")
        
        if result.role_score.secondary_roles:
            print(f"Secondary Roles: ", end="")
            for role, score in result.role_score.secondary_roles:
                print(f"{COLORS[role]}{role.value}{RESET} ({score:.2f}) ", end="")
            print()
        
        if args.entities and result.entity_roles:
            print("\nEntity-level roles:")
            for entity in result.entity_roles:
                print(f"  L{entity.lineno} [{entity.type}] {entity.name}: {COLORS[entity.role]}{entity.role.value}{RESET} ({entity.score:.2f})")
    else:
        print(f"{'Filename':<50} | {'Primary Role':<15} | {'Secondary':<20}")
        print("-" * 90)
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if f.endswith('.py'):
                    full_path = os.path.join(root, f)
                    result = classifier.analyze_file(full_path)
                    rel_path = os.path.relpath(full_path, target)
                    
                    primary = result.role_score.primary_role
                    secondary_str = ", ".join([r.value for r, _ in result.role_score.secondary_roles[:2]])
                    
                    print(f"{rel_path:<50} | {COLORS[primary]}{primary.value:<15}{RESET} | {secondary_str:<20}")