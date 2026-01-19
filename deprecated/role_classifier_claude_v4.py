import ast
import os
import argparse
from collections import defaultdict
from typing import Set, List, Dict, Any, Tuple, Optional
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
class StructuralPattern:
    """结构模式分析结果"""
    # 交互模式
    is_io_boundary: bool = False          # 是否处理外部IO
    is_data_transformer: bool = False     # 是否做数据转换
    is_orchestrator: bool = False         # 是否编排调用
    
    # 抽象程度
    abstraction_ratio: float = 0.0        # 抽象方法比例
    implementation_density: float = 0.0   # 实现密度
    
    # 数据特征
    data_definition_ratio: float = 0.0    # 数据定义比例
    computation_intensity: float = 0.0    # 计算强度
    
    # 验证特征
    assertion_density: float = 0.0        # 断言密度
    fixture_ratio: float = 0.0            # 测试fixture比例
    
    # 配置特征
    is_pure_declaration: bool = False     # 纯声明式
    constant_ratio: float = 0.0           # 常量比例
    
    # 脚本特征
    has_entry_point: bool = False         # 有入口点
    imperative_ratio: float = 0.0         # 命令式语句比例

@dataclass
class RoleScore:
    """角色打分结果"""
    primary_role: Role
    primary_score: float
    secondary_roles: List[Tuple[Role, float]] = field(default_factory=list)
    reasoning: str = ""  # LLM推理结果
    
    def get_all_roles(self) -> List[Role]:
        roles = [self.primary_role]
        roles.extend([r for r, _ in self.secondary_roles])
        return roles

@dataclass
class EntityRole:
    """实体级角色（类/函数）"""
    name: str
    type: str
    role: Role
    score: float
    lineno: int
    pattern: StructuralPattern

@dataclass
class FileAnalysis:
    """文件分析结果"""
    file_path: str
    role_score: RoleScore
    entity_roles: List[EntityRole] = field(default_factory=list)
    pattern: Optional[StructuralPattern] = None
    features: Dict[str, Any] = field(default_factory=dict)

class CodeRoleClassifier:
    """
    RAACS 3.0: 基于结构模式的代码角色分类器
    核心思想：通过代码的结构特征判断角色，而非依赖框架指纹
    """
    
    # 路径启发（保留，但降低权重）
    PATH_HINTS = {
        Role.TEST: ['test', 'tests', '__tests__', 'spec'],
        Role.CONFIG: ['config', 'conf', 'settings'],
        Role.SCRIPT: ['scripts', 'bin', 'tools', 'cli'],
        Role.ADAPTER: ['views', 'api', 'controllers', 'handlers'],
        Role.SCHEMA: ['models', 'schemas', 'entities'],
        Role.UTIL: ['utils', 'helpers', 'common', 'lib']
    }
    
    def __init__(self, debug: bool = False, use_llm: bool = False):
        self.debug = debug
        self.use_llm = use_llm
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """分析单个文件"""
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

        # 1. 提取基础统计特征
        features = self._extract_basic_features(tree, file_path)
        
        # 2. 分析结构模式（核心）
        pattern = self._analyze_structural_pattern(tree, features)
        
        # 3. 实体级分析
        entity_roles = self._analyze_entities(tree, features)
        
        # 4. 基于结构模式打分
        role_score = self._score_by_pattern(pattern, features, file_path)
        
        # 5. 可选：LLM 辅助判断
        if self.use_llm and role_score.primary_score < 3.0:  # 低置信度时调用LLM
            role_score = self._llm_assist(source_code, pattern, role_score)
        
        return FileAnalysis(
            file_path=file_path,
            role_score=role_score,
            entity_roles=entity_roles,
            pattern=pattern,
            features=features
        )

    def _extract_basic_features(self, tree: ast.AST, file_path: str) -> Dict[str, Any]:
        """提取基础统计特征（去框架指纹化）"""
        f = defaultdict(int)
        
        # 路径信息
        f['file_path'] = file_path
        f['filename'] = os.path.basename(file_path).lower() if file_path else ""
        f['dir_path'] = os.path.dirname(file_path).lower() if file_path else ""
        
        # 定义统计
        f['class_count'] = 0
        f['function_count'] = 0
        f['method_count'] = 0
        
        # 语句统计
        f['assign_count'] = 0
        f['expr_count'] = 0
        f['control_flow_count'] = 0
        f['call_count'] = 0
        
        # 特殊模式
        f['has_main_entry'] = False
        f['is_pure_init'] = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                f['class_count'] += 1
            elif isinstance(node, ast.FunctionDef):
                f['function_count'] += 1
            elif isinstance(node, ast.Assign):
                f['assign_count'] += 1
            elif isinstance(node, ast.Expr):
                f['expr_count'] += 1
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                if isinstance(node, ast.If) and self._is_main_check(node):
                    f['has_main_entry'] = True
                else:
                    f['control_flow_count'] += 1
            elif isinstance(node, ast.Call):
                f['call_count'] += 1
        
        # 判断纯__init__
        if f['filename'] == '__init__.py':
            f['is_pure_init'] = (f['function_count'] == 0 and 
                                f['class_count'] == 0 and 
                                f['control_flow_count'] == 0)
        
        return f

    def _analyze_structural_pattern(self, tree: ast.AST, features: Dict) -> StructuralPattern:
        """
        核心：基于结构模式识别角色，不依赖框架名称
        """
        pattern = StructuralPattern()
        
        # 统计各类特征
        total_funcs = 0
        abstract_funcs = 0      # 空实现/抽象方法
        data_classes = 0        # 数据类（字段多于方法）
        io_operations = 0       # IO操作特征
        assertions = 0          # 断言
        fixtures = 0            # 测试fixture模式
        pure_computations = 0   # 纯计算函数
        orchestrations = 0      # 编排式调用
        constants = 0           # 常量定义
        configurations = 0      # 配置模式
        
        for node in ast.walk(tree):
            # === 类级分析 ===
            if isinstance(node, ast.ClassDef):
                # 数据类识别：字段多于方法
                fields = sum(1 for n in node.body if isinstance(n, ast.AnnAssign))
                methods = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                
                if fields > methods and methods < 3:
                    data_classes += 1
                
                # 检查是否为异常类
                for base in node.bases:
                    base_name = self._get_name(base)
                    if 'Exception' in base_name or 'Error' in base_name:
                        data_classes += 1
            
            # === 函数级分析 ===
            elif isinstance(node, ast.FunctionDef):
                total_funcs += 1
                
                # 1. 抽象/空实现检测
                if self._is_empty_body(node.body):
                    abstract_funcs += 1
                elif self._has_not_implemented(node):
                    abstract_funcs += 1
                
                # 2. IO边界检测（通过行为模式，非框架名）
                if self._is_io_boundary_func(node):
                    io_operations += 1
                
                # 3. 测试特征检测
                if self._is_test_func(node):
                    assertions += sum(1 for n in ast.walk(node) if isinstance(n, ast.Assert))
                
                if self._is_fixture_func(node):
                    fixtures += 1
                
                # 4. 编排模式检测
                if self._is_orchestrator_func(node):
                    orchestrations += 1
                
                # 5. 纯计算检测
                if self._is_pure_computation(node):
                    pure_computations += 1
            
            # === 模块级赋值分析 ===
            elif isinstance(node, ast.Assign):
                # 检测常量模式
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():  # 大写常量
                            constants += 1
                        # 检测配置模式：特定类型的赋值
                        if self._is_config_assignment(node):
                            configurations += 1
        
        # === 计算结构指标 ===
        
        # 抽象比例
        if total_funcs > 0:
            pattern.abstraction_ratio = abstract_funcs / total_funcs
        
        # 数据定义比例
        total_defs = features['class_count'] + features['function_count']
        if total_defs > 0:
            pattern.data_definition_ratio = data_classes / features['class_count'] if features['class_count'] > 0 else 0
        
        # IO边界判断
        pattern.is_io_boundary = io_operations > 0 or self._has_request_response_pattern(tree)
        
        # 编排模式判断
        pattern.is_orchestrator = orchestrations > total_funcs * 0.3 if total_funcs > 0 else False
        
        # 断言密度
        if total_funcs > 0:
            pattern.assertion_density = assertions / total_funcs
            pattern.fixture_ratio = fixtures / total_funcs
        
        # 配置特征
        pattern.is_pure_declaration = (features['function_count'] == 0 and 
                                       features['class_count'] == 0 and
                                       features['assign_count'] > 0)
        if features['assign_count'] > 0:
            pattern.constant_ratio = constants / features['assign_count']
        
        # 脚本特征
        pattern.has_entry_point = features['has_main_entry']
        pattern.imperative_ratio = features['expr_count'] / (total_defs + 1)
        
        # 计算密度
        if total_funcs > 0:
            pattern.computation_intensity = features['control_flow_count'] / total_funcs
        
        return pattern

    def _is_io_boundary_func(self, node: ast.FunctionDef) -> bool:
        """
        判断函数是否为IO边界（通过结构特征，非框架名）
        特征：
        1. 参数名包含 request/response/context
        2. 返回类型包含 Response/Dict/JSON
        3. 函数内有序列化/反序列化操作
        """
        # 检查参数名
        param_names = [arg.arg.lower() for arg in node.args.args]
        io_keywords = ['request', 'response', 'req', 'resp', 'ctx', 'context', 'event', 'payload']
        if any(kw in param_names for kw in io_keywords):
            return True
        
        # 检查返回注解
        if node.returns:
            return_type = self._get_name(node.returns).lower()
            if any(kw in return_type for kw in ['response', 'dict', 'json', 'result']):
                return True
        
        # 检查函数体是否有序列化操作
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                func_name = self._get_name(n.func).lower()
                if any(kw in func_name for kw in ['serialize', 'deserialize', 'json', 'parse', 'validate']):
                    return True
        
        return False

    def _has_request_response_pattern(self, tree: ast.AST) -> bool:
        """检测请求-响应模式（通过类型注解和命名，非框架）"""
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign):
                if node.annotation:
                    type_name = self._get_name(node.annotation).lower()
                    if 'request' in type_name or 'response' in type_name:
                        return True
        return False

    def _is_test_func(self, node: ast.FunctionDef) -> bool:
        """判断是否为测试函数（通过命名和结构）"""
        return (node.name.startswith('test_') or 
                node.name.startswith('should_') or
                any('test' in self._get_name(d).lower() for d in node.decorator_list))

    def _is_fixture_func(self, node: ast.FunctionDef) -> bool:
        """判断是否为fixture（通过装饰器模式）"""
        for deco in node.decorator_list:
            deco_name = self._get_name(deco).lower()
            if 'fixture' in deco_name or 'setup' in deco_name or 'teardown' in deco_name:
                return True
        return False

    def _is_orchestrator_func(self, node: ast.FunctionDef) -> bool:
        """
        判断是否为编排函数（调用多个其他函数，自身逻辑少）
        """
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        control_flow = [n for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While))]
        
        # 调用多，逻辑少 -> 编排模式
        return len(calls) > 3 and len(control_flow) < 2

    def _is_pure_computation(self, node: ast.FunctionDef) -> bool:
        """
        判断是否为纯计算函数（无IO，无副作用）
        启发：无全局变量访问，无属性修改，只有计算和返回
        """
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        has_side_effect = any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node))
        
        return has_return and not has_side_effect

    def _is_config_assignment(self, node: ast.Assign) -> bool:
        """判断是否为配置式赋值（赋值右侧是字面量/环境变量读取）"""
        if isinstance(node.value, (ast.Constant, ast.Num, ast.Str, ast.List, ast.Dict)):
            return True
        if isinstance(node.value, ast.Call):
            func_name = self._get_name(node.value.func).lower()
            if 'getenv' in func_name or 'env' in func_name:
                return True
        return False

    def _has_not_implemented(self, node: ast.FunctionDef) -> bool:
        """检测 raise NotImplementedError"""
        for n in ast.walk(node):
            if isinstance(n, ast.Raise):
                if isinstance(n.exc, ast.Call):
                    if 'NotImplemented' in self._get_name(n.exc.func):
                        return True
                if isinstance(n.exc, ast.Name):
                    if 'NotImplemented' in n.exc.id:
                        return True
        return False

    def _score_by_pattern(self, pattern: StructuralPattern, 
                          features: Dict, file_path: str) -> RoleScore:
        """
        基于结构模式打分（去框架指纹化）
        """
        scores = defaultdict(float)
        
        # 1. TEST - 基于测试结构模式
        if pattern.assertion_density > 0.5:
            scores[Role.TEST] += 3.0
        if pattern.fixture_ratio > 0.3:
            scores[Role.TEST] += 2.0
        if features['filename'].startswith('test_') or 'test' in features['filename']:
            scores[Role.TEST] += 1.5
        if self._path_hint(file_path, Role.TEST):
            scores[Role.TEST] += 2.0
        
        # 2. SCRIPT - 基于入口点和命令式特征
        if pattern.has_entry_point:
            scores[Role.SCRIPT] += 3.5
        if pattern.imperative_ratio > 1.0:
            scores[Role.SCRIPT] += 1.5
        if self._path_hint(file_path, Role.SCRIPT):
            scores[Role.SCRIPT] += 2.0
        
        # 3. NAMESPACE - 纯__init__.py
        if features['is_pure_init']:
            scores[Role.NAMESPACE] += 5.0
        elif features['filename'] == '__init__.py':
            scores[Role.NAMESPACE] += 1.0
        
        # 4. CONFIG - 基于纯声明和常量比例
        if pattern.is_pure_declaration:
            scores[Role.CONFIG] += 3.0
        if pattern.constant_ratio > 0.5:
            scores[Role.CONFIG] += 2.0
        if self._path_hint(file_path, Role.CONFIG):
            scores[Role.CONFIG] += 2.0
        
        # 5. INTERFACE - 基于抽象比例
        if pattern.abstraction_ratio > 0.7:
            scores[Role.INTERFACE] += 3.5
        elif pattern.abstraction_ratio > 0.4:
            scores[Role.INTERFACE] += 2.0
        
        # 6. SCHEMA - 基于数据定义比例
        if pattern.data_definition_ratio > 0.7:
            scores[Role.SCHEMA] += 3.0
        elif pattern.data_definition_ratio > 0.4:
            scores[Role.SCHEMA] += 1.5
        if pattern.computation_intensity < 0.5:
            scores[Role.SCHEMA] += 1.0
        if self._path_hint(file_path, Role.SCHEMA):
            scores[Role.SCHEMA] += 1.5
        
        # 7. ADAPTER - 基于IO边界模式（不看框架名）
        if pattern.is_io_boundary:
            scores[Role.ADAPTER] += 3.0
        if pattern.is_orchestrator:
            scores[Role.ADAPTER] += 2.0
        if self._path_hint(file_path, Role.ADAPTER):
            scores[Role.ADAPTER] += 2.0
        
        # 8. UTIL - 基于纯函数和低复杂度
        if features['function_count'] > 0 and features['class_count'] == 0:
            if pattern.computation_intensity < 2.0:
                scores[Role.UTIL] += 2.0
        if self._path_hint(file_path, Role.UTIL):
            scores[Role.UTIL] += 2.0
        
        # 9. LOGIC - 兜底
        if features['function_count'] > 0 or features['class_count'] > 0:
            scores[Role.LOGIC] += 1.0
        if pattern.computation_intensity > 2.0:
            scores[Role.LOGIC] += 1.5
        
        # 选择最佳角色
        if not scores:
            return RoleScore(Role.UNKNOWN, 0.0)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_role, primary_score = sorted_scores[0]
        
        threshold = primary_score * 0.5
        secondary_roles = [(role, score) for role, score in sorted_scores[1:] 
                          if score > threshold][:2]
        
        return RoleScore(
            primary_role=primary_role,
            primary_score=primary_score,
            secondary_roles=secondary_roles
        )

    def _path_hint(self, file_path: str, role: Role) -> bool:
        """路径提示（降低权重，仅作辅助）"""
        if not file_path or role not in self.PATH_HINTS:
            return False
        path_lower = file_path.lower()
        return any(hint in path_lower for hint in self.PATH_HINTS[role])

    def _llm_assist(self, source_code: str, pattern: StructuralPattern, 
                    initial_score: RoleScore) -> RoleScore:
        """
        LLM辅助判断（当规则置信度低时）
        这里提供接口，实际实现需要调用LLM API
        """
        # TODO: 实现LLM调用
        # 1. 构造prompt: 代码片段 + 结构模式 + 初步判断
        # 2. 让LLM基于架构知识推理角色
        # 3. 返回更新的RoleScore，包含reasoning
        
        # 暂时返回原始结果
        initial_score.reasoning = "LLM assist not implemented"
        return initial_score

    def _analyze_entities(self, tree: ast.AST, features: Dict) -> List[EntityRole]:
        """实体级分析（简化版）"""
        entities = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                pattern = self._analyze_class_pattern(node)
                role, score = self._classify_by_pattern(pattern)
                entities.append(EntityRole(
                    name=node.name,
                    type='class',
                    role=role,
                    score=score,
                    lineno=node.lineno,
                    pattern=pattern
                ))
        return entities

    def _analyze_class_pattern(self, node: ast.ClassDef) -> StructuralPattern:
        """分析单个类的结构模式"""
        pattern = StructuralPattern()
        
        fields = sum(1 for n in node.body if isinstance(n, ast.AnnAssign))
        methods = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
        
        if methods > 0:
            empty_methods = sum(1 for n in node.body 
                              if isinstance(n, ast.FunctionDef) and self._is_empty_body(n.body))
            pattern.abstraction_ratio = empty_methods / methods
        
        if fields + methods > 0:
            pattern.data_definition_ratio = fields / (fields + methods)
        
        return pattern

    def _classify_by_pattern(self, pattern: StructuralPattern) -> Tuple[Role, float]:
        """基于模式分类单个实体"""
        if pattern.abstraction_ratio > 0.7:
            return Role.INTERFACE, 3.0
        if pattern.data_definition_ratio > 0.7:
            return Role.SCHEMA, 3.0
        return Role.LOGIC, 1.0

    # === Helpers ===
    def _get_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return self._get_name(node.func)
        return ""

    def _is_empty_body(self, body):
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt, (ast.Pass, ast.Ellipsis)):
                continue
            return False
        return True

    def _is_main_check(self, node: ast.If):
        try:
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    return True
        except:
            pass
        return False

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAACS 3.0 Pattern-based Classifier")
    parser.add_argument("path", help="Path to analyze")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--entities", action="store_true", help="Show entity-level roles")
    parser.add_argument("--llm", action="store_true", help="Enable LLM assistance")
    parser.add_argument("--pattern", action="store_true", help="Show structural patterns")
    args = parser.parse_args()

    classifier = CodeRoleClassifier(debug=args.debug, use_llm=args.llm)
    
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
        print(f"Primary: {COLORS[result.role_score.primary_role]}{result.role_score.primary_role.value}{RESET} ({result.role_score.primary_score:.2f})")
        
        if result.role_score.secondary_roles:
            print(f"Secondary: ", end="")
            for role, score in result.role_score.secondary_roles:
                print(f"{COLORS[role]}{role.value}{RESET} ({score:.2f}) ", end="")
            print()
        
        if args.pattern and result.pattern:
            p = result.pattern
            print("\n=== Structural Pattern ===")
            print(f"IO Boundary: {p.is_io_boundary}")
            print(f"Orchestrator: {p.is_orchestrator}")
            print(f"Abstraction: {p.abstraction_ratio:.2f}")
            print(f"Data Definition: {p.data_definition_ratio:.2f}")
            print(f"Assertion Density: {p.assertion_density:.2f}")
            print(f"Constant Ratio: {p.constant_ratio:.2f}")
    else:
        print(f"{'Path':<50} | {'Primary':<15} | {'Secondary':<20}")
        print("-" * 90)
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if f.endswith('.py'):
                    full_path = os.path.join(root, f)
                    result = classifier.analyze_file(full_path)
                    rel_path = os.path.relpath(full_path, target)
                    
                    primary = result.role_score.primary_role
                    secondary_str = ", ".join([r.value for r, _ in result.role_score.secondary_roles])
                    
                    print(f"{rel_path:<50} | {COLORS[primary]}{primary.value:<15}{RESET} | {secondary_str:<20}")