"""
RAACS 4.0: Role-Aware Adaptive Context System - Code Role Classifier v7

核心改进：
1. 三层信号加权：框架指纹(4.0) > 结构模式(2.5) > 路径提示(1.5)
2. 实体级分析：类和顶层函数都参与角色判定
3. 混合度指标：基于角色兼容性矩阵计算
4. 权重定义：0.6×AST复杂度 + 0.4×LOC
5. 工厂类区分：便捷工厂(UTIL) vs 策略工厂(LOGIC)
6. SCHEMA 识别增强：同时统计 Assign 和 AnnAssign
"""

import ast
import os
import argparse
from collections import defaultdict
from typing import Set, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class Role(Enum):
    """代码角色枚举 - 9个角色对应9种 Morphing 策略"""
    TEST = "TEST"             # 验证层 - 通常忽略或极简处理
    NAMESPACE = "NAMESPACE"   # 命名空间层 - 仅保留导出列表
    INTERFACE = "INTERFACE"   # 抽象契约层 - 保留完整接口定义
    SCHEMA = "SCHEMA"         # 数据契约层 - 保留字段定义
    ADAPTER = "ADAPTER"       # 适配/入口层 - 骨架化
    CONFIG = "CONFIG"         # 配置层 - 保留键值
    SCRIPT = "SCRIPT"         # 脚本层 - 通常忽略
    UTIL = "UTIL"             # 无状态工具层 - 签名+用例
    LOGIC = "LOGIC"           # 核心逻辑层 - 保留骨架+关键逻辑
    UNKNOWN = "UNKNOWN"


# ============================================================================
# 角色兼容性矩阵 - 用于计算混合度
# 值越高，混用的代价越低
# ============================================================================
ROLE_COMPATIBILITY = {
    # 高兼容组：数据定义类
    (Role.SCHEMA, Role.CONFIG): 0.9,
    (Role.SCHEMA, Role.INTERFACE): 0.8,
    (Role.CONFIG, Role.INTERFACE): 0.7,
    
    # 中兼容组：行为类
    (Role.UTIL, Role.LOGIC): 0.5,
    (Role.ADAPTER, Role.LOGIC): 0.4,
    (Role.ADAPTER, Role.UTIL): 0.5,
    
    # 低兼容组：跨类型
    (Role.SCHEMA, Role.LOGIC): 0.2,
    (Role.SCHEMA, Role.UTIL): 0.3,
    (Role.SCHEMA, Role.ADAPTER): 0.2,
    (Role.CONFIG, Role.LOGIC): 0.3,
    (Role.INTERFACE, Role.LOGIC): 0.4,
    
    # 特殊角色
    (Role.TEST, Role.LOGIC): 0.1,
    (Role.SCRIPT, Role.LOGIC): 0.3,
    (Role.NAMESPACE, Role.SCHEMA): 0.5,
}


# ============================================================================
# 框架指纹库 - 强信号 (权重 4.0)
# ============================================================================
FRAMEWORK_SIGNATURES = {
    # Web 框架 -> ADAPTER (注意：django 需要细粒度处理)
    'web_frameworks': {
        'imports': {'flask', 'fastapi', 'starlette', 'aiohttp', 'tornado', 'bottle', 'falcon', 'sanic'},
        'decorators': {'route', 'get', 'post', 'put', 'delete', 'patch', 'api_view', 'action', 
                      'login_required', 'permission_classes', 'throttle_classes'},
    },
    # 数据框架 -> SCHEMA
    'data_frameworks': {
        'imports': {'pydantic', 'sqlalchemy', 'marshmallow', 'attrs', 'tortoise', 'peewee', 'mongoengine'},
        'base_classes': {'BaseModel', 'Model', 'Schema', 'Document', 'Entity', 'ModelForm', 'Form'},
        'decorators': {'dataclass', 'dataclasses.dataclass'},
    },
    # 测试框架 -> TEST
    'test_frameworks': {
        'imports': {'pytest', 'unittest', 'nose', 'hypothesis', 'mock', 'faker'},
        'base_classes': {'TestCase', 'TestSuite'},
        'decorators': {'fixture', 'mark', 'parametrize', 'patch', 'mock'},
    },
    # 异步/任务框架 -> ADAPTER (特殊的 IO 边界)
    'async_frameworks': {
        'imports': {'celery', 'rq', 'dramatiq', 'huey'},
        'decorators': {'task', 'shared_task', 'job'},
    },
}

# Django 子模块细粒度角色映射
DJANGO_SUBMODULE_ROLES = {
    # SCHEMA 类
    'django.forms': Role.SCHEMA,
    'django.db.models': Role.SCHEMA,
    'django.db': Role.SCHEMA,
    # ADAPTER 类
    'django.views': Role.ADAPTER,
    'django.urls': Role.ADAPTER,
    'django.contrib.admin': Role.ADAPTER,
    'django.middleware': Role.ADAPTER,
    'django.http': Role.ADAPTER,
    # CONFIG 类
    'django.conf': Role.CONFIG,
    'django.apps': Role.CONFIG,
}


# ============================================================================
# 信号权重配置
# ============================================================================
class SignalWeight:
    FRAMEWORK = 4.0      # 框架指纹
    STRUCTURE = 2.5      # 结构模式
    PATH_HINT = 1.5      # 路径提示
    NAME_HINT = 1.0      # 命名提示


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class StructuralFeatures:
    """结构特征"""
    # 基础统计
    class_count: int = 0
    function_count: int = 0
    method_count: int = 0
    
    # 数据特征
    field_count: int = 0           # 字段数量 (AnnAssign + Assign)
    typed_field_count: int = 0     # 带类型注解的字段
    
    # 模块级赋值
    module_assign_count: int = 0   # 顶层赋值语句数量
    
    # 行为特征
    abstract_method_count: int = 0
    empty_method_count: int = 0
    io_boundary_count: int = 0
    orchestrator_count: int = 0
    pure_function_count: int = 0
    
    # 控制流
    control_flow_count: int = 0
    call_count: int = 0
    
    # 测试特征
    assertion_count: int = 0
    test_function_count: int = 0
    fixture_count: int = 0
    
    # 配置特征
    constant_count: int = 0        # 大写变量
    config_assign_count: int = 0   # 配置式赋值
    
    # 特殊标记
    has_main_entry: bool = False
    is_pure_init: bool = False
    has_exception_class: bool = False


@dataclass
class FrameworkSignals:
    """框架指纹信号"""
    web_framework_imports: Set[str] = field(default_factory=set)
    data_framework_imports: Set[str] = field(default_factory=set)
    test_framework_imports: Set[str] = field(default_factory=set)
    async_framework_imports: Set[str] = field(default_factory=set)
    
    web_decorators: Set[str] = field(default_factory=set)
    data_decorators: Set[str] = field(default_factory=set)
    test_decorators: Set[str] = field(default_factory=set)
    
    data_base_classes: Set[str] = field(default_factory=set)
    test_base_classes: Set[str] = field(default_factory=set)
    
    # Django 细粒度信号
    django_schema_imports: Set[str] = field(default_factory=set)  # django.forms, django.db.models
    django_adapter_imports: Set[str] = field(default_factory=set)  # django.views, django.urls
    django_config_imports: Set[str] = field(default_factory=set)  # django.conf, django.apps


@dataclass
class EntityRole:
    """实体级角色（类/函数）"""
    name: str
    entity_type: str          # 'class' | 'function'
    role: Role
    confidence: float
    weight: float             # 混合权重
    lineno: int
    reasoning: str = ""


@dataclass
class RoleScore:
    """角色打分结果"""
    primary_role: Role
    primary_score: float
    all_scores: Dict[Role, float] = field(default_factory=dict)
    reasoning: str = ""
    
    @property
    def secondary_roles(self) -> List[Tuple[Role, float]]:
        """获取次要角色（得分超过主角色50%的）"""
        threshold = self.primary_score * 0.5
        return [
            (role, score) for role, score in self.all_scores.items()
            if role != self.primary_role and score > threshold
        ][:2]


@dataclass
class FileAnalysis:
    """文件分析结果"""
    file_path: str
    role_score: RoleScore
    entities: List[EntityRole] = field(default_factory=list)
    structural_features: Optional[StructuralFeatures] = None
    framework_signals: Optional[FrameworkSignals] = None
    
    @property
    def role_purity(self) -> float:
        """计算考虑兼容性的有效纯度"""
        if not self.entities:
            return 1.0
        
        role_weights = defaultdict(float)
        for e in self.entities:
            role_weights[e.role] += e.weight
        
        total = sum(role_weights.values())
        if total == 0:
            return 1.0
            
        primary_weight = role_weights[self.role_score.primary_role]
        base_purity = primary_weight / total
        
        # 兼容性加成
        compat_bonus = 0
        for role, weight in role_weights.items():
            if role != self.role_score.primary_role:
                compat = get_role_compatibility(self.role_score.primary_role, role)
                compat_bonus += (weight / total) * compat * 0.2
        
        return min(base_purity + compat_bonus, 1.0)
    
    @property
    def use_file_level_morphing(self) -> bool:
        """是否使用文件级 Morphing（vs 实体级）"""
        return self.role_purity > 0.75


def get_role_compatibility(role1: Role, role2: Role) -> float:
    """获取两个角色的兼容度"""
    if role1 == role2:
        return 1.0
    key = (role1, role2)
    if key in ROLE_COMPATIBILITY:
        return ROLE_COMPATIBILITY[key]
    key = (role2, role1)
    if key in ROLE_COMPATIBILITY:
        return ROLE_COMPATIBILITY[key]
    return 0.3  # 默认中低兼容


# ============================================================================
# 核心分类器
# ============================================================================

class CodeRoleClassifier:
    """
    RAACS 4.0: 基于三层信号的代码角色分类器
    
    信号层级：
    1. 框架指纹 (权重 4.0) - 最强信号，直接命中
    2. 结构模式 (权重 2.5) - 语义特征分析
    3. 路径提示 (权重 1.5) - 辅助判断
    """
    
    PATH_HINTS = {
        Role.TEST: ['test', 'tests', '__tests__', 'spec', 'specs'],
        Role.CONFIG: ['config', 'conf', 'settings', 'constants'],
        Role.SCRIPT: ['scripts', 'bin', 'tools', 'cli', 'commands'],
        Role.ADAPTER: ['views', 'api', 'controllers', 'handlers', 'routes', 'endpoints'],
        Role.SCHEMA: ['models', 'schemas', 'entities', 'types', 'dto'],
        Role.UTIL: ['utils', 'helpers', 'common', 'lib', 'shared'],
    }
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """分析单个文件"""
        if not os.path.exists(file_path):
            return self._empty_result(file_path, "File not found")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self.analyze_source(source_code, file_path=file_path)
        except SyntaxError as e:
            if self.debug:
                print(f"[SyntaxError] {file_path}: {e}")
            return self._empty_result(file_path, f"Syntax error: {e}")
        except Exception as e:
            if self.debug:
                print(f"[Error] {file_path}: {e}")
            return self._empty_result(file_path, f"Error: {e}")
    
    def analyze_source(self, source_code: str, file_path: str = "") -> FileAnalysis:
        """核心分析流水线"""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return self._empty_result(file_path, "Syntax error")
        
        filename = os.path.basename(file_path).lower() if file_path else ""
        
        # 1. 提取框架指纹信号
        framework_signals = self._extract_framework_signals(tree)
        
        # 2. 提取结构特征
        structural_features = self._extract_structural_features(tree, filename)
        
        # 3. 实体级分析（类和顶层函数）
        entities = self._analyze_entities(tree, source_code, framework_signals)
        
        # 4. 综合打分
        role_score = self._compute_role_score(
            framework_signals, 
            structural_features, 
            entities,
            file_path
        )
        
        return FileAnalysis(
            file_path=file_path,
            role_score=role_score,
            entities=entities,
            structural_features=structural_features,
            framework_signals=framework_signals
        )
    
    # ========================================================================
    # 框架指纹提取
    # ========================================================================
    
    def _extract_framework_signals(self, tree: ast.AST) -> FrameworkSignals:
        """提取框架指纹信号"""
        signals = FrameworkSignals()
        
        for node in ast.walk(tree):
            # 导入分析
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        full_module = alias.name
                        top_module = full_module.split('.')[0]
                        self._classify_import(top_module, signals, full_module)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    full_module = node.module
                    top_module = full_module.split('.')[0]
                    self._classify_import(top_module, signals, full_module)
            
            # 类分析
            elif isinstance(node, ast.ClassDef):
                # 基类
                for base in node.bases:
                    base_name = self._get_name(base)
                    if base_name in FRAMEWORK_SIGNATURES['data_frameworks'].get('base_classes', set()):
                        signals.data_base_classes.add(base_name)
                    if base_name in FRAMEWORK_SIGNATURES['test_frameworks'].get('base_classes', set()):
                        signals.test_base_classes.add(base_name)
                
                # 装饰器
                for deco in node.decorator_list:
                    deco_name = self._get_name(deco)
                    self._classify_decorator(deco_name, signals)
            
            # 函数装饰器
            elif isinstance(node, ast.FunctionDef):
                for deco in node.decorator_list:
                    deco_name = self._get_name(deco)
                    self._classify_decorator(deco_name, signals)
        
        return signals
    
    def _classify_import(self, module: str, signals: FrameworkSignals, full_module: str = ""):
        """分类导入
        
        Args:
            module: 顶级模块名 (e.g., 'django')
            signals: 框架信号对象
            full_module: 完整模块路径 (e.g., 'django.forms')
        """
        # Django 细粒度处理
        if module == 'django' and full_module:
            for django_path, role in DJANGO_SUBMODULE_ROLES.items():
                if full_module.startswith(django_path):
                    if role == Role.SCHEMA:
                        signals.django_schema_imports.add(full_module)
                    elif role == Role.ADAPTER:
                        signals.django_adapter_imports.add(full_module)
                    elif role == Role.CONFIG:
                        signals.django_config_imports.add(full_module)
                    return  # 匹配到细粒度规则，不再走通用规则
            # Django 未匹配到细粒度规则，默认为 ADAPTER
            signals.django_adapter_imports.add(full_module)
            return
        
        # 通用框架处理
        if module in FRAMEWORK_SIGNATURES['web_frameworks']['imports']:
            signals.web_framework_imports.add(module)
        if module in FRAMEWORK_SIGNATURES['data_frameworks']['imports']:
            signals.data_framework_imports.add(module)
        if module in FRAMEWORK_SIGNATURES['test_frameworks']['imports']:
            signals.test_framework_imports.add(module)
        if module in FRAMEWORK_SIGNATURES['async_frameworks']['imports']:
            signals.async_framework_imports.add(module)
    
    def _classify_decorator(self, deco_name: str, signals: FrameworkSignals):
        """分类装饰器"""
        deco_lower = deco_name.lower()
        if deco_lower in FRAMEWORK_SIGNATURES['web_frameworks']['decorators']:
            signals.web_decorators.add(deco_name)
        if deco_lower in FRAMEWORK_SIGNATURES['data_frameworks']['decorators']:
            signals.data_decorators.add(deco_name)
        if deco_lower in FRAMEWORK_SIGNATURES['test_frameworks']['decorators']:
            signals.test_decorators.add(deco_name)
    
    # ========================================================================
    # 结构特征提取
    # ========================================================================
    
    def _extract_structural_features(self, tree: ast.AST, filename: str) -> StructuralFeatures:
        """提取结构特征"""
        f = StructuralFeatures()
        
        # 先统计顶层赋值（模块级）
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                f.module_assign_count += 1
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                f.class_count += 1
                
                # 检查异常类
                for base in node.bases:
                    base_name = self._get_name(base)
                    if 'Exception' in base_name or 'Error' in base_name:
                        f.has_exception_class = True
                
                # 统计类内部
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        f.method_count += 1
                        if self._is_empty_body(item.body):
                            f.empty_method_count += 1
                        if self._has_not_implemented(item):
                            f.abstract_method_count += 1
                    elif isinstance(item, ast.AnnAssign):
                        f.typed_field_count += 1
                        f.field_count += 1
                    elif isinstance(item, ast.Assign):
                        f.field_count += 1
            
            elif isinstance(node, ast.FunctionDef):
                # 只统计顶层函数
                if self._is_top_level_function(tree, node):
                    f.function_count += 1
                    
                    if node.name.startswith('test_'):
                        f.test_function_count += 1
                    
                    if self._is_fixture_func(node):
                        f.fixture_count += 1
                    
                    if self._is_io_boundary_func(node):
                        f.io_boundary_count += 1
                    
                    if self._is_orchestrator_func(node):
                        f.orchestrator_count += 1
                    
                    if self._is_pure_function(node):
                        f.pure_function_count += 1
            
            elif isinstance(node, ast.If):
                if self._is_main_check(node):
                    f.has_main_entry = True
                else:
                    f.control_flow_count += 1
            
            elif isinstance(node, (ast.For, ast.While, ast.Try)):
                f.control_flow_count += 1
            
            elif isinstance(node, ast.Call):
                f.call_count += 1
            
            elif isinstance(node, ast.Assert):
                f.assertion_count += 1
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():
                            f.constant_count += 1
                        if self._is_config_assignment(node):
                            f.config_assign_count += 1
        
        # 判断纯 __init__.py
        if filename == '__init__.py':
            f.is_pure_init = (
                f.function_count == 0 and 
                f.class_count == 0 and 
                f.control_flow_count == 0
            )
        
        return f
    
    def _is_top_level_function(self, tree: ast.AST, func_node: ast.FunctionDef) -> bool:
        """判断是否为顶层函数"""
        for node in tree.body:
            if node is func_node:
                return True
        return False
    
    # ========================================================================
    # 实体级分析
    # ========================================================================
    
    def _analyze_entities(self, tree: ast.AST, source_code: str, 
                          framework_signals: FrameworkSignals) -> List[EntityRole]:
        """分析实体级角色（类和顶层函数）"""
        entities = []
        source_lines = source_code.split('\n')
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                entity = self._analyze_class_entity(node, source_lines, framework_signals)
                entities.append(entity)
            
            elif isinstance(node, ast.FunctionDef):
                entity = self._analyze_function_entity(node, source_lines, framework_signals)
                entities.append(entity)
        
        return entities
    
    def _analyze_class_entity(self, node: ast.ClassDef, source_lines: List[str],
                               framework_signals: FrameworkSignals) -> EntityRole:
        """分析类实体的角色"""
        # 计算权重
        weight = self._calculate_entity_weight(node, source_lines)
        
        # 统计特征
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        # 统计字段：包括 AnnAssign 和 Assign（排除 __xxx__ 形式的私有属性定义）
        fields = []
        for n in node.body:
            if isinstance(n, ast.AnnAssign):
                fields.append(n)
            elif isinstance(n, ast.Assign):
                # 排除 __slots__ 等魔术属性
                for target in n.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith('__'):
                        fields.append(n)
                        break
        
        # 排除 __init__ 和其他魔术方法，计算"业务方法"数量
        non_magic_methods = [m for m in methods if not m.name.startswith('__')]
        init_only = len(methods) == 1 and methods[0].name == '__init__'
        
        empty_methods = sum(1 for m in methods if self._is_empty_body(m.body))
        abstract_methods = sum(1 for m in methods if self._has_not_implemented(m))
        
        # 检查基类
        base_names = [self._get_name(b) for b in node.bases]
        is_exception = any('Exception' in b or 'Error' in b for b in base_names)
        is_data_class = any(b in FRAMEWORK_SIGNATURES['data_frameworks'].get('base_classes', set()) for b in base_names)
        is_test_class = any(b in FRAMEWORK_SIGNATURES['test_frameworks'].get('base_classes', set()) for b in base_names)
        
        # 检查是否继承自可能是数据基类的类（启发式）
        data_base_hints = ['entity', 'model', 'schema', 'dto', 'mixin', 'base']
        inherits_data_base = any(
            any(hint in b.lower() for hint in data_base_hints) 
            for b in base_names
        )
        
        # 检查装饰器
        deco_names = [self._get_name(d) for d in node.decorator_list]
        has_dataclass = any('dataclass' in d.lower() for d in deco_names)
        
        # 判断是否为"空壳数据类"（只有 pass 或只有 __init__，但继承自数据基类）
        is_empty_data_class = (
            (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)) or
            (init_only and len(fields) == 0)
        ) and (inherits_data_base or is_data_class)
        
        # 判断角色
        role = Role.LOGIC
        confidence = 0.5
        reasoning = ""
        
        # 1. 框架指纹判断（最高优先级）
        if is_test_class or 'test' in node.name.lower():
            role = Role.TEST
            confidence = 0.95
            reasoning = "Test class (framework signal)"
        elif is_data_class or has_dataclass:
            role = Role.SCHEMA
            confidence = 0.9
            reasoning = "Data class (framework signal)"
        elif is_exception:
            role = Role.SCHEMA
            confidence = 0.85
            reasoning = "Exception class -> SCHEMA"
        elif is_empty_data_class:
            role = Role.SCHEMA
            confidence = 0.8
            reasoning = f"Empty data class (inherits from {', '.join(base_names[:2])})"
        
        # 2. 结构模式判断
        elif len(methods) > 0 and (empty_methods + abstract_methods) / len(methods) > 0.7:
            role = Role.INTERFACE
            confidence = 0.85
            reasoning = f"Abstract class ({empty_methods + abstract_methods}/{len(methods)} abstract methods)"
        elif len(fields) > 0 and len(fields) >= len(non_magic_methods):
            # 字段数 >= 非魔术方法数 -> 倾向于数据类
            role = Role.SCHEMA
            confidence = 0.75 if inherits_data_base else 0.65
            reasoning = f"Data class ({len(fields)} fields >= {len(non_magic_methods)} methods)"
        elif init_only and inherits_data_base:
            # 只有 __init__，但继承自数据基类
            role = Role.SCHEMA
            confidence = 0.7
            reasoning = f"Data class (init-only, inherits {base_names[0] if base_names else 'base'})"
        elif self._is_factory_class(node):
            # 区分便捷工厂和策略工厂
            if self._has_business_logic(node):
                role = Role.LOGIC
                confidence = 0.7
                reasoning = "Strategy factory (has business logic)"
            else:
                role = Role.UTIL
                confidence = 0.7
                reasoning = "Convenience factory"
        
        return EntityRole(
            name=node.name,
            entity_type='class',
            role=role,
            confidence=confidence,
            weight=weight,
            lineno=node.lineno,
            reasoning=reasoning
        )
    
    def _analyze_function_entity(self, node: ast.FunctionDef, source_lines: List[str],
                                  framework_signals: FrameworkSignals) -> EntityRole:
        """分析函数实体的角色"""
        weight = self._calculate_entity_weight(node, source_lines)
        
        role = Role.LOGIC
        confidence = 0.5
        reasoning = ""
        
        # 检查装饰器
        deco_names = [self._get_name(d).lower() for d in node.decorator_list]
        
        # 1. 框架指纹判断
        if node.name.startswith('test_') or any('test' in d for d in deco_names):
            role = Role.TEST
            confidence = 0.95
            reasoning = "Test function"
        elif any(d in deco_names for d in ['fixture', 'setup', 'teardown']):
            role = Role.TEST
            confidence = 0.9
            reasoning = "Test fixture"
        elif any(d in FRAMEWORK_SIGNATURES['web_frameworks']['decorators'] for d in deco_names):
            role = Role.ADAPTER
            confidence = 0.9
            reasoning = f"Web route decorator ({', '.join(deco_names)})"
        elif any(d in ['task', 'shared_task', 'job'] for d in deco_names):
            role = Role.ADAPTER
            confidence = 0.85
            reasoning = "Async task decorator"
        
        # 2. 结构模式判断
        elif self._is_io_boundary_func(node):
            role = Role.ADAPTER
            confidence = 0.75
            reasoning = "IO boundary function"
        elif self._is_decorator_definition(node):
            role = Role.UTIL
            confidence = 0.8
            reasoning = "Decorator definition"
        elif self._is_pure_function(node) and not self._has_business_logic_func(node):
            role = Role.UTIL
            confidence = 0.7
            reasoning = "Pure utility function"
        elif self._is_orchestrator_func(node):
            role = Role.LOGIC
            confidence = 0.7
            reasoning = "Orchestrator function"
        
        return EntityRole(
            name=node.name,
            entity_type='function',
            role=role,
            confidence=confidence,
            weight=weight,
            lineno=node.lineno,
            reasoning=reasoning
        )
    
    def _is_factory_class(self, node: ast.ClassDef) -> bool:
        """判断是否为工厂类"""
        # 类名包含 Factory
        if 'factory' in node.name.lower():
            return True
        # 方法名多为 create_xxx / make_xxx / build_xxx
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        factory_methods = sum(1 for m in methods if any(
            m.name.startswith(p) for p in ['create', 'make', 'build', 'get_']
        ))
        return factory_methods > len(methods) * 0.5
    
    def _has_business_logic(self, node: ast.ClassDef) -> bool:
        """判断类是否包含业务逻辑"""
        for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
            if self._has_business_logic_func(method):
                return True
        return False
    
    def _has_business_logic_func(self, node: ast.FunctionDef) -> bool:
        """判断函数是否包含业务逻辑"""
        # 有控制流分支
        branches = sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.Match)))
        if branches >= 2:
            return True
        
        # 读取配置/环境
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                func_name = self._get_name(n.func).lower()
                if any(kw in func_name for kw in ['getenv', 'config', 'settings']):
                    return True
        
        return False
    
    def _is_decorator_definition(self, node: ast.FunctionDef) -> bool:
        """判断是否为装饰器定义"""
        # 返回函数的函数
        for n in ast.walk(node):
            if isinstance(n, ast.Return):
                if isinstance(n.value, ast.Name):
                    # 检查返回的是内部定义的函数
                    inner_funcs = [f.name for f in ast.walk(node) if isinstance(f, ast.FunctionDef) and f is not node]
                    if n.value.id in inner_funcs:
                        return True
        return False
    
    def _calculate_entity_weight(self, node: ast.AST, source_lines: List[str]) -> float:
        """计算实体权重 = 0.6×AST复杂度 + 0.4×LOC"""
        # AST 复杂度
        ast_nodes = sum(1 for _ in ast.walk(node))
        max_depth = self._get_max_depth(node)
        ast_complexity = ast_nodes * (1 + 0.1 * max_depth)
        
        # LOC
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line) or start_line
        loc = end_line - start_line + 1
        
        # 归一化
        normalized_ast = min(ast_complexity / 500, 1.0)
        normalized_loc = min(loc / 200, 1.0)
        
        return 0.6 * normalized_ast + 0.4 * normalized_loc
    
    def _get_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """获取最大嵌套深度"""
        max_child_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth
    
    # ========================================================================
    # 综合打分
    # ========================================================================
    
    def _compute_role_score(self, framework_signals: FrameworkSignals,
                            structural_features: StructuralFeatures,
                            entities: List[EntityRole],
                            file_path: str) -> RoleScore:
        """综合三层信号计算最终角色得分"""
        scores = defaultdict(float)
        reasoning_parts = []
        
        filename = os.path.basename(file_path).lower() if file_path else ""
        
        # ====== Layer 1: 框架指纹 (权重 4.0) ======
        
        # TEST
        if framework_signals.test_framework_imports or framework_signals.test_base_classes:
            scores[Role.TEST] += SignalWeight.FRAMEWORK
            reasoning_parts.append(f"Test framework: {framework_signals.test_framework_imports}")
        
        # ADAPTER (Web，非 Django)
        if framework_signals.web_framework_imports or framework_signals.web_decorators:
            scores[Role.ADAPTER] += SignalWeight.FRAMEWORK
            reasoning_parts.append(f"Web framework: {framework_signals.web_framework_imports}")
        
        # ADAPTER (Async)
        if framework_signals.async_framework_imports:
            scores[Role.ADAPTER] += SignalWeight.FRAMEWORK * 0.8
            reasoning_parts.append(f"Async framework: {framework_signals.async_framework_imports}")
        
        # Django 细粒度处理
        if framework_signals.django_schema_imports:
            scores[Role.SCHEMA] += SignalWeight.FRAMEWORK
            reasoning_parts.append(f"Django schema: {framework_signals.django_schema_imports}")
        if framework_signals.django_adapter_imports:
            scores[Role.ADAPTER] += SignalWeight.FRAMEWORK
            reasoning_parts.append(f"Django adapter: {framework_signals.django_adapter_imports}")
        if framework_signals.django_config_imports:
            scores[Role.CONFIG] += SignalWeight.FRAMEWORK * 0.5
            reasoning_parts.append(f"Django config: {framework_signals.django_config_imports}")
        
        # SCHEMA (数据框架)
        if framework_signals.data_framework_imports or framework_signals.data_base_classes or framework_signals.data_decorators:
            scores[Role.SCHEMA] += SignalWeight.FRAMEWORK
            reasoning_parts.append(f"Data framework: {framework_signals.data_framework_imports}")
        
        # ====== Layer 2: 结构模式 (权重 2.5) ======
        
        f = structural_features
        total_funcs = f.function_count + f.method_count
        
        # TEST
        if f.test_function_count > 0:
            scores[Role.TEST] += SignalWeight.STRUCTURE * (f.test_function_count / max(f.function_count, 1))
        if f.assertion_count > 0 and total_funcs > 0:
            assertion_density = f.assertion_count / total_funcs
            if assertion_density > 0.5:
                scores[Role.TEST] += SignalWeight.STRUCTURE
        
        # SCRIPT
        # 修复：如果有 __main__ 但同时有多个类/函数定义，降低 SCRIPT 权重
        # 因为这些文件主要功能是定义库，__main__ 只是调试入口
        if f.has_main_entry:
            entity_count = f.class_count + f.function_count
            if entity_count > 3:
                # 库文件带调试入口，SCRIPT 权重降低
                scores[Role.SCRIPT] += SignalWeight.STRUCTURE * 0.3
                reasoning_parts.append("Has __main__ entry (lib with debug)")
            else:
                # 真正的脚本文件
                scores[Role.SCRIPT] += SignalWeight.STRUCTURE * 1.5
                reasoning_parts.append("Has __main__ entry (script)")
        
        # NAMESPACE
        if f.is_pure_init:
            scores[Role.NAMESPACE] += SignalWeight.STRUCTURE * 2
            reasoning_parts.append("Pure __init__.py")
        elif filename == '__init__.py':
            scores[Role.NAMESPACE] += SignalWeight.STRUCTURE * 0.5
        
        # CONFIG
        if f.function_count == 0 and f.class_count == 0:
            if f.constant_count > 0 or f.config_assign_count > 0:
                scores[Role.CONFIG] += SignalWeight.STRUCTURE
                reasoning_parts.append(f"Pure config ({f.constant_count} constants)")
            elif f.module_assign_count > 0:
                # 简单模块赋值（如 _i18n.py, version.py）
                # 没有大写常量，但有赋值语句，归为 CONFIG
                scores[Role.CONFIG] += SignalWeight.STRUCTURE * 0.8
                reasoning_parts.append(f"Simple module ({f.module_assign_count} assigns)")
        
        # INTERFACE
        if total_funcs > 0:
            abstract_ratio = (f.abstract_method_count + f.empty_method_count) / total_funcs
            if abstract_ratio > 0.7:
                scores[Role.INTERFACE] += SignalWeight.STRUCTURE
                reasoning_parts.append(f"Abstract ratio: {abstract_ratio:.2f}")
            elif abstract_ratio > 0.4:
                scores[Role.INTERFACE] += SignalWeight.STRUCTURE * 0.6
        
        # SCHEMA
        if f.class_count > 0:
            # 异常类
            if f.has_exception_class:
                scores[Role.SCHEMA] += SignalWeight.STRUCTURE * 0.8
            
            # 字段 vs 方法比例
            if f.field_count > f.method_count and f.method_count <= 5:
                ratio_score = min(f.field_count / max(f.method_count, 1), 3) / 3
                scores[Role.SCHEMA] += SignalWeight.STRUCTURE * ratio_score
                reasoning_parts.append(f"Data ratio: {f.field_count}F/{f.method_count}M")
        
        # ADAPTER
        if f.io_boundary_count > 0:
            scores[Role.ADAPTER] += SignalWeight.STRUCTURE * (f.io_boundary_count / max(f.function_count, 1))
        if f.orchestrator_count > 0 and f.function_count > 0:
            if f.orchestrator_count / f.function_count > 0.3:
                scores[Role.ADAPTER] += SignalWeight.STRUCTURE * 0.5
        
        # UTIL
        if f.function_count > 0 and f.class_count == 0:
            if f.pure_function_count / f.function_count > 0.5:
                scores[Role.UTIL] += SignalWeight.STRUCTURE
                reasoning_parts.append(f"Pure functions: {f.pure_function_count}/{f.function_count}")
        
        # LOGIC (兜底基础分)
        if f.function_count > 0 or f.class_count > 0:
            scores[Role.LOGIC] += SignalWeight.STRUCTURE * 0.3
        
        # ====== Layer 3: 路径提示 (权重 1.5) ======
        
        for role, hints in self.PATH_HINTS.items():
            if self._path_matches(file_path, hints):
                scores[role] += SignalWeight.PATH_HINT
                reasoning_parts.append(f"Path hint -> {role.value}")
        
        # ====== Layer 4: 实体聚合 ======
        
        if entities:
            entity_role_weights = defaultdict(float)
            for e in entities:
                entity_role_weights[e.role] += e.weight * e.confidence
            
            total_entity_weight = sum(entity_role_weights.values())
            if total_entity_weight > 0:
                for role, weight in entity_role_weights.items():
                    # 实体聚合贡献中等权重
                    scores[role] += (weight / total_entity_weight) * SignalWeight.STRUCTURE
        
        # ====== 选择最佳角色 ======
        
        if not scores:
            return RoleScore(Role.UNKNOWN, 0.0, {}, "No signals detected")
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_role, primary_score = sorted_scores[0]
        
        return RoleScore(
            primary_role=primary_role,
            primary_score=primary_score,
            all_scores=dict(scores),
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Default classification"
        )
    
    def _path_matches(self, file_path: str, hints: List[str]) -> bool:
        """检查路径是否匹配提示"""
        if not file_path:
            return False
        path_lower = file_path.lower()
        return any(f'/{hint}/' in path_lower or path_lower.endswith(f'/{hint}') for hint in hints)
    
    # ========================================================================
    # 辅助方法
    # ========================================================================
    
    def _is_empty_body(self, body: List[ast.stmt]) -> bool:
        """判断函数体是否为空"""
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue  # docstring
            if isinstance(stmt, (ast.Pass,)):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                continue  # Ellipsis
            return False
        return True
    
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
    
    def _is_io_boundary_func(self, node: ast.FunctionDef) -> bool:
        """判断函数是否为IO边界"""
        # 参数名检查（更严格：ctx/context 太通用，容易误判）
        param_names = [arg.arg.lower() for arg in node.args.args]
        # 高置信度 IO 关键词
        strong_io_keywords = ['request', 'response', 'req', 'resp', 'httpcontext']
        # 中等置信度 IO 关键词（需要配合其他信号）
        weak_io_keywords = ['event', 'payload', 'conn', 'session']
        
        if any(kw in param_names for kw in strong_io_keywords):
            return True
        
        # 弱信号需要配合返回类型或函数名
        has_weak_signal = any(kw in param_names for kw in weak_io_keywords)
        
        # 返回注解检查
        if node.returns:
            return_type = self._get_name(node.returns).lower()
            if any(kw in return_type for kw in ['response', 'json', 'httpresponse', 'result']):
                return True
            # 弱信号 + 返回类型暗示
            if has_weak_signal and 'dict' in return_type:
                return True
        
        # 函数名检查（配合弱信号）
        func_name = node.name.lower()
        if has_weak_signal and any(kw in func_name for kw in ['handle', 'process', 'dispatch']):
            return True
        
        return False
    
    def _is_fixture_func(self, node: ast.FunctionDef) -> bool:
        """判断是否为 fixture"""
        for deco in node.decorator_list:
            deco_name = self._get_name(deco).lower()
            if any(kw in deco_name for kw in ['fixture', 'setup', 'teardown']):
                return True
        return False
    
    def _is_orchestrator_func(self, node: ast.FunctionDef) -> bool:
        """判断是否为编排函数（调用多，逻辑少）"""
        calls = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call))
        control_flow = sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While)))
        return calls > 5 and control_flow < 3
    
    def _is_pure_function(self, node: ast.FunctionDef) -> bool:
        """判断是否为纯函数"""
        # 有返回值
        has_return = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node))
        if not has_return:
            return False
        
        # 无全局/非局部变量访问
        has_global = any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node))
        if has_global:
            return False
        
        # 无 IO 操作（简化检测）
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                func_name = self._get_name(n.func).lower()
                if any(kw in func_name for kw in ['open', 'read', 'write', 'print', 'request', 'fetch', 'send']):
                    return False
        
        return True
    
    def _is_config_assignment(self, node: ast.Assign) -> bool:
        """判断是否为配置式赋值"""
        if isinstance(node.value, (ast.Constant, ast.List, ast.Dict, ast.Set, ast.Tuple)):
            return True
        if isinstance(node.value, ast.Call):
            func_name = self._get_name(node.value.func).lower()
            if any(kw in func_name for kw in ['getenv', 'environ', 'config']):
                return True
        return False
    
    def _is_main_check(self, node: ast.If) -> bool:
        """检测 if __name__ == '__main__'"""
        try:
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    return True
        except:
            pass
        return False
    
    def _get_name(self, node) -> str:
        """安全获取节点名称"""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return self._get_name(node.func)
        if isinstance(node, ast.Subscript):
            return self._get_name(node.value)
        return ""
    
    def _empty_result(self, file_path: str, reason: str = "") -> FileAnalysis:
        """返回空结果"""
        return FileAnalysis(
            file_path=file_path,
            role_score=RoleScore(Role.UNKNOWN, 0.0, {}, reason)
        )


# ============================================================================
# CLI 接口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAACS 4.0 Code Role Classifier v7")
    parser.add_argument("path", help="Path to analyze (file or directory)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--entities", action="store_true", help="Show entity-level roles")
    parser.add_argument("--scores", action="store_true", help="Show all role scores")
    parser.add_argument("--purity", action="store_true", help="Show role purity")
    args = parser.parse_args()

    classifier = CodeRoleClassifier(debug=args.debug)
    
    # 颜色定义
    COLORS = {
        Role.TEST: '\033[90m',       # Grey
        Role.NAMESPACE: '\033[36m',  # Cyan
        Role.INTERFACE: '\033[35m',  # Magenta
        Role.SCHEMA: '\033[34m',     # Blue
        Role.ADAPTER: '\033[33m',    # Yellow
        Role.CONFIG: '\033[37m',     # White
        Role.SCRIPT: '\033[31m',     # Red
        Role.UTIL: '\033[96m',       # Light Cyan
        Role.LOGIC: '\033[32m',      # Green
        Role.UNKNOWN: '\033[0m'
    }
    RESET = '\033[0m'
    
    target = os.path.abspath(args.path)
    
    if os.path.isfile(target):
        # 单文件分析
        result = classifier.analyze_file(target)
        
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(target)}")
        print(f"{'='*60}")
        
        primary = result.role_score.primary_role
        print(f"\nPrimary Role: {COLORS[primary]}{primary.value}{RESET} (score: {result.role_score.primary_score:.2f})")
        
        if result.role_score.secondary_roles:
            print(f"Secondary: ", end="")
            for role, score in result.role_score.secondary_roles:
                print(f"{COLORS[role]}{role.value}{RESET}({score:.2f}) ", end="")
            print()
        
        if result.role_score.reasoning:
            print(f"Reasoning: {result.role_score.reasoning}")
        
        if args.purity:
            print(f"\nRole Purity: {result.role_purity:.2f}")
            print(f"Use File-level Morphing: {result.use_file_level_morphing}")
        
        if args.scores:
            print(f"\nAll Scores:")
            for role, score in sorted(result.role_score.all_scores.items(), key=lambda x: -x[1]):
                if score > 0:
                    print(f"  {COLORS[role]}{role.value:<12}{RESET}: {score:.2f}")
        
        if args.entities and result.entities:
            print(f"\nEntities ({len(result.entities)}):")
            for e in result.entities:
                print(f"  [{e.entity_type}] {e.name:<25} -> {COLORS[e.role]}{e.role.value:<10}{RESET} "
                      f"(conf={e.confidence:.2f}, w={e.weight:.2f}) {e.reasoning}")
    
    else:
        # 目录分析
        print(f"{'Path':<55} | {'Role':<12} | {'Score':>6} | {'Purity':>6} | {'Entities'}")
        print("-" * 110)
        
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if f.endswith('.py'):
                    full_path = os.path.join(root, f)
                    result = classifier.analyze_file(full_path)
                    rel_path = os.path.relpath(full_path, target)
                    
                    primary = result.role_score.primary_role
                    entity_summary = ""
                    if result.entities:
                        role_counts = defaultdict(int)
                        for e in result.entities:
                            role_counts[e.role] += 1
                        entity_summary = ", ".join([f"{r.value}:{c}" for r, c in role_counts.items()])
                    
                    print(f"{rel_path[:55]:<55} | "
                          f"{COLORS[primary]}{primary.value:<12}{RESET} | "
                          f"{result.role_score.primary_score:>6.2f} | "
                          f"{result.role_purity:>6.2f} | "
                          f"{entity_summary}")

