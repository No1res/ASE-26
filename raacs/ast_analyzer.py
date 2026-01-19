import ast
import os
import argparse
import warnings
from collections import defaultdict
from typing import Set, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re


def safe_parse_source(source: str, filename: str = "<unknown>") -> Optional[ast.AST]:
    """
    安全地解析 Python 源代码，抑制 SyntaxWarning。
    
    被分析的代码可能包含无效的转义序列（如 JSON 中的 \\/ ），
    这些在 Python 中会产生 SyntaxWarning，但不影响 AST 解析。
    
    Args:
        source: Python 源代码字符串
        filename: 文件名（用于错误信息）
        
    Returns:
        AST 树，解析失败返回 None
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        try:
            return ast.parse(source, filename=filename)
        except SyntaxError:
            return None


class Role(Enum):
    """代码角色枚举 - 9个角色对应9种 Morphing 策略"""
    TEST = "TEST"
    NAMESPACE = "NAMESPACE"
    INTERFACE = "INTERFACE"
    SCHEMA = "SCHEMA"
    ADAPTER = "ADAPTER"
    CONFIG = "CONFIG"
    SCRIPT = "SCRIPT"
    UTIL = "UTIL"
    LOGIC = "LOGIC"
    UNKNOWN = "UNKNOWN"


class RoleSource(Enum):
    """角色来源 - 用于追踪角色判定的信号来源"""
    INITIAL_FRAMEWORK = "initial_framework"  # 框架指纹（强信号）
    INITIAL_DECORATOR = "initial_decorator"  # 装饰器（强信号）
    INITIAL_NAME = "initial_name"            # 名称 hint（弱信号）
    INHERITED = "inherited"                  # 直接继承（强信号）
    PROPAGATED = "propagated"                # 传播推断（中等信号）
    STRUCTURAL = "structural"                # 结构模式（中等信号）
    UNKNOWN = "unknown"                      # 未知来源


# 角色来源强度映射 - 用于判断是否允许覆盖
ROLE_SOURCE_STRENGTH = {
    RoleSource.INITIAL_FRAMEWORK: 0.95,  # 框架指纹，最强
    RoleSource.INITIAL_DECORATOR: 0.90,  # 装饰器，强
    RoleSource.INHERITED: 0.85,          # 直接继承，强
    RoleSource.PROPAGATED: 0.75,         # 传播推断，中
    RoleSource.STRUCTURAL: 0.65,         # 结构模式，中
    RoleSource.INITIAL_NAME: 0.50,       # 名称 hint，弱
    RoleSource.UNKNOWN: 0.0,             # 未知，最弱
}


# ============================================================================
# 角色兼容性矩阵
# ============================================================================
ROLE_COMPATIBILITY = {
    (Role.SCHEMA, Role.CONFIG): 0.9,
    (Role.SCHEMA, Role.INTERFACE): 0.8,
    (Role.CONFIG, Role.INTERFACE): 0.7,
    (Role.UTIL, Role.LOGIC): 0.5,
    (Role.ADAPTER, Role.LOGIC): 0.4,
    (Role.ADAPTER, Role.UTIL): 0.5,
    (Role.SCHEMA, Role.LOGIC): 0.2,
    (Role.SCHEMA, Role.UTIL): 0.3,
    (Role.SCHEMA, Role.ADAPTER): 0.2,
    (Role.CONFIG, Role.LOGIC): 0.3,
    (Role.INTERFACE, Role.LOGIC): 0.4,
    (Role.TEST, Role.LOGIC): 0.1,
    (Role.SCRIPT, Role.LOGIC): 0.3,
    (Role.NAMESPACE, Role.SCHEMA): 0.5,
}


# ============================================================================
# 框架指纹库
# ============================================================================
FRAMEWORK_SIGNATURES = {
    'web_frameworks': {
        'imports': {'flask', 'fastapi', 'starlette', 'aiohttp', 'tornado', 'bottle', 'falcon', 'sanic'},
        'decorators': {'route', 'get', 'post', 'put', 'delete', 'patch', 'api_view', 'action', 
                      'login_required', 'permission_classes', 'throttle_classes'},
    },
    'data_frameworks': {
        'imports': {'pydantic', 'sqlalchemy', 'marshmallow', 'attrs', 'tortoise', 'peewee', 'mongoengine'},
        'base_classes': {'BaseModel', 'Model', 'Schema', 'Document', 'Entity', 'ModelForm', 'Form'},
        'decorators': {'dataclass', 'dataclasses.dataclass'},
    },
    'test_frameworks': {
        'imports': {'pytest', 'unittest', 'nose', 'hypothesis', 'mock', 'faker'},
        'base_classes': {'TestCase', 'TestSuite'},
        'decorators': {'fixture', 'mark', 'parametrize', 'patch', 'mock'},
    },
    'async_frameworks': {
        'imports': {'celery', 'rq', 'dramatiq', 'huey'},
        'decorators': {'task', 'shared_task', 'job'},
    },
}

# Django 子模块细粒度角色映射
DJANGO_SUBMODULE_ROLES = {
    'django.forms': Role.SCHEMA,
    'django.db.models': Role.SCHEMA,
    'django.db': Role.SCHEMA,
    'django.views': Role.ADAPTER,
    'django.urls': Role.ADAPTER,
    'django.contrib.admin': Role.ADAPTER,
    'django.middleware': Role.ADAPTER,
    'django.http': Role.ADAPTER,
    'django.conf': Role.CONFIG,
    'django.apps': Role.CONFIG,
}

# 已知基类 -> 角色的映射（用于角色传播）
KNOWN_BASE_ROLES = {
    # 数据类基类
    'BaseModel': Role.SCHEMA,
    'Model': Role.SCHEMA,
    'Schema': Role.SCHEMA,
    'Document': Role.SCHEMA,
    'Entity': Role.SCHEMA,
    'ModelForm': Role.SCHEMA,
    'Form': Role.SCHEMA,
    'TypedDict': Role.SCHEMA,
    'NamedTuple': Role.SCHEMA,
    # 异常基类
    'Exception': Role.SCHEMA,
    'BaseException': Role.SCHEMA,
    # 抽象基类
    'ABC': Role.INTERFACE,
    'Protocol': Role.INTERFACE,
    # 测试基类
    'TestCase': Role.TEST,
    'TestSuite': Role.TEST,
}


class SignalWeight:
    FRAMEWORK = 4.0
    STRUCTURE = 2.5
    PATH_HINT = 1.5
    NAME_HINT = 1.0
    INHERITANCE = 3.5  # 新增：继承信号权重


# ============================================================================
# 符号表数据结构
# ============================================================================

@dataclass
class BaseInfo:
    """结构化基类信息"""
    raw: str                    # 原始字符串，如 "models.Model"
    simple: str                 # 简单名，如 "Model"
    head: str                   # 头部（可能是别名），如 "models"
    qual_candidates: List[str] = field(default_factory=list)  # 可能的 FQN 候选
    
    @classmethod
    def from_raw(cls, raw: str) -> 'BaseInfo':
        """从原始字符串创建 BaseInfo"""
        parts = raw.split('.')
        return cls(
            raw=raw,
            simple=parts[-1] if parts else raw,
            head=parts[0] if len(parts) > 1 else "",
            qual_candidates=[]
        )


@dataclass
class ClassSymbol:
    """类符号信息"""
    name: str                                    # 类名
    file_path: str                               # 所在文件
    base_names: List[str]                        # 基类名（字符串形式，向后兼容）
    base_infos: List[BaseInfo] = None            # 结构化基类信息（新增）
    resolved_bases: List[str] = None             # 解析后的基类全限定名
    initial_role: Role = Role.UNKNOWN            # 初步角色
    final_role: Role = Role.UNKNOWN              # 最终角色
    role_source: RoleSource = RoleSource.UNKNOWN # 角色来源（新增）
    role_confidence: float = 0.0
    inherited_from_bases: List[str] = None       # 角色继承来源（新增，支持多基类）
    lineno: int = 0
    
    def __post_init__(self):
        if self.resolved_bases is None:
            self.resolved_bases = []
        if self.base_infos is None:
            self.base_infos = [BaseInfo.from_raw(b) for b in self.base_names]
        if self.inherited_from_bases is None:
            self.inherited_from_bases = []
    
    @property
    def source_strength(self) -> float:
        """获取当前角色来源的强度"""
        return ROLE_SOURCE_STRENGTH.get(self.role_source, 0.0)


@dataclass
class ImportInfo:
    """导入信息"""
    local_name: str       # 本地名称（可能是别名）
    source_module: str    # 来源模块
    original_name: str    # 原始名称（如果是 from x import y as z，则是 y）
    is_from_import: bool = True
    is_alias: bool = False  # 是否有 as 别名（新增）
    
    def resolve_attr(self, attr: str) -> str:
        """解析属性访问，如 m.Base -> pkg.mod.Base"""
        if self.is_from_import:
            # from x import Y -> Y 就是 x.Y
            return f"{self.source_module}.{attr}" if attr else f"{self.source_module}.{self.original_name}"
        else:
            # import pkg.mod as m -> m.Base 就是 pkg.mod.Base
            return f"{self.source_module}.{attr}"


@dataclass
class FileSymbols:
    """文件符号信息"""
    file_path: str
    classes: Dict[str, ClassSymbol] = field(default_factory=dict)
    imports: Dict[str, ImportInfo] = field(default_factory=dict)  # local_name -> ImportInfo
    module_path: str = ""  # 模块路径（如 'mypackage.models'）
    
    def resolve_qualname(self, name: str, attr_chain: List[str] = None) -> List[str]:
        """
        解析本地名称到可能的 FQN 列表
        
        示例:
            import pkg.mod as m
            resolve_qualname("m", ["Base"]) -> ["pkg.mod.Base"]
            
            from x import Y as Z
            resolve_qualname("Z", []) -> ["x.Y"]
        
        Args:
            name: 本地名称（可能是别名或直接名称）
            attr_chain: 属性链，如 ["Base", "Inner"] 表示 name.Base.Inner
            
        Returns:
            可能的 FQN 列表
        """
        candidates = []
        attr_chain = attr_chain or []
        
        # 1. 检查是否是导入的名称
        if name in self.imports:
            import_info = self.imports[name]
            if attr_chain:
                # name.attr1.attr2 形式
                attr_path = '.'.join(attr_chain)
                candidates.append(f"{import_info.source_module}.{attr_path}")
            else:
                # 直接使用导入的名称
                if import_info.is_from_import:
                    candidates.append(f"{import_info.source_module}.{import_info.original_name}")
                else:
                    candidates.append(import_info.source_module)
        
        # 2. 检查是否是本文件定义的类
        full_name = '.'.join([name] + attr_chain) if attr_chain else name
        if full_name in self.classes:
            if self.module_path:
                candidates.append(f"{self.module_path}.{full_name}")
            candidates.append(full_name)
        
        # 3. 直接使用原始名称作为候选
        if full_name not in candidates:
            candidates.append(full_name)
        
        return candidates


@dataclass
class ProjectSymbolTable:
    """项目级符号表"""
    root_path: str
    files: Dict[str, FileSymbols] = field(default_factory=dict)  # file_path -> FileSymbols
    
    # 分离的全局类索引（新增）
    global_classes_by_fqn: Dict[str, ClassSymbol] = field(default_factory=dict)    # FQN -> ClassSymbol（唯一）
    global_classes_by_simple: Dict[str, List[ClassSymbol]] = field(default_factory=lambda: defaultdict(list))  # simple_name -> [ClassSymbol]（可能多个）
    
    # 向后兼容：保留 global_classes（将指向 by_fqn）
    @property
    def global_classes(self) -> Dict[str, ClassSymbol]:
        """向后兼容的全局类索引"""
        return self.global_classes_by_fqn
    
    def get_class_by_name(self, class_name: str, from_file: str = None) -> Optional[ClassSymbol]:
        """
        根据类名查找类符号（增强版）
        
        查找优先级：
        1. FQN 精确匹配
        2. 文件上下文中的本地类
        3. 通过导入解析的 FQN
        4. Simple name 匹配（选择最近的）
        """
        # 1. FQN 精确匹配
        if class_name in self.global_classes_by_fqn:
            return self.global_classes_by_fqn[class_name]
        
        # 2. 如果提供了文件上下文
        if from_file and from_file in self.files:
            file_symbols = self.files[from_file]
            
            # 2a. 检查是否是本文件定义的类
            if class_name in file_symbols.classes:
                return file_symbols.classes[class_name]
            
            # 2b. 解析可能的 FQN 候选
            # 处理 alias.ClassName 形式
            parts = class_name.split('.')
            if len(parts) > 1:
                head = parts[0]
                attr_chain = parts[1:]
                candidates = file_symbols.resolve_qualname(head, attr_chain)
            else:
                candidates = file_symbols.resolve_qualname(class_name)
            
            # 在 FQN 索引中查找候选
            for candidate in candidates:
                if candidate in self.global_classes_by_fqn:
                    return self.global_classes_by_fqn[candidate]
            
            # 2c. 检查导入的类
            if class_name in file_symbols.imports:
                import_info = file_symbols.imports[class_name]
                possible_names = [
                    f"{import_info.source_module}.{import_info.original_name}",
                    import_info.source_module,
                ]
                for name in possible_names:
                    if name in self.global_classes_by_fqn:
                        return self.global_classes_by_fqn[name]
        
        # 3. Simple name 匹配
        simple_name = class_name.split('.')[-1]
        if simple_name in self.global_classes_by_simple:
            candidates = self.global_classes_by_simple[simple_name]
            if len(candidates) == 1:
                return candidates[0]
            elif len(candidates) > 1 and from_file:
                # 多个候选时，选择模块路径最接近的
                return self._select_nearest_class(candidates, from_file)
        
        return None
    
    def get_class_by_base_info(self, base_info: BaseInfo, from_file: str = None) -> Optional[ClassSymbol]:
        """
        使用结构化的 BaseInfo 查找类符号
        
        比 get_class_by_name 更精确，因为可以利用 qual_candidates
        """
        # 1. 优先使用预计算的候选
        for candidate in base_info.qual_candidates:
            if candidate in self.global_classes_by_fqn:
                return self.global_classes_by_fqn[candidate]
        
        # 2. 回退到普通查找
        return self.get_class_by_name(base_info.raw, from_file)
    
    def _select_nearest_class(self, candidates: List[ClassSymbol], from_file: str) -> Optional[ClassSymbol]:
        """从多个同名类中选择最近的"""
        if not candidates:
            return None
        
        if from_file not in self.files:
            return candidates[0]
        
        from_module = self.files[from_file].module_path
        
        # 计算模块路径相似度
        def module_similarity(cls: ClassSymbol) -> int:
            cls_file = self.files.get(cls.file_path)
            if not cls_file:
                return 0
            cls_module = cls_file.module_path
            
            # 计算共同前缀长度
            from_parts = from_module.split('.')
            cls_parts = cls_module.split('.')
            common = 0
            for a, b in zip(from_parts, cls_parts):
                if a == b:
                    common += 1
                else:
                    break
            return common
        
        return max(candidates, key=module_similarity)


# ============================================================================
# 第一阶段：符号收集器
# ============================================================================

class SymbolCollector:
    """符号收集器 - 扫描项目建立符号表"""
    
    def __init__(self, root_path: str):
        self.root_path = os.path.abspath(root_path)
        self.symbol_table = ProjectSymbolTable(root_path=self.root_path)
    
    def collect(self) -> ProjectSymbolTable:
        """收集整个项目的符号"""
        # 1. 扫描所有 Python 文件
        python_files = self._find_python_files()
        
        # 2. 第一遍：收集每个文件的类和导入
        for file_path in python_files:
            self._collect_file_symbols(file_path)
        
        # 3. 建立全局类索引
        self._build_global_index()
        
        return self.symbol_table
    
    def _find_python_files(self) -> List[str]:
        """查找所有 Python 文件"""
        python_files = []
        for root, _, files in os.walk(self.root_path):
            # 跳过隐藏目录和常见的排除目录
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(excl in root for excl in ['__pycache__', 'venv', 'env', '.git', 'node_modules']):
                continue
            
            for f in files:
                if f.endswith('.py'):
                    python_files.append(os.path.join(root, f))
        
        return python_files
    
    def _collect_file_symbols(self, file_path: str):
        """收集单个文件的符号"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = safe_parse_source(source, filename=file_path)
            if tree is None:
                return
        except UnicodeDecodeError:
            return
        
        file_symbols = FileSymbols(
            file_path=file_path,
            module_path=self._path_to_module(file_path)
        )
        
        # 收集导入
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local_name = alias.asname or alias.name.split('.')[-1]
                    file_symbols.imports[local_name] = ImportInfo(
                        local_name=local_name,
                        source_module=alias.name,
                        original_name=alias.name.split('.')[-1],
                        is_from_import=False,
                        is_alias=alias.asname is not None
                    )
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                # 处理相对导入
                if node.level > 0:
                    module = self._resolve_relative_import(file_path, node.level, module)
                
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    file_symbols.imports[local_name] = ImportInfo(
                        local_name=local_name,
                        source_module=module,
                        original_name=alias.name,
                        is_from_import=True,
                        is_alias=alias.asname is not None
                    )
        
        # 收集类定义
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                base_names = [self._get_base_name(base) for base in node.bases]
                base_infos = [BaseInfo.from_raw(b) for b in base_names]
                initial_role, role_source = self._determine_initial_role(node, file_symbols)
                
                class_symbol = ClassSymbol(
                    name=node.name,
                    file_path=file_path,
                    base_names=base_names,
                    base_infos=base_infos,
                    initial_role=initial_role,
                    final_role=initial_role,
                    role_source=role_source,
                    lineno=node.lineno
                )
                file_symbols.classes[node.name] = class_symbol
        
        self.symbol_table.files[file_path] = file_symbols
    
    def _path_to_module(self, file_path: str) -> str:
        """将文件路径转换为模块路径"""
        rel_path = os.path.relpath(file_path, self.root_path)
        module_path = rel_path.replace(os.sep, '.').replace('/', '.')
        if module_path.endswith('.py'):
            module_path = module_path[:-3]
        if module_path.endswith('.__init__'):
            module_path = module_path[:-9]
        return module_path
    
    def _resolve_relative_import(self, file_path: str, level: int, module: str) -> str:
        """解析相对导入"""
        # 获取当前文件的目录
        current_dir = os.path.dirname(file_path)
        
        # 向上跳转 level-1 级（level=1 是当前包）
        for _ in range(level - 1):
            current_dir = os.path.dirname(current_dir)
        
        # 转换为模块路径
        base_module = self._path_to_module(current_dir + '/__init__.py')
        if base_module and module:
            return f"{base_module}.{module}"
        elif base_module:
            return base_module
        else:
            return module
    
    def _get_base_name(self, node: ast.expr) -> str:
        """获取基类名"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parts = []
            current = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        elif isinstance(node, ast.Subscript):
            # Generic types like List[int], Optional[str]
            return self._get_base_name(node.value)
        return ""
    
    def _determine_initial_role(self, node: ast.ClassDef, file_symbols: FileSymbols) -> Tuple[Role, RoleSource]:
        """
        确定类的初步角色（基于框架指纹）
        
        Returns:
            (Role, RoleSource) 元组，包含角色和角色来源
        """
        base_names = [self._get_base_name(base) for base in node.bases]
        
        # 1. 检查基类 - 框架指纹（强信号）
        for base_name in base_names:
            simple_name = base_name.split('.')[-1]
            if simple_name in KNOWN_BASE_ROLES:
                return KNOWN_BASE_ROLES[simple_name], RoleSource.INITIAL_FRAMEWORK
        
        # 2. 检查装饰器（强信号）
        for deco in node.decorator_list:
            deco_name = self._get_decorator_name(deco)
            if deco_name in FRAMEWORK_SIGNATURES['data_frameworks'].get('decorators', set()):
                return Role.SCHEMA, RoleSource.INITIAL_DECORATOR
            if deco_name in FRAMEWORK_SIGNATURES['test_frameworks'].get('decorators', set()):
                return Role.TEST, RoleSource.INITIAL_DECORATOR
        
        # 3. 检查类名模式（弱信号）
        class_name_lower = node.name.lower()
        if 'test' in class_name_lower:
            return Role.TEST, RoleSource.INITIAL_NAME
        if 'schema' in class_name_lower or 'model' in class_name_lower:
            return Role.SCHEMA, RoleSource.INITIAL_NAME
        if 'view' in class_name_lower or 'handler' in class_name_lower:
            return Role.ADAPTER, RoleSource.INITIAL_NAME
        
        return Role.UNKNOWN, RoleSource.UNKNOWN
    
    def _get_decorator_name(self, node: ast.expr) -> str:
        """获取装饰器名"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        return ""
    
    def _build_global_index(self):
        """建立全局类索引（分离 FQN 和 Simple Name）"""
        for file_path, file_symbols in self.symbol_table.files.items():
            for class_name, class_symbol in file_symbols.classes.items():
                # 使用模块路径.类名作为全限定名
                full_name = f"{file_symbols.module_path}.{class_name}" if file_symbols.module_path else class_name
                
                # FQN 索引（唯一）
                self.symbol_table.global_classes_by_fqn[full_name] = class_symbol
                
                # Simple name 索引（可能多个同名类）
                self.symbol_table.global_classes_by_simple[class_name].append(class_symbol)
                
                # 预计算 BaseInfo 的 qual_candidates
                self._resolve_base_candidates(class_symbol, file_symbols)
    
    def _resolve_base_candidates(self, class_symbol: ClassSymbol, file_symbols: FileSymbols):
        """预计算基类的 FQN 候选"""
        for base_info in class_symbol.base_infos:
            # 解析可能的 FQN
            parts = base_info.raw.split('.')
            if len(parts) > 1:
                # alias.ClassName 形式
                head = parts[0]
                attr_chain = parts[1:]
                candidates = file_symbols.resolve_qualname(head, attr_chain)
            else:
                # 简单名称
                candidates = file_symbols.resolve_qualname(base_info.raw)
            
            base_info.qual_candidates = candidates


# ============================================================================
# 第二阶段：角色传播器（增强版）
# ============================================================================

@dataclass
class BaseRoleInfo:
    """基类角色信息（用于多基类融合）"""
    base_name: str
    role: Role
    source: RoleSource
    confidence: float


class RolePropagator:
    """
    角色传播器 - 通过继承关系传播角色（增强版）
    
    改进：
    1. 支持弱信号覆盖：强继承信号可以覆盖弱初始角色
    2. 多基类角色收集与融合
    3. 记录角色继承来源
    """
    
    # 弱信号阈值：低于此强度的角色可被覆盖
    WEAK_SIGNAL_THRESHOLD = 0.6
    
    def __init__(self, symbol_table: ProjectSymbolTable):
        self.symbol_table = symbol_table
        self.propagation_count = 0
    
    def propagate(self, max_iterations: int = 10) -> int:
        """
        迭代传播角色
        
        Returns:
            传播的角色数量
        """
        total_propagated = 0
        
        for iteration in range(max_iterations):
            propagated_this_round = self._propagate_one_round()
            total_propagated += propagated_this_round
            
            if propagated_this_round == 0:
                break  # 没有新的传播，收敛了
        
        return total_propagated
    
    def _propagate_one_round(self) -> int:
        """执行一轮角色传播（增强版）"""
        propagated = 0
        
        for file_path, file_symbols in self.symbol_table.files.items():
            for class_name, class_symbol in file_symbols.classes.items():
                # 获取当前角色的强度
                current_strength = class_symbol.source_strength
                
                # 收集所有基类的角色信息
                base_roles = self._collect_base_roles(class_symbol, file_path)
                
                if not base_roles:
                    continue
                
                # 融合多基类角色
                fused_role, fused_confidence, inherited_from = self._fuse_base_roles(base_roles)
                
                if fused_role == Role.UNKNOWN:
                    continue
                
                # 判断是否应该更新角色
                should_update = False
                
                if class_symbol.final_role == Role.UNKNOWN:
                    # 情况1：当前没有角色，直接更新
                    should_update = True
                elif current_strength < self.WEAK_SIGNAL_THRESHOLD:
                    # 情况2：当前是弱信号，允许被强继承信号覆盖
                    # 但只有当继承信号足够强时才覆盖
                    if fused_confidence > current_strength:
                        should_update = True
                
                if should_update:
                    class_symbol.final_role = fused_role
                    class_symbol.role_source = RoleSource.PROPAGATED
                    class_symbol.role_confidence = fused_confidence
                    class_symbol.inherited_from_bases = inherited_from
                    propagated += 1
        
        return propagated
    
    def _collect_base_roles(self, class_symbol: ClassSymbol, file_path: str) -> List[BaseRoleInfo]:
        """收集所有可解析基类的角色信息"""
        base_roles = []
        
        for base_info in class_symbol.base_infos:
            # 1. 检查是否是已知的框架基类
            simple_name = base_info.simple
            if simple_name in KNOWN_BASE_ROLES:
                base_roles.append(BaseRoleInfo(
                    base_name=simple_name,
                    role=KNOWN_BASE_ROLES[simple_name],
                    source=RoleSource.INITIAL_FRAMEWORK,
                    confidence=0.95
                ))
                continue
            
            # 2. 使用增强的查找方法
            base_class = self.symbol_table.get_class_by_base_info(base_info, file_path)
            if base_class and base_class.final_role != Role.UNKNOWN:
                base_roles.append(BaseRoleInfo(
                    base_name=base_info.raw,
                    role=base_class.final_role,
                    source=base_class.role_source,
                    confidence=base_class.role_confidence or ROLE_SOURCE_STRENGTH.get(base_class.role_source, 0.5)
                ))
        
        return base_roles
    
    def _fuse_base_roles(self, base_roles: List[BaseRoleInfo]) -> Tuple[Role, float, List[str]]:
        """
        融合多个基类的角色
        
        策略：
        1. 如果所有基类角色相同 -> 直接返回，置信度提升
        2. 如果有 KNOWN_BASE_ROLES 中的角色 -> 优先
        3. 如果有冲突 -> 选择置信度最高的
        
        Returns:
            (fused_role, confidence, inherited_from_list)
        """
        if not base_roles:
            return Role.UNKNOWN, 0.0, []
        
        if len(base_roles) == 1:
            info = base_roles[0]
            return info.role, info.confidence, [info.base_name]
        
        # 收集所有角色
        role_infos: Dict[Role, List[BaseRoleInfo]] = defaultdict(list)
        for info in base_roles:
            role_infos[info.role].append(info)
        
        # 如果所有角色相同
        if len(role_infos) == 1:
            role = list(role_infos.keys())[0]
            # 多个相同角色，提升置信度
            max_conf = max(info.confidence for info in base_roles)
            boosted_conf = min(max_conf * 1.1, 1.0)
            inherited_from = [info.base_name for info in base_roles]
            return role, boosted_conf, inherited_from
        
        # 角色冲突：按置信度排序，选择最高的
        sorted_roles = sorted(
            [(role, infos) for role, infos in role_infos.items()],
            key=lambda x: max(info.confidence for info in x[1]),
            reverse=True
        )
        
        best_role, best_infos = sorted_roles[0]
        best_confidence = max(info.confidence for info in best_infos)
        inherited_from = [info.base_name for info in best_infos]
        
        return best_role, best_confidence, inherited_from


# ============================================================================
# 增强版文件分析器
# ============================================================================

@dataclass
class StructuralFeatures:
    """结构特征"""
    class_count: int = 0
    function_count: int = 0
    method_count: int = 0
    field_count: int = 0
    typed_field_count: int = 0
    module_assign_count: int = 0
    abstract_method_count: int = 0
    empty_method_count: int = 0
    io_boundary_count: int = 0
    orchestrator_count: int = 0
    pure_function_count: int = 0
    control_flow_count: int = 0
    call_count: int = 0
    assertion_count: int = 0
    test_function_count: int = 0
    fixture_count: int = 0
    constant_count: int = 0
    config_assign_count: int = 0
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
    django_schema_imports: Set[str] = field(default_factory=set)
    django_adapter_imports: Set[str] = field(default_factory=set)
    django_config_imports: Set[str] = field(default_factory=set)


@dataclass
class EntityRole:
    """实体级角色"""
    name: str
    entity_type: str
    role: Role
    confidence: float
    weight: float
    lineno: int
    reasoning: str = ""
    inherited_from: str = ""  # 新增：角色继承来源


@dataclass
class RoleScore:
    """角色打分结果"""
    primary_role: Role
    primary_score: float
    all_scores: Dict[Role, float] = field(default_factory=dict)
    reasoning: str = ""
    
    @property
    def secondary_roles(self) -> List[Tuple[Role, float]]:
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
        compat_bonus = 0
        for role, weight in role_weights.items():
            if role != self.role_score.primary_role:
                compat = get_role_compatibility(self.role_score.primary_role, role)
                compat_bonus += (weight / total) * compat * 0.2
        return min(base_purity + compat_bonus, 1.0)
    
    @property
    def use_file_level_morphing(self) -> bool:
        return self.role_purity > 0.75


def get_role_compatibility(role1: Role, role2: Role) -> float:
    if role1 == role2:
        return 1.0
    key = (role1, role2)
    if key in ROLE_COMPATIBILITY:
        return ROLE_COMPATIBILITY[key]
    key = (role2, role1)
    if key in ROLE_COMPATIBILITY:
        return ROLE_COMPATIBILITY[key]
    return 0.3


# ============================================================================
# 主分类器
# ============================================================================

class CodeRoleClassifier:
    """
    RAACS 5.0: 两阶段代码角色分类器
    
    支持两种模式：
    1. 单文件模式：向后兼容，无跨文件上下文
    2. 项目模式：两阶段分析，支持跨文件继承传播
    """
    
    PATH_HINTS = {
        Role.TEST: ['test', 'tests', '__tests__', 'spec', 'specs'],
        Role.CONFIG: ['config', 'conf', 'settings', 'constants'],
        Role.SCRIPT: ['scripts', 'bin', 'tools', 'cli', 'commands'],
        Role.ADAPTER: ['views', 'api', 'controllers', 'handlers', 'routes', 'endpoints'],
        Role.SCHEMA: ['models', 'schemas', 'entities', 'types', 'dto'],
        Role.UTIL: ['utils', 'helpers', 'common', 'lib', 'shared'],
    }
    
    def __init__(self, debug: bool = False, symbol_table: ProjectSymbolTable = None):
        self.debug = debug
        self.symbol_table = symbol_table
    
    @classmethod
    def create_project_analyzer(cls, project_root: str, debug: bool = False) -> 'CodeRoleClassifier':
        """
        创建项目级分析器
        
        执行两阶段分析：
        1. 符号收集
        2. 角色传播
        """
        # 第一阶段：符号收集
        collector = SymbolCollector(project_root)
        symbol_table = collector.collect()
        
        # 第二阶段：角色传播
        propagator = RolePropagator(symbol_table)
        propagated = propagator.propagate()
        
        if debug:
            print(f"[Symbol Table] Collected {len(symbol_table.global_classes)} classes")
            print(f"[Role Propagation] Propagated {propagated} roles")
        
        return cls(debug=debug, symbol_table=symbol_table)
    
    def analyze_file(self, file_path: str) -> FileAnalysis:
        """分析单个文件"""
        if not os.path.exists(file_path):
            return self._empty_result(file_path, "File not found")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            return self.analyze_source(source_code, file_path=file_path)
        except SyntaxError as e:
            return self._empty_result(file_path, f"Syntax error: {e}")
        except Exception as e:
            if self.debug:
                print(f"[Error] {file_path}: {e}")
            return self._empty_result(file_path, f"Error: {e}")
    
    def analyze_source(self, source_code: str, file_path: str = "") -> FileAnalysis:
        """核心分析流水线"""
        tree = safe_parse_source(source_code, filename=file_path or "<string>")
        if tree is None:
            return self._empty_result(file_path, "Syntax error")
        
        filename = os.path.basename(file_path).lower() if file_path else ""
        
        framework_signals = self._extract_framework_signals(tree)
        structural_features = self._extract_structural_features(tree, filename)
        entities = self._analyze_entities(tree, source_code, framework_signals, file_path)
        role_score = self._compute_role_score(
            framework_signals, structural_features, entities, file_path
        )
        
        return FileAnalysis(
            file_path=file_path,
            role_score=role_score,
            entities=entities,
            structural_features=structural_features,
            framework_signals=framework_signals
        )
    
    def _analyze_entities(self, tree: ast.AST, source_code: str,
                          framework_signals: FrameworkSignals, file_path: str) -> List[EntityRole]:
        """分析实体级角色（类和顶层函数）"""
        entities = []
        source_lines = source_code.split('\n')
        
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                entity = self._analyze_class_entity(node, source_lines, framework_signals, file_path)
                entities.append(entity)
            elif isinstance(node, ast.FunctionDef):
                entity = self._analyze_function_entity(node, source_lines, framework_signals)
                entities.append(entity)
        
        return entities
    
    def _analyze_class_entity(self, node: ast.ClassDef, source_lines: List[str],
                               framework_signals: FrameworkSignals, file_path: str) -> EntityRole:
        """分析类实体的角色"""
        weight = self._calculate_entity_weight(node, source_lines)
        
        # 统计特征
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        fields = []
        for n in node.body:
            if isinstance(n, ast.AnnAssign):
                fields.append(n)
            elif isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith('__'):
                        fields.append(n)
                        break
        
        non_magic_methods = [m for m in methods if not m.name.startswith('__')]
        init_only = len(methods) == 1 and methods[0].name == '__init__'
        empty_methods = sum(1 for m in methods if self._is_empty_body(m.body))
        abstract_methods = sum(1 for m in methods if self._has_not_implemented(m))
        
        # 获取基类信息
        base_names = [self._get_name(base) for base in node.bases]
        
        # === 新增：从符号表查找继承的角色 ===
        inherited_role = Role.UNKNOWN
        inherited_from = ""
        
        if self.symbol_table:
            # 首先检查符号表中这个类的角色
            class_symbol = self.symbol_table.get_class_by_name(node.name, file_path)
            if class_symbol and class_symbol.final_role != Role.UNKNOWN:
                inherited_role = class_symbol.final_role
                inherited_from = "symbol_table"
            else:
                # 尝试从基类继承
                for base_name in base_names:
                    base_class = self.symbol_table.get_class_by_name(base_name, file_path)
                    if base_class and base_class.final_role != Role.UNKNOWN:
                        inherited_role = base_class.final_role
                        inherited_from = base_name
                        break
        
        # 检查已知基类
        is_exception = any('Exception' in b or 'Error' in b for b in base_names)
        is_data_class = any(b in FRAMEWORK_SIGNATURES['data_frameworks'].get('base_classes', set()) for b in base_names)
        is_test_class = any(b in FRAMEWORK_SIGNATURES['test_frameworks'].get('base_classes', set()) for b in base_names)
        
        # 数据基类提示
        data_base_hints = ['entity', 'model', 'schema', 'dto', 'mixin', 'base']
        inherits_data_base = any(
            any(hint in b.lower() for hint in data_base_hints)
            for b in base_names
        )
        
        # 检查装饰器
        deco_names = [self._get_name(d) for d in node.decorator_list]
        has_dataclass = any('dataclass' in d.lower() for d in deco_names)
        
        # 空壳数据类判断
        is_empty_data_class = (
            (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)) or
            (init_only and len(fields) == 0)
        ) and (inherits_data_base or is_data_class)
        
        # === 角色判定 ===
        role = Role.LOGIC
        confidence = 0.5
        reasoning = ""
        
        # 1. 符号表继承（最高优先级）
        if inherited_role != Role.UNKNOWN:
            role = inherited_role
            confidence = 0.85
            reasoning = f"Inherited from {inherited_from}"
        
        # 2. 框架指纹判断
        elif is_test_class or 'test' in node.name.lower():
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
        
        # 3. 结构模式判断
        elif len(methods) > 0 and (empty_methods + abstract_methods) / len(methods) > 0.7:
            role = Role.INTERFACE
            confidence = 0.85
            reasoning = f"Abstract class ({empty_methods + abstract_methods}/{len(methods)} abstract methods)"
        elif len(fields) > 0 and len(fields) >= len(non_magic_methods):
            role = Role.SCHEMA
            confidence = 0.75 if inherits_data_base else 0.65
            reasoning = f"Data class ({len(fields)} fields >= {len(non_magic_methods)} methods)"
        elif init_only and inherits_data_base:
            role = Role.SCHEMA
            confidence = 0.7
            reasoning = f"Data class (init-only, inherits {base_names[0] if base_names else 'base'})"
        elif self._is_factory_class(node):
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
            reasoning=reasoning,
            inherited_from=inherited_from
        )
    
    def _analyze_function_entity(self, node: ast.FunctionDef, source_lines: List[str],
                                  framework_signals: FrameworkSignals) -> EntityRole:
        """分析函数实体的角色"""
        weight = self._calculate_entity_weight(node, source_lines)
        
        role = Role.LOGIC
        confidence = 0.5
        reasoning = ""
        
        deco_names = [self._get_name(d).lower() for d in node.decorator_list]
        
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
            reasoning = f"Web route decorator"
        elif any(d in ['task', 'shared_task', 'job'] for d in deco_names):
            role = Role.ADAPTER
            confidence = 0.85
            reasoning = "Async task decorator"
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
    
    # ========================================================================
    # 辅助方法（与 v7 相同）
    # ========================================================================
    
    def _extract_framework_signals(self, tree: ast.AST) -> FrameworkSignals:
        signals = FrameworkSignals()
        for node in ast.walk(tree):
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
            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = self._get_name(base)
                    if base_name in FRAMEWORK_SIGNATURES['data_frameworks'].get('base_classes', set()):
                        signals.data_base_classes.add(base_name)
                    if base_name in FRAMEWORK_SIGNATURES['test_frameworks'].get('base_classes', set()):
                        signals.test_base_classes.add(base_name)
                for deco in node.decorator_list:
                    self._classify_decorator(self._get_name(deco), signals)
            elif isinstance(node, ast.FunctionDef):
                for deco in node.decorator_list:
                    self._classify_decorator(self._get_name(deco), signals)
        return signals
    
    def _classify_import(self, module: str, signals: FrameworkSignals, full_module: str = ""):
        if module == 'django' and full_module:
            for django_path, role in DJANGO_SUBMODULE_ROLES.items():
                if full_module.startswith(django_path):
                    if role == Role.SCHEMA:
                        signals.django_schema_imports.add(full_module)
                    elif role == Role.ADAPTER:
                        signals.django_adapter_imports.add(full_module)
                    elif role == Role.CONFIG:
                        signals.django_config_imports.add(full_module)
                    return
            signals.django_adapter_imports.add(full_module)
            return
        if module in FRAMEWORK_SIGNATURES['web_frameworks']['imports']:
            signals.web_framework_imports.add(module)
        if module in FRAMEWORK_SIGNATURES['data_frameworks']['imports']:
            signals.data_framework_imports.add(module)
        if module in FRAMEWORK_SIGNATURES['test_frameworks']['imports']:
            signals.test_framework_imports.add(module)
        if module in FRAMEWORK_SIGNATURES['async_frameworks']['imports']:
            signals.async_framework_imports.add(module)
    
    def _classify_decorator(self, deco_name: str, signals: FrameworkSignals):
        deco_lower = deco_name.lower()
        if deco_lower in FRAMEWORK_SIGNATURES['web_frameworks']['decorators']:
            signals.web_decorators.add(deco_name)
        if deco_lower in FRAMEWORK_SIGNATURES['data_frameworks']['decorators']:
            signals.data_decorators.add(deco_name)
        if deco_lower in FRAMEWORK_SIGNATURES['test_frameworks']['decorators']:
            signals.test_decorators.add(deco_name)
    
    def _extract_structural_features(self, tree: ast.AST, filename: str) -> StructuralFeatures:
        f = StructuralFeatures()
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                f.module_assign_count += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                f.class_count += 1
                for base in node.bases:
                    base_name = self._get_name(base)
                    if 'Exception' in base_name or 'Error' in base_name:
                        f.has_exception_class = True
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
        if filename == '__init__.py':
            f.is_pure_init = (f.function_count == 0 and f.class_count == 0 and f.control_flow_count == 0)
        return f
    
    def _compute_role_score(self, framework_signals: FrameworkSignals,
                            structural_features: StructuralFeatures,
                            entities: List[EntityRole],
                            file_path: str) -> RoleScore:
        scores = defaultdict(float)
        reasoning_parts = []
        filename = os.path.basename(file_path).lower() if file_path else ""
        
        # Layer 1: 框架指纹
        if framework_signals.test_framework_imports or framework_signals.test_base_classes:
            scores[Role.TEST] += SignalWeight.FRAMEWORK
            reasoning_parts.append(f"Test framework")
        if framework_signals.web_framework_imports or framework_signals.web_decorators:
            scores[Role.ADAPTER] += SignalWeight.FRAMEWORK
        if framework_signals.async_framework_imports:
            scores[Role.ADAPTER] += SignalWeight.FRAMEWORK * 0.8
        if framework_signals.django_schema_imports:
            scores[Role.SCHEMA] += SignalWeight.FRAMEWORK
        if framework_signals.django_adapter_imports:
            scores[Role.ADAPTER] += SignalWeight.FRAMEWORK
        if framework_signals.django_config_imports:
            scores[Role.CONFIG] += SignalWeight.FRAMEWORK * 0.5
        if framework_signals.data_framework_imports or framework_signals.data_base_classes or framework_signals.data_decorators:
            scores[Role.SCHEMA] += SignalWeight.FRAMEWORK
        
        # Layer 2: 结构模式
        f = structural_features
        total_funcs = f.function_count + f.method_count
        
        if f.test_function_count > 0:
            scores[Role.TEST] += SignalWeight.STRUCTURE * (f.test_function_count / max(f.function_count, 1))
        if f.assertion_count > 0 and total_funcs > 0:
            if f.assertion_count / total_funcs > 0.5:
                scores[Role.TEST] += SignalWeight.STRUCTURE
        
        if f.has_main_entry:
            entity_count = f.class_count + f.function_count
            if entity_count > 3:
                scores[Role.SCRIPT] += SignalWeight.STRUCTURE * 0.3
            else:
                scores[Role.SCRIPT] += SignalWeight.STRUCTURE * 1.5
        
        if f.is_pure_init:
            scores[Role.NAMESPACE] += SignalWeight.STRUCTURE * 2
        elif filename == '__init__.py':
            scores[Role.NAMESPACE] += SignalWeight.STRUCTURE * 0.5
        
        if f.function_count == 0 and f.class_count == 0:
            if f.constant_count > 0 or f.config_assign_count > 0:
                scores[Role.CONFIG] += SignalWeight.STRUCTURE
            elif f.module_assign_count > 0:
                scores[Role.CONFIG] += SignalWeight.STRUCTURE * 0.8
        
        if total_funcs > 0:
            abstract_ratio = (f.abstract_method_count + f.empty_method_count) / total_funcs
            if abstract_ratio > 0.7:
                scores[Role.INTERFACE] += SignalWeight.STRUCTURE
            elif abstract_ratio > 0.4:
                scores[Role.INTERFACE] += SignalWeight.STRUCTURE * 0.6
        
        if f.class_count > 0:
            if f.has_exception_class:
                scores[Role.SCHEMA] += SignalWeight.STRUCTURE * 0.8
            if f.field_count > f.method_count and f.method_count <= 5:
                ratio_score = min(f.field_count / max(f.method_count, 1), 3) / 3
                scores[Role.SCHEMA] += SignalWeight.STRUCTURE * ratio_score
        
        if f.io_boundary_count > 0:
            scores[Role.ADAPTER] += SignalWeight.STRUCTURE * (f.io_boundary_count / max(f.function_count, 1))
        if f.orchestrator_count > 0 and f.function_count > 0:
            if f.orchestrator_count / f.function_count > 0.3:
                scores[Role.ADAPTER] += SignalWeight.STRUCTURE * 0.5
        
        if f.function_count > 0 and f.class_count == 0:
            if f.pure_function_count / f.function_count > 0.5:
                scores[Role.UTIL] += SignalWeight.STRUCTURE
        
        if f.function_count > 0 or f.class_count > 0:
            scores[Role.LOGIC] += SignalWeight.STRUCTURE * 0.3
        
        # Layer 3: 路径提示
        for role, hints in self.PATH_HINTS.items():
            if self._path_matches(file_path, hints):
                scores[role] += SignalWeight.PATH_HINT
        
        # Layer 4: 实体聚合（增强：考虑继承信号）
        if entities:
            entity_role_weights = defaultdict(float)
            for e in entities:
                # 如果角色是继承来的，给予额外权重
                weight_multiplier = 1.2 if e.inherited_from else 1.0
                entity_role_weights[e.role] += e.weight * e.confidence * weight_multiplier
            
            total_entity_weight = sum(entity_role_weights.values())
            if total_entity_weight > 0:
                for role, weight in entity_role_weights.items():
                    scores[role] += (weight / total_entity_weight) * SignalWeight.STRUCTURE
        
        if not scores:
            return RoleScore(Role.UNKNOWN, 0.0, {}, "No signals")
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_role, primary_score = sorted_scores[0]
        
        return RoleScore(
            primary_role=primary_role,
            primary_score=primary_score,
            all_scores=dict(scores),
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "Default"
        )
    
    def _path_matches(self, file_path: str, hints: List[str]) -> bool:
        if not file_path:
            return False
        path_lower = file_path.lower()
        return any(f'/{hint}/' in path_lower or path_lower.endswith(f'/{hint}') for hint in hints)
    
    # === 其他辅助方法 ===
    
    def _is_top_level_function(self, tree: ast.AST, func_node: ast.FunctionDef) -> bool:
        for node in tree.body:
            if node is func_node:
                return True
        return False
    
    def _is_empty_body(self, body: List[ast.stmt]) -> bool:
        for stmt in body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                continue
            return False
        return True
    
    def _has_not_implemented(self, node: ast.FunctionDef) -> bool:
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
        param_names = [arg.arg.lower() for arg in node.args.args]
        strong_io_keywords = ['request', 'response', 'req', 'resp', 'httpcontext']
        weak_io_keywords = ['event', 'payload', 'conn', 'session']
        if any(kw in param_names for kw in strong_io_keywords):
            return True
        has_weak_signal = any(kw in param_names for kw in weak_io_keywords)
        if node.returns:
            return_type = self._get_name(node.returns).lower()
            if any(kw in return_type for kw in ['response', 'json', 'httpresponse', 'result']):
                return True
            if has_weak_signal and 'dict' in return_type:
                return True
        func_name = node.name.lower()
        if has_weak_signal and any(kw in func_name for kw in ['handle', 'process', 'dispatch']):
            return True
        return False
    
    def _is_fixture_func(self, node: ast.FunctionDef) -> bool:
        for deco in node.decorator_list:
            deco_name = self._get_name(deco).lower()
            if any(kw in deco_name for kw in ['fixture', 'setup', 'teardown']):
                return True
        return False
    
    def _is_orchestrator_func(self, node: ast.FunctionDef) -> bool:
        calls = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call))
        control_flow = sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.For, ast.While)))
        return calls > 5 and control_flow < 3
    
    def _is_pure_function(self, node: ast.FunctionDef) -> bool:
        has_return = any(isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node))
        if not has_return:
            return False
        has_global = any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(node))
        if has_global:
            return False
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                func_name = self._get_name(n.func).lower()
                if any(kw in func_name for kw in ['open', 'read', 'write', 'print', 'request', 'fetch', 'send']):
                    return False
        return True
    
    def _is_config_assignment(self, node: ast.Assign) -> bool:
        if isinstance(node.value, (ast.Constant, ast.List, ast.Dict, ast.Set, ast.Tuple)):
            return True
        if isinstance(node.value, ast.Call):
            func_name = self._get_name(node.value.func).lower()
            if any(kw in func_name for kw in ['getenv', 'environ', 'config']):
                return True
        return False
    
    def _is_main_check(self, node: ast.If) -> bool:
        try:
            if isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "__name__":
                    return True
        except:
            pass
        return False
    
    def _is_factory_class(self, node: ast.ClassDef) -> bool:
        if 'factory' in node.name.lower():
            return True
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        factory_methods = sum(1 for m in methods if any(
            m.name.startswith(p) for p in ['create', 'make', 'build', 'get_']
        ))
        return factory_methods > len(methods) * 0.5
    
    def _has_business_logic(self, node: ast.ClassDef) -> bool:
        for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
            if self._has_business_logic_func(method):
                return True
        return False
    
    def _has_business_logic_func(self, node: ast.FunctionDef) -> bool:
        branches = sum(1 for n in ast.walk(node) if isinstance(n, (ast.If, ast.Match)))
        if branches >= 2:
            return True
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                func_name = self._get_name(n.func).lower()
                if any(kw in func_name for kw in ['getenv', 'config', 'settings']):
                    return True
        return False
    
    def _is_decorator_definition(self, node: ast.FunctionDef) -> bool:
        for n in ast.walk(node):
            if isinstance(n, ast.Return):
                if isinstance(n.value, ast.Name):
                    inner_funcs = [f.name for f in ast.walk(node) if isinstance(f, ast.FunctionDef) and f is not node]
                    if n.value.id in inner_funcs:
                        return True
        return False
    
    def _calculate_entity_weight(self, node: ast.AST, source_lines: List[str]) -> float:
        ast_nodes = sum(1 for _ in ast.walk(node))
        max_depth = self._get_max_depth(node)
        ast_complexity = ast_nodes * (1 + 0.1 * max_depth)
        start_line = node.lineno
        end_line = getattr(node, 'end_lineno', start_line) or start_line
        loc = end_line - start_line + 1
        normalized_ast = min(ast_complexity / 500, 1.0)
        normalized_loc = min(loc / 200, 1.0)
        return 0.6 * normalized_ast + 0.4 * normalized_loc
    
    def _get_max_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        max_child_depth = current_depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_max_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth
    
    def _get_name(self, node) -> str:
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
        return FileAnalysis(
            file_path=file_path,
            role_score=RoleScore(Role.UNKNOWN, 0.0, {}, reason)
        )


# ============================================================================
# CLI 接口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAACS 5.0 Code Role Classifier v8 (Two-Pass Analysis)")
    parser.add_argument("path", help="Path to analyze (file or directory)")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--entities", action="store_true", help="Show entity-level roles")
    parser.add_argument("--scores", action="store_true", help="Show all role scores")
    parser.add_argument("--purity", action="store_true", help="Show role purity")
    parser.add_argument("--single-file", action="store_true", help="Force single-file mode (no cross-file analysis)")
    parser.add_argument("--symbol-table", action="store_true", help="Show symbol table stats")
    args = parser.parse_args()

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
    
    # 决定使用哪种分析模式
    if os.path.isdir(target) and not args.single_file:
        # 项目模式：两阶段分析
        print(f"[Mode] Project analysis with cross-file inheritance")
        classifier = CodeRoleClassifier.create_project_analyzer(target, debug=args.debug)
        
        if args.symbol_table:
            st = classifier.symbol_table
            print(f"\n[Symbol Table Statistics]")
            print(f"  Files scanned: {len(st.files)}")
            print(f"  Classes indexed: {len(st.global_classes)}")
            
            # 显示角色分布
            role_counts = defaultdict(int)
            for cls in st.global_classes.values():
                role_counts[cls.final_role] += 1
            print(f"  Role distribution:")
            for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
                print(f"    {role.value}: {count}")
            print()
    else:
        # 单文件模式
        print(f"[Mode] Single-file analysis (no cross-file context)")
        classifier = CodeRoleClassifier(debug=args.debug)
    
    if os.path.isfile(target):
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
                inherited_info = f" [inherited from {e.inherited_from}]" if e.inherited_from else ""
                print(f"  [{e.entity_type}] {e.name:<25} -> {COLORS[e.role]}{e.role.value:<10}{RESET} "
                      f"(conf={e.confidence:.2f}) {e.reasoning}{inherited_info}")
    
    else:
        print(f"\n{'Path':<55} | {'Role':<12} | {'Score':>6} | {'Purity':>6} | {'Entities'}")
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
                        inherited_count = sum(1 for e in result.entities if e.inherited_from)
                        for e in result.entities:
                            role_counts[e.role] += 1
                        entity_summary = ", ".join([f"{r.value}:{c}" for r, c in role_counts.items()])
                        if inherited_count > 0:
                            entity_summary += f" ({inherited_count} inherited)"
                    
                    print(f"{rel_path[:55]:<55} | "
                          f"{COLORS[primary]}{primary.value:<12}{RESET} | "
                          f"{result.role_score.primary_score:>6.2f} | "
                          f"{result.role_purity:>6.2f} | "
                          f"{entity_summary}")

