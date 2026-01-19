# AST Analyzer v8.1 Changelog

**版本**: v8.1  
**日期**: 2025-12-04  
**文件**: `raacs/ast_analyzer.py`

## 概述

本次重构针对符号表解析和继承传播机制进行了系统性增强，解决了以下核心问题：
1. 全局类索引的命名冲突风险
2. `import pkg as alias; class A(alias.Base)` 形式的解析困难
3. 弱信号（名称 hint）无法被强信号（继承）覆盖
4. 缺乏角色来源追踪

---

## 新增数据结构

### 1. `RoleSource` 枚举

```python
class RoleSource(Enum):
    INITIAL_FRAMEWORK = "initial_framework"  # 框架指纹（强信号）
    INITIAL_DECORATOR = "initial_decorator"  # 装饰器（强信号）
    INITIAL_NAME = "initial_name"            # 名称 hint（弱信号）
    INHERITED = "inherited"                  # 直接继承（强信号）
    PROPAGATED = "propagated"                # 传播推断（中等信号）
    STRUCTURAL = "structural"                # 结构模式（中等信号）
    UNKNOWN = "unknown"                      # 未知来源
```

**用途**: 追踪每个类角色的判定来源，用于后续权重调整和弱信号覆盖判断。

### 2. `ROLE_SOURCE_STRENGTH` 映射

```python
ROLE_SOURCE_STRENGTH = {
    RoleSource.INITIAL_FRAMEWORK: 0.95,
    RoleSource.INITIAL_DECORATOR: 0.90,
    RoleSource.INHERITED: 0.85,
    RoleSource.PROPAGATED: 0.75,
    RoleSource.STRUCTURAL: 0.65,
    RoleSource.INITIAL_NAME: 0.50,
    RoleSource.UNKNOWN: 0.0,
}
```

**用途**: 定义各信号来源的强度，用于判断是否允许覆盖。

### 3. `BaseInfo` 结构

```python
@dataclass
class BaseInfo:
    raw: str                    # 原始字符串，如 "models.Model"
    simple: str                 # 简单名，如 "Model"
    head: str                   # 头部（可能是别名），如 "models"
    qual_candidates: List[str]  # 可能的 FQN 候选
```

**用途**: 结构化存储基类信息，支持多候选 FQN 解析。

---

## 数据结构修改

### `ClassSymbol` 增强

```python
@dataclass
class ClassSymbol:
    # 原有字段...
    base_infos: List[BaseInfo] = None            # 新增：结构化基类信息
    role_source: RoleSource = RoleSource.UNKNOWN # 新增：角色来源
    inherited_from_bases: List[str] = None       # 新增：支持多基类继承来源
```

### `ImportInfo` 增强

```python
@dataclass
class ImportInfo:
    # 原有字段...
    is_alias: bool = False  # 新增：是否有 as 别名
    
    def resolve_attr(self, attr: str) -> str:
        """解析属性访问，如 m.Base -> pkg.mod.Base"""
```

### `ProjectSymbolTable` 重构

```python
@dataclass
class ProjectSymbolTable:
    # 分离的索引结构
    global_classes_by_fqn: Dict[str, ClassSymbol]        # FQN -> ClassSymbol（唯一）
    global_classes_by_simple: Dict[str, List[ClassSymbol]]  # simple_name -> [ClassSymbol]
    
    # 增强的查找方法
    def get_class_by_name(...)  # 支持 alias.ClassName 解析
    def get_class_by_base_info(base_info: BaseInfo, ...)  # 使用 qual_candidates
```

---

## 核心逻辑改进

### 1. 分离全局索引

**问题**: 原实现将 FQN 和 simple name 混合存储在同一个 dict 中，存在冲突风险。

**解决**: 
- `global_classes_by_fqn`: 存储唯一的全限定名映射
- `global_classes_by_simple`: 存储简单名到类列表的映射（支持同名类）

### 2. 增强的类名解析

**问题**: `import pkg.mod as m; class A(m.Base)` 无法正确解析。

**解决**:
1. `FileSymbols.resolve_qualname()`: 解析本地名称到 FQN 候选列表
2. `_resolve_base_candidates()`: 预计算每个基类的 FQN 候选
3. `get_class_by_base_info()`: 优先使用预计算候选查找

### 3. 弱信号覆盖逻辑

**问题**: 名称 hint（如 `class MyModel`）产生的角色无法被强继承信号覆盖。

**解决**:
```python
# RolePropagator._propagate_one_round()
if class_symbol.final_role == Role.UNKNOWN:
    should_update = True
elif current_strength < WEAK_SIGNAL_THRESHOLD:  # 0.6
    if fused_confidence > current_strength:
        should_update = True
```

### 4. 多基类角色融合

**新增**: `RolePropagator._fuse_base_roles()` 方法

融合策略：
1. 所有基类角色相同 → 直接返回，置信度提升
2. 有已知框架基类角色 → 优先选择
3. 角色冲突 → 选择置信度最高的

---

## 方法签名变更

### `_determine_initial_role()`

```python
# 旧
def _determine_initial_role(self, node, file_symbols) -> Role

# 新
def _determine_initial_role(self, node, file_symbols) -> Tuple[Role, RoleSource]
```

---

## 向后兼容性

- `ProjectSymbolTable.global_classes` 属性保留，返回 `global_classes_by_fqn`
- `ClassSymbol.base_names` 保留，`base_infos` 在 `__post_init__` 中自动生成

---

## 测试验证

```bash
$ python role_classifier_v9.py repos_to_be_examined/federation/ --debug

[Init] Symbol table: 211 classes
[Init] Role sources: {'unknown': 121, 'initial_framework': 7, 'initial_name': 82, 'initial_decorator': 1}
[Init] Role propagation: 0 roles propagated
```

角色来源分布正常：
- `initial_framework`: 框架基类识别（如 TestCase, BaseModel）
- `initial_decorator`: 装饰器识别（如 @dataclass）
- `initial_name`: 名称 hint（如 class FooSchema）
- `unknown`: 无初始信号，待后续分析

