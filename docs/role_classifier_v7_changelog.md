# Role Classifier v7/v8 Changelog

> **注意**: v9 的变更记录已迁移到独立文件 `role_classifier_v9_changelog.md`

---

## V8: 两阶段分析 + 跨文件符号表

### 核心改进

**解决问题**：跨文件继承链的角色传递断裂

```python
# models.py
class User(BaseModel):  # ✅ 能识别为 SCHEMA

# services.py  
class SuperUser(User):  # V7: ❌ 误判为 LOGIC | V8: ✅ 继承 SCHEMA
```

### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    两阶段分析流程                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Stage 1: Symbol Collection (符号收集)                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │ • 扫描所有 .py 文件                              │   │
│  │ • 收集类定义、基类、导入关系                     │   │
│  │ • 初步角色判定（基于框架指纹）                   │   │
│  │ • 构建全局类索引                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                         ↓                               │
│  Stage 2: Role Propagation (角色传播)                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ • 迭代遍历未知角色的类                           │   │
│  │ • 解析基类的实际来源                             │   │
│  │ • 从已知角色的基类继承角色                       │   │
│  │ • 直到收敛（无新传播）                           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 新增数据结构

| 结构 | 用途 |
|------|------|
| `ClassSymbol` | 类符号信息（名称、文件、基类、角色） |
| `ImportInfo` | 导入信息（支持别名、相对导入） |
| `FileSymbols` | 文件级符号表 |
| `ProjectSymbolTable` | 项目级符号表 + 全局类索引 |
| `SymbolCollector` | 第一阶段：符号收集器 |
| `RolePropagator` | 第二阶段：角色传播器 |

### 使用方式

```bash
# 单文件模式（向后兼容）
python role_classifier_claude_v8.py path/to/file.py

# 项目模式（两阶段分析）
python role_classifier_claude_v8.py path/to/project/

# 查看符号表统计
python role_classifier_claude_v8.py path/to/project/ --symbol-table
```

### 效果验证

```
[Symbol Table Statistics]
  Files scanned: 115
  Classes indexed: 407
  Role distribution:
    UNKNOWN: 233
    TEST: 151    ← 通过继承 TestCase 识别
    SCHEMA: 13   ← 通过继承 BaseModel/Entity 识别
    ADAPTER: 10

entities/exceptions.py  | SCHEMA | 4 entities (4 inherited)
tests/test_outbound.py  | TEST   | 2 entities (2 inherited)
```

### 局限性

1. **动态继承无法处理**：如 `class Foo(get_base())`
2. **外部包基类**：非项目内的基类只能依赖 `KNOWN_BASE_ROLES` 列表
3. **性能开销**：大型项目首次扫描需要额外时间

---

# Role Classifier V7 Changelog

## V7.1 修复 (基于新 repo 测试)

### 修复的问题

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| `_i18n.py` 被判为 UNKNOWN | 得分 0.0 | CONFIG (2.0) |
| `version.py` 被判为 UNKNOWN | 得分 0.0 | CONFIG (2.0) |
| `forms.py` 被误判为 ADAPTER | ADAPTER (4.0) | **SCHEMA (7.89)** |
| `paths.py` 被误判为 SCRIPT | SCRIPT (3.75) | **LOGIC (2.54)** |
| `operators.py` 被误判为 ADAPTER | ADAPTER (3.95) | **LOGIC (1.94)** |
| `utils.py` (planb) 被误判为 ADAPTER | ADAPTER | **LOGIC (3.25)** |

### 具体改动

1. **Django 细粒度处理**
   - 添加 `DJANGO_SUBMODULE_ROLES` 映射表
   - `django.forms` → SCHEMA
   - `django.db.models` → SCHEMA  
   - `django.views` → ADAPTER
   - `django.urls` → ADAPTER

2. **SCRIPT 误判修复**
   - 如果有 `__main__` 但同时有 >3 个类/函数定义，降低 SCRIPT 权重
   - 这些文件主要是库，`__main__` 只是调试入口

3. **简单模块赋值处理**
   - 添加 `module_assign_count` 特征
   - 没有函数/类但有模块级赋值 → CONFIG

4. **IO 边界判断优化**
   - `ctx`/`context` 参数不再直接触发 IO 边界判断（太通用）
   - 分离强信号（request/response）和弱信号（event/payload）
   - 弱信号需配合返回类型或函数名

---

## 版本对比

| 特性 | V6 | V7 |
|------|-----|-----|
| 信号层级 | 结构模式为主 | 三层加权：框架(4.0) > 结构(2.5) > 路径(1.5) |
| 框架指纹 | 去框架指纹化 | **恢复框架指纹作为强信号** |
| 字段识别 | 只统计 `AnnAssign` | 同时统计 `Assign` 和 `AnnAssign` |
| 实体分析 | 只分析类 | **类 + 顶层函数** |
| 混合度计算 | 无 | **基于角色兼容性矩阵** |
| 权重定义 | 未定义 | **0.6×AST + 0.4×LOC** |
| 工厂类区分 | 统一归为 UTIL | **便捷工厂(UTIL) vs 策略工厂(LOGIC)** |
| 空数据类识别 | 误判为 LOGIC | **正确识别为 SCHEMA** |

## 核心改进

### 1. 三层信号加权体系

```python
class SignalWeight:
    FRAMEWORK = 4.0      # 框架指纹（最强信号）
    STRUCTURE = 2.5      # 结构模式
    PATH_HINT = 1.5      # 路径提示
```

### 2. 恢复框架指纹作为强信号

```python
FRAMEWORK_SIGNATURES = {
    'web_frameworks': {'django', 'flask', 'fastapi', ...},   # -> ADAPTER
    'data_frameworks': {'pydantic', 'sqlalchemy', ...},      # -> SCHEMA
    'test_frameworks': {'pytest', 'unittest', ...},          # -> TEST
    'async_frameworks': {'celery', 'dramatiq', ...},         # -> ADAPTER
}
```

### 3. 增强的 SCHEMA 识别

**改进前**：
- 只统计 `ast.AnnAssign`（带类型注解的字段）
- `methods < 3` 的硬编码阈值
- 继承自数据基类的空类被误判为 LOGIC

**改进后**：
```python
# 同时统计普通赋值和类型注解赋值
fields = []
for n in node.body:
    if isinstance(n, ast.AnnAssign):
        fields.append(n)
    elif isinstance(n, ast.Assign):
        # 排除 __slots__ 等魔术属性
        ...

# 基类语义提示
data_base_hints = ['entity', 'model', 'schema', 'dto', 'mixin', 'base']

# 空壳数据类识别
is_empty_data_class = (
    (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)) or
    (init_only and len(fields) == 0)
) and inherits_data_base
```

### 4. 角色兼容性矩阵

用于计算混合度（purity），决定使用文件级还是实体级 Morphing：

```python
ROLE_COMPATIBILITY = {
    # 高兼容组：数据定义类
    (Role.SCHEMA, Role.CONFIG): 0.9,
    (Role.SCHEMA, Role.INTERFACE): 0.8,
    
    # 中兼容组：行为类
    (Role.UTIL, Role.LOGIC): 0.5,
    (Role.ADAPTER, Role.LOGIC): 0.4,
    
    # 低兼容组：跨类型
    (Role.SCHEMA, Role.LOGIC): 0.2,
    ...
}
```

### 5. 实体级权重计算

```python
def calculate_entity_weight(node, source_lines) -> float:
    """权重 = 0.6×AST复杂度 + 0.4×LOC"""
    ast_complexity = ast_nodes * (1 + 0.1 * max_depth)
    normalized_ast = min(ast_complexity / 500, 1.0)
    normalized_loc = min(loc / 200, 1.0)
    return 0.6 * normalized_ast + 0.4 * normalized_loc
```

### 6. 工厂类智能区分

```python
# 便捷工厂（无业务决策）-> UTIL
class UserFactory:
    @staticmethod
    def create_admin() -> User:
        return User(role="admin")

# 策略工厂（有业务决策）-> LOGIC
class PaymentProcessorFactory:
    def create(self, country: str):
        if country == "CN":
            return AlipayProcessor()
        # ... 业务判断逻辑
```

## 测试结果对比

### `federation/entities/base.py`

| 版本 | 主角色 | 得分 | Purity | 问题 |
|------|-------|------|--------|------|
| V6 | **SCRIPT** | 1.50 | N/A | ❌ 严重误判 |
| V7 | **SCHEMA** | 4.00 | 1.00 | ✅ 正确 |

### `federation/entities/mixins.py`

| 版本 | 主角色 | Purity | 实体分布 |
|------|-------|--------|---------|
| V6 | SCHEMA | 0.18 | LOGIC:6, SCHEMA:4 |
| V7 | SCHEMA | **1.00** | SCHEMA:10 |

## 使用方法

```bash
# 单文件分析
python role_classifier_claude_v7.py path/to/file.py --entities --scores --purity

# 目录分析
python role_classifier_claude_v7.py path/to/directory/

# 输出字段说明
# Role     - 主角色
# Score    - 主角色得分
# Purity   - 角色纯度（0-1，越高越纯）
# Entities - 实体级角色分布
```

## Morphing 决策建议

```
if purity > 0.75:
    使用文件级 Morphing（按主角色策略处理整个文件）
else:
    使用实体级 Morphing（对每个实体分别处理）
```

