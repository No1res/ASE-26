# Role Classifier v9 Changelog

**版本**: v9.1  
**日期**: 2024-12-04

---

## v9.1: 动态阈值系统

### 问题

v9.0 中存在大量硬编码常量：
```python
# 问题代码示例
if in_deg >= max_in * 0.3 or in_deg >= 5:    # 0.3? 5?
if in_deg > 10 and out_deg <= 3:              # 10? 3?
if adapter_test_ratio > 0.7:                  # 0.7?
```

**后果**：不同规模仓库的鲁棒性差。小仓库可能没有 `in_deg >= 5` 的模块。

### 解决方案：基于统计分布的动态阈值

```python
@dataclass
class DynamicThresholds:
    """所有阈值基于仓库统计分布动态计算"""
    
    hub_in_degree_threshold: float      # 入度 >= P90 (或 P85 for small)
    orchestrator_out_degree_threshold: float
    infra_in_degree_threshold: float    # 入度 >= P75
    infra_out_degree_threshold: float   # 出度 <= median
    app_layer_caller_ratio: float       # 根据规模调整
    ...
```

### 仓库规模分类

| 规模 | 模块数 | HUB 百分位 | ORCH 百分位 | APP 比例 |
|------|--------|------------|-------------|----------|
| tiny | <30 | P80 | P80 | 0.50 |
| small | 30-100 | P85 | P85 | 0.55 |
| medium | 100-300 | P90 | P90 | 0.60 |
| large | 300-1000 | P92 | P92 | 0.65 |
| huge | >1000 | P95 | P95 | 0.70 |

### 实际效果

```
# federation (medium, 115 modules)
DynamicThresholds(
  HUB: in_degree >= 13.1 (P90)
  ORCHESTRATOR: out_degree >= 10.0 (P90)
  INFRA: in >= 4.0, out <= 4.0
)

# planb (small, 63 modules)  
DynamicThresholds(
  HUB: in_degree >= 6.3 (P85)    ← 自动降低
  ORCHESTRATOR: out_degree >= 4.8 (P85)
  INFRA: in >= 3.0, out <= 1.0
)
```

### 新增组件

| 组件 | 功能 |
|------|------|
| `RepositoryStats` | 仓库统计信息（入度/出度分布） |
| `DynamicThresholds` | 动态阈值配置 |
| `compute_repository_stats()` | 计算统计信息 |
| `DynamicThresholds.from_stats()` | 从统计信息生成阈值 |

---

## v9.0: 图结构角色层 + 三层融合

**日期**: 2024-12-04

---

## 核心改进：图结构角色层 + 三层融合

### 背景

> 很多架构角色是"在网络中的位置"定义的，而不是"自己长什么样"定义的。

**之前的局限**（v7/v8）：
- 纯粹的单文件 AST 分析，缺乏交互语义
- 不看调用图（谁调用它，它调用谁）
- 不看跨文件角色关系
- 无法识别"结构中心性角色"

### 新架构：三层信号融合

```
┌─────────────────────────────────────────────────────────────────┐
│                    三层信号融合架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: AST 层                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • 文件内部结构分析                                        │   │
│  │ • 框架指纹 + 结构模式 + 路径提示                          │   │
│  │ → 回答"文件做什么"                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  Layer 2: 符号表层                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • 跨文件继承关系                                          │   │
│  │ • 角色传播（子类继承父类角色）                            │   │
│  │ → 回答"文件是什么的子类"                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  Layer 3: 图结构层 (NEW!)                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • 依赖图分析（pydeps 自动生成）                           │   │
│  │ • 图角色: HUB / ORCHESTRATOR / BRIDGE / LEAF / SINK      │   │
│  │ • 架构层次推断                                            │   │
│  │ → 回答"文件在哪里"                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  Fusion: 三层融合                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ • 融合规则表                                              │   │
│  │ • 架构层次调整                                            │   │
│  │ → 最终架构角色                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 图结构角色 (GraphRole)

| 角色 | 定义 | 特征 | 含义 |
|------|------|------|------|
| **HUB** | 被广泛依赖的核心模块 | in-degree ≥ 30% max | 基础设施 |
| **ORCHESTRATOR** | 协调多个模块 | out-degree ≥ 30% max | 聚合/协调 |
| **BRIDGE** | 连接不同角色层 | 调用者/被调用者角色差异大 | 适配层 |
| **LEAF** | 只被调用，不调用其他 | in > 0, out = 0 | 纯工具/配置 |
| **SINK** | 只调用，不被调用 | in = 0, out > 0 | 入口/脚本 |
| **ISOLATE** | 孤立节点 | in = 0, out = 0 | 未使用 |

---

## 融合规则

```python
FUSION_RULES = {
    # HUB 节点：被广泛依赖
    (LOGIC, HUB):          (UTIL,   0.7, "High centrality → core utility"),
    (UNKNOWN, HUB):        (UTIL,   0.6, "Hub with unknown AST"),
    
    # ORCHESTRATOR 节点：协调多个模块
    (LOGIC, ORCHESTRATOR): (LOGIC,  1.0, "Orchestrator confirms business logic"),
    (UTIL, ORCHESTRATOR):  (LOGIC,  0.8, "Orchestrator → coordination logic"),
    (SCHEMA, ORCHESTRATOR):(SCHEMA, 0.9, "Schema with many deps (aggregation)"),
    
    # SINK 节点：只调用，不被调用（入口点）
    (LOGIC, SINK):         (SCRIPT, 0.5, "Sink node → entry point"),
    (ADAPTER, SINK):       (ADAPTER,1.0, "Entry adapter confirmed"),
    
    # LEAF 节点：只被调用，不调用其他
    (LOGIC, LEAF):         (UTIL,   0.6, "Leaf node → stateless utility"),
    (SCHEMA, LEAF):        (SCHEMA, 1.0, "Pure schema/entity"),
    (CONFIG, LEAF):        (CONFIG, 1.0, "Pure config"),
    
    # BRIDGE 节点：连接不同层
    (LOGIC, BRIDGE):       (ADAPTER,0.6, "Bridge → adapter layer"),
}
```

---

## 新特性

### 1. 自动依赖图生成

不再需要手动运行 pydeps，分析器会自动生成依赖图：

```bash
# 自动生成（默认）
python role_classifier_v9.py your_project/

# 保存依赖图供复用
python role_classifier_v9.py your_project/ --save-deps deps.json

# 使用预生成的依赖图
python role_classifier_v9.py your_project/ --dep-map deps.json

# 禁用自动生成
python role_classifier_v9.py your_project/ --no-auto-deps
```

### 2. 版本号支持

```bash
python role_classifier_v9.py --version
# 输出: v9.0
```

---

## 验证效果

| 文件路径 | AST 角色 | 图角色 | 融合角色 | 说明 |
|----------|----------|--------|----------|------|
| `entities/diaspora/mappers.py` | UTIL | ORCHESTRATOR | →ADAPTER | Interface layer + coord |
| `entities/matrix/enums.py` | LOGIC | LEAF | →UTIL | Stateless leaf |
| `entities/activitypub/views.py` | ADAPTER | ORCHESTRATOR | ADAPTER | Entry confirmed |
| `federation.utils` | UTIL | HUB | UTIL | Core utility hub |

---

## 使用方式

```bash
# 基本用法（自动生成依赖图）
python role_classifier_v9.py your_project/

# 分析单个文件
python role_classifier_v9.py your_project/ --file your_project/utils.py

# 显示详细信息
python role_classifier_v9.py your_project/ --debug --show-graph --show-fusion

# 保存依赖图
python role_classifier_v9.py your_project/ --save-deps project_deps.json
```

---

## 文件结构

| 文件 | 版本 | 功能 |
|------|------|------|
| `role_classifier_v9.py` | v9.0 | 三层融合集成分析器（主入口） |
| `role_classifier_claude_v8.py` | v8.0 | AST 分析 + 符号表层 |
| `graph_role_analyzer_v9.py` | v9.0 | 图结构角色分析器 |

---

## 依赖

- Python 3.8+
- pydeps（可选，用于自动生成依赖图）

```bash
pip install pydeps
```

---

## 版本历史

- **v9.0** (2024-12-04): 三层融合 + 自动依赖图生成
- **v8.0**: 两阶段分析 + 跨文件符号表
- **v7.x**: 单文件 AST 分析增强

