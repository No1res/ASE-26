这是一个非常落地的工程问题。将 **PPR（相关度量化）** 和 **代码角色（语义定性）** 结合起来，本质上是在构建一个**二维决策矩阵**。

简单来说：

- **PPR 分数 ($S_{ppr}$)** 决定了 **“排队优先级”**（谁先进 Context）。
    
- **代码角色 ($R$)** 决定了 **“展示形态”**（穿什么衣服进 Context）。
    

以下是具体的综合决策与执行框架：

---

### 1. 核心隐喻：VIP 晚宴的邀请策略

把构建上下文想象成组织一场**容量有限**（Token Limit）的晚宴：

1. **PPR** 是客人的 **身价/地位** —— 决定了谁在必须邀请的名单上。
    
2. **Role** 是客人的 **职业** —— 决定了他们需要带什么东西入场（厨师带刀、歌手带麦克风、老板带钱）。
    

我们的目标是：**按身价排序，按职业换装，直到房间塞满。**

---

### 2. 决策矩阵 (The Decision Matrix)

我们需要定义不同 **PPR 分段** 与 **角色** 交叉时的具体**形态变换（Morphing）策略**。

#### 2.1 定义 PPR 分段 (Tiers)

首先，根据 PPR 分数分布（通常是长尾分布），将候选节点划分为三个梯队：

- **Tier 1 (Core)**: 累计概率前 50% 或 Top 5 节点。（核心依赖）
    
- **Tier 2 (Relevant)**: 累计概率 50%-85% 或 Top 6-20 节点。（次要依赖）
    
- **Tier 3 (Marginal)**: 剩余节点。（边缘/微弱依赖）
    

#### 2.2 角色 x 分数 = 展示形态

| **角色 (Role)**                                  | **Tier 1 (Core) 极高相关性**                           | **Tier 2 (Relevant) 中等相关性**                             | **Tier 3 (Marginal) 边缘相关性**                 |
| ---------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------- |
| **Schema / Model**<br><br>  <br><br>_(数据结构)_   | **完整定义**<br><br>  <br><br>(字段 + 方法签名 + Docstring) | **仅数据契约**<br><br>  <br><br>(仅字段名与类型)                    | **仅类名**<br><br>  <br><br>(作为类型提示存在)         |
| **Orchestrator**<br><br>  <br><br>_(业务逻辑)_     | **聚焦源码**<br><br>  <br><br>(保留核心逻辑 + 注释)           | **骨架化 (Skeleton)**<br><br>  <br><br>(保留控制流 if/try，隐去计算) | **丢弃 (Drop)**<br><br>  <br><br>(逻辑太远，无参考价值) |
| **Utility / Helper**<br><br>  <br><br>_(工具函数)_ | **白盒模式**<br><br>  <br><br>(签名 + Doc + **源码**)     | **黑盒模式**<br><br>  <br><br>(签名 + Doc + **1个用例**)         | **签名模式**<br><br>  <br><br>(仅函数名+参数类型)       |
| **Interface / Abs**<br><br>  <br><br>_(抽象定义)_  | **完整接口**<br><br>  <br><br>(所有抽象方法定义)              | **压缩接口**<br><br>  <br><br>(仅关键方法签名)                     | **丢弃**                                      |
| **Config / Const**<br><br>  <br><br>_(配置常量)_   | **全量值**                                           | **全量值**                                                 | **仅键名**                                     |

---

### 3. 执行算法流程 (The Algorithm)

这是一个 **贪心算法 (Greedy Strategy)**，优先满足高价值信息，同时动态压缩体积。

#### 步骤 1：全图计算与打标

1. 计算全图 PPR，得到每个节点的 $S_{ppr}$。
    
2. 通过 AST 特征或轻量级规则，给每个节点打上标签 $R$。
    
3. 生成一个候选列表 `Candidates`，包含 `(Node, Score, Role)`。
    

#### 步骤 2：基于“单位信息密度”的排序

不要直接按 PPR 排序，要引入 **“形态压缩比”** 的概念。

- Schema 的压缩比很高（Token 少，价值大）。
    
- Orchestrator 的压缩比低（Token 多，可能只有几行有用）。
    

修正排序公式：

$$Score_{final} = S_{ppr} \times W_{role}$$

- 给予 Schema 和 Interface 更高的权重 $W_{role}$（比如 1.2），因为它们是“理解代码的前提”，且通常比较短，性价比较高。
    
- 给予 Utility 普通权重（1.0）。
    
- 给予 Orchestrator 略低权重（0.9），因为它们太占 Token，除非 PPR 特别高否则不划算。
    

#### 步骤 3：动态填充循环 (The Filling Loop)

Python

```
def build_context(target_node, candidates, max_tokens=8000):
    current_tokens = 0
    context_blocks = []
    
    # 1. 强制包含：目标函数本身及上下文 (Prompt Engineering)
    target_block = get_target_context(target_node)
    current_tokens += count_tokens(target_block)
    
    # 2. 对候选列表按 Score_final 排序
    sorted_candidates = sort_by_score(candidates)
    
    # 3. 贪心填充
    for node in sorted_candidates:
        if current_tokens >= max_tokens:
            break
            
        # --- 核心决策逻辑 Start ---
        
        # 判断所处梯队
        tier = get_tier(node.ppr_score) 
        
        # 根据 (Role, Tier) 决定形态
        morph_strategy = DECISION_MATRIX[node.role][tier]
        
        # 如果策略是 Drop，直接跳过
        if morph_strategy == "DROP":
            continue
            
        # 执行形态变换 (AST操作 / LLM Summary)
        content = morph_code(node.source_code, strategy=morph_strategy)
        
        # --- 核心决策逻辑 End ---
        
        token_cost = count_tokens(content)
        
        # 预算检查
        if current_tokens + token_cost <= max_tokens:
            context_blocks.append(content)
            current_tokens += token_cost
        else:
            # 尝试降级策略 (如果放不下完整版，试着放个更压缩的版本？)
            fallback_strategy = downgrade(morph_strategy)
            fallback_content = morph_code(node.source_code, strategy=fallback_strategy)
            if current_tokens + count_tokens(fallback_content) <= max_tokens:
                context_blocks.append(fallback_content)
                current_tokens += count_tokens(fallback_content)
    
    return assemble(context_blocks, target_block)
```

---

### 4. 关键 Insight：为什么这种组合是“降维打击”？

#### **1. 解决了“相关但冗余”的问题**

- **场景**：PPR 算出 10 个相关的 Utils 函数。
    
- **传统做法**：只要相关度 > 0.5 全塞进去，Token 爆炸。
    
- **组合策略**：识别出它们是 `Util` 角色，且处于 `Tier 2`。根据矩阵，策略是 **“Signature + 1-Shot Example”**。
    
- **效果**：原本需要 2000 Token 的 10 个函数源码，变成了只需要 300 Token 的接口列表。**你省下了 Token 空间去放更关键的业务逻辑。**
    

#### **2. 解决了“深层依赖缺失”的问题**

- **场景**：PPR 挖出了 Hop=4 的 `UserSchema`，分值很高。
    
- **传统做法**：因为它物理距离远，容易被忽略；或者因为是 Schema，被当成普通代码塞进去。
    
- **组合策略**：PPR 把它排在前面，角色识别它为 `Schema`。策略是 **“完整定义”**。
    
- **效果**：模型即使没看到中间层的代码，也能精确知道 `User` 对象有哪些属性，从而写出正确的 `user.is_vip` 逻辑。
    

#### **3. 实现了“Token 预算的帕累托最优”**

这不仅仅是筛选，而是**信息重构**。

- 对于模型“只要看一眼就知道怎么用”的代码（Utils），我们只给“一眼”（签名）。
    
- 对于模型“必须理解内部逻辑”的代码（Orchestrator），我们给“骨架”。
    
- 对于模型“必须严丝合缝”的数据（Schema），我们给“契约”。
    

这就是“最小充分上下文”的终极形态：

Content = $\sum$ (Top PPR Nodes $\times$ Role-Optimized Representation)