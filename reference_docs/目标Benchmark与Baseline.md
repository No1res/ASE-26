基于你提出的 RAACS (PPR + Role Morphing) 方法，要冲击顶会（ICSE/FSE/ASE 或 ACL），实验设计必须非常扎实。你的方法核心优势在于 **“结构化召回（PPR）”** 和 **“高信息密度（Morphing）”** ，因此选取的 Benchmark 和 Baseline 必须能突显这两点。

以下是为你定制的实验方案建议：

### 1. 核心战场：数据集 (Benchmarks)

你需要覆盖“检索质量”和“生成质量”两个维度。

#### **Tier 0: 必须跑通的标准数据集 (The Standard)**

这俩是目前 Repository-level Code Completion 领域的“黄金标准”，审稿人必看。

- **RepoBench (ICLR 2024)**
    
    - **为什么选它**：专门为跨文件补全设计，分为 `Retrieval` (找得准不准) 和 `Completion` (补得对不对) 两个子任务。非常适合验证你 PPR 的检索效果。
        
    - **策略**：重点跑 `RepoBench-R` (Retrieval) 和 `RepoBench-C` (Completion)。
        
- **CrossCodeEval**
    
    - **为什么选它**：另一个主流的跨文件补全基准，包含 Python, Java, TypeScript 等多种语言。
        
    - **策略**：利用它的多语言特性，证明你的“角色感知”和“PPR”是跨语言通用的（Language-Agnostic）。
        

#### **Tier 1: 提升档次的高级数据集 (The Advanced)**

如果你想证明“不仅是文本匹配，而是真的懂逻辑”，加上这个：

- **ExecRepoBench (或类似的 Execution-based Benchmark)**
    
    - **为什么选它**：大多数 Benchmark 只看文本匹配 (Exact Match)。这个包含单元测试，能验证**代码能不能跑**。
        
    - **Insight**：你的 Morphing 策略保留了 Schema 和签名，理论上生成的代码调用会更准确，不容易瞎编参数，Execution Rate 应该是你的强项。
        

#### **Tier 2: 差异化数据集 (The Niche)**

- **DI-BENCH (Dependency Inference Benchmark)**
    
    - **为什么选它**：专门测试 LLM 是否理解依赖关系。你的 PPR 核心就是解决依赖，跑这个 Benchmark 你应该能碾压基于 Embedding 的方法。
        

---

### 2. 对手选择：Baselines (To Beat)

你需要挑选三类对手，分别代表不同的技术路线，以全方位证明你的优越性。

#### **A. 传统 RAG (Embedding-based)**

- **BM25 / OpenAI Embeddings (Standard RAG)**
    
    - **定位**：基础线 (Strawman)。
        
    - **目的**：证明“无脑切片 + 向量检索”是不够的，结构化信息（Structure）是必要的。
        

#### **B. 迭代式检索 (Iterative Retrieval)**

- **RepoCoder (EMNLP 2023)**
    
    - **定位**：前 SOTA，必须比较。
        
    - **原理**：先生成一点 -> 拿生成的去检索 -> 再生成。
        
    - **打点**：证明你的 PPR 一次性全局计算（Global Planning）比它的多次迭代（Local Search）更准、更快、更省 Token。
        

#### **C. 最强竞品 (Direct Competitor)**

- **GraphCoder (ASE 2024 / arXiv 2024)**
    
    - **定位**：**这是你最大的敌人**。
        
    - **原理**：它也用了 Code Context Graph 来做检索。
        
    - **你的胜负手**：GraphCoder 虽然用了图，但它检索回来的可能还是**原始代码块**。你的核心差异在于 **Morphing (变形)**。
        
    - **话术**：_“GraphCoder 只是利用图找到了代码，而我们不仅找到了，还根据角色对它进行了语义压缩。在同样的 Token 窗口下，我们能塞入 GraphCoder 3 倍的上下文信息。”_
        

---

### 3. 评价指标 (Metrics)

除了常规指标，你需要引入**凸显你特性的指标**。

#### **常规指标**

- **Exact Match (EM)**: 严格匹配。
    
- **CodeBLEU**: 考虑语法结构的模糊匹配。
    

#### **杀手锏指标 (你的 Highlight)**

- **Token Efficiency (Token 效率)**
    
    - **定义**：达到相同 EM 分数所需的平均 Prompt Token 数。
        
    - **预期结果**：你应该能画出一条曲线，显示你的方法在只需 **500 Tokens** 时就能达到 Embedding 方法 **2000 Tokens** 的效果。这是证明 Morphing 价值的最强证据。
        
- **Dependency Coverage (依赖覆盖率)**
    
    - **定义**：Target 代码中实际用到的依赖，有多少被包含在了 Context 里？
        
    - **预期结果**：PPR 应该能挖掘出 Hop=3 甚至 Hop=4 的深层依赖，而 Baseline 往往在 Hop=1 处截断。
        

### 总结建议

|**实验模块**|**推荐内容**|**核心目的**|
|---|---|---|
|**主战场**|**RepoBench**|证明综合实力 (SOTA)|
|**基础对比**|**vs. RepoCoder**|证明比“迭代式”更优|
|**硬核对比**|**vs. GraphCoder**|证明 **Morphing (变形)** 比纯检索更强|
|**关键图表**|**Performance vs. Token Cost**|证明极高的**信息密度** (省钱又快)|

如果你能复现 **GraphCoder** 并在 **Token Efficiency** 上大幅击败它，这就稳稳是一篇 **ASE** 或 **ICSE** 级别的论文了。