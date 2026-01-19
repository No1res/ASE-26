import ast
import os
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# === 核心定义 ===

class Role(Enum):
    TEST = "TEST"             # 验证
    INTERFACE = "INTERFACE"   # 抽象/接口
    SCHEMA = "SCHEMA"         # 数据定义
    ADAPTER = "ADAPTER"       # IO/适配
    CONFIG = "CONFIG"         # 配置/常量
    SCRIPT = "SCRIPT"         # 脚本/入口
    UTIL = "UTIL"             # 工具/纯计算
    LOGIC = "LOGIC"           # 核心业务逻辑
    UNKNOWN = "UNKNOWN"

@dataclass
class EntityScore:
    """单个实体（类/函数）的角色评分"""
    name: str
    role: Role
    confidence: float
    weight: int  # 权重（基于AST节点数/行数）
    features: Dict[str, Any]

@dataclass
class FileRoleResult:
    """文件级最终分析结果"""
    file_path: str
    primary_role: Role
    primary_confidence: float
    secondary_roles: List[Tuple[Role, float]]
    entities: List[EntityScore]
    reasoning: str = "Rule-based aggregation"

# === 具体的特征提取器 ===

class FeatureExtractor(ast.NodeVisitor):
    def __init__(self):
        self.features = defaultdict(int)
        self.io_keywords = {'request', 'response', 'session', 'api', 'json', 'xml', 'db', 'query'}
    
    def visit_Call(self, node):
        self.features['calls'] += 1
        # 简单推断调用类型
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ['get', 'post', 'put', 'delete', 'execute']:
                self.features['io_calls'] += 1
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.features['assertions'] += 1
        self.generic_visit(node)
    
    def visit_Raise(self, node):
        self.features['raises'] += 1
        if isinstance(node.exc, ast.Name) and 'NotImplemented' in node.exc.id:
            self.features['abstract_raises'] += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.features['branches'] += 1
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.features['loops'] += 1
        self.generic_visit(node)

    def visit_Return(self, node):
        self.features['returns'] += 1
        self.generic_visit(node)

# === 核心分类器 ===

class CodeRoleClassifierV4:
    """
    RAACS 4.0: Entity-First Aggregation Classifier
    策略：先识别局部(Entity)，再聚合全局(File)
    """
    
    def analyze_file(self, file_path: str) -> FileRoleResult:
        if not os.path.exists(file_path):
            return self._empty_result(file_path)
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            return self.analyze_source(source, file_path)
        except Exception as e:
            return self._empty_result(file_path, str(e))

    def analyze_source(self, source: str, file_path: str = "") -> FileRoleResult:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._empty_result(file_path, "Syntax Error")

        # 1. 实体级分析 (Bottom-Up)
        entities = []
        
        # 预处理：获取文件名特征
        filename = os.path.basename(file_path).lower()
        is_test_file = any(k in filename for k in ['test', 'spec', 'mock'])
        
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                entity_score = self._analyze_entity(node, is_test_file)
                entities.append(entity_score)
        
        # 2. 模块级特征 (Top-Down 补充)
        # 如果文件主要是顶层赋值（常量/配置），没有函数/类
        global_assigns = sum(1 for n in tree.body if isinstance(n, ast.Assign))
        has_entry = self._check_entry_point(tree)
        
        # 3. 聚合策略
        return self._aggregate_roles(file_path, entities, global_assigns, has_entry)

    def _analyze_entity(self, node: ast.AST, context_is_test: bool) -> EntityScore:
        """分析单个类或函数的角色"""
        
        # 1. 计算权重 (基于AST节点数量，代表代码量/复杂度)
        weight = sum(1 for _ in ast.walk(node))
        
        # 2. 提取微观特征
        extractor = FeatureExtractor()
        extractor.visit(node)
        f = extractor.features
        
        # 计算特定比率
        loc = weight  # 近似代码行数
        branch_density = f['branches'] / loc if loc > 0 else 0
        io_density = f['io_calls'] / loc if loc > 0 else 0
        
        # 3. 规则判定逻辑 (Insights 2 & 3: 结构特征 > 框架指纹)
        
        role = Role.UNKNOWN
        conf = 0.5
        
        # --- 类分析 ---
        if isinstance(node, ast.ClassDef):
            # 统计方法和字段
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            fields = [n for n in node.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
            
            # SCHEMA: 字段多，方法少，且方法多为简单的魔术方法
            if len(fields) > len(methods) * 1.5:
                role = Role.SCHEMA
                conf = 0.8
            # INTERFACE: 抽象方法多
            elif f['abstract_raises'] > 0 or not methods: 
                # 即使没有显式抛出 NotImplemented，只有 pass 的类也像接口
                pass_methods = sum(1 for m in methods if len(m.body) == 1 and isinstance(m.body[0], ast.Pass))
                if pass_methods == len(methods) and len(methods) > 0:
                    role = Role.INTERFACE
                    conf = 0.9
            # TEST: 测试类上下文或包含断言
            elif context_is_test or f['assertions'] > 0:
                role = Role.TEST
                conf = 0.9
            # ADAPTER / LOGIC: 看复杂度
            else:
                if io_density > 0.05: # 有一定密度的IO调用
                    role = Role.ADAPTER
                    conf = 0.7
                else:
                    role = Role.LOGIC # 默认是业务逻辑
                    conf = 0.6

        # --- 函数分析 ---
        elif isinstance(node, ast.FunctionDef):
            # TEST
            if context_is_test or node.name.startswith('test_'):
                role = Role.TEST
                conf = 0.95
            # ADAPTER: IO关键词 + 编排特征 (调用多，分支少)
            elif self._is_io_boundary_naming(node.name) or (f['calls'] > 3 and f['branches'] < 2):
                role = Role.ADAPTER
                conf = 0.7
            # UTIL: 纯计算 (分支多，无IO，无副作用推断困难但可近似)
            elif f['branches'] > 2 and f['io_calls'] == 0:
                role = Role.UTIL
                conf = 0.6
            # LOGIC
            else:
                role = Role.LOGIC
                conf = 0.5

        return EntityScore(
            name=node.name,
            role=role,
            confidence=conf,
            weight=weight,
            features=dict(f)
        )

    def _aggregate_roles(self, file_path: str, entities: List[EntityScore], 
                         global_assigns: int, has_entry: bool) -> FileRoleResult:
        """
        Insight 4: 实体聚合优于文件粒度
        使用加权投票机制
        """
        
        # 特殊情况1: 脚本入口
        if has_entry:
            return FileRoleResult(file_path, Role.SCRIPT, 1.0, [], entities, "Has __main__ entry")
            
        # 特殊情况2: 纯配置/常量文件 (无实体，只有赋值)
        if not entities and global_assigns > 0:
            return FileRoleResult(file_path, Role.CONFIG, 0.9, [], [], "Pure assignments")
            
        if not entities:
            return FileRoleResult(file_path, Role.UNKNOWN, 0.0, [], [], "Empty or unstructured")

        # 加权计分
        role_weights = defaultdict(float)
        total_weight = 0
        
        for e in entities:
            # 核心算法：分数 = 实体权重 * 实体置信度
            # 这样一个 100 行的 Logic 类会压倒 5 个 10 行的 Schema 类
            points = e.weight * e.confidence
            role_weights[e.role] += points
            total_weight += points
            
        if total_weight == 0:
            return FileRoleResult(file_path, Role.UNKNOWN, 0.0, [], entities)

        # 归一化并排序
        final_scores = [(r, s/total_weight) for r, s in role_weights.items()]
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        primary_role, primary_score = final_scores[0]
        secondary_roles = final_scores[1:]
        
        return FileRoleResult(
            file_path=file_path,
            primary_role=primary_role,
            primary_confidence=primary_score,
            secondary_roles=secondary_roles,
            entities=entities,
            reasoning=f"Aggregated from {len(entities)} entities with total weight {total_weight}"
        )

    def _is_io_boundary_naming(self, name: str) -> bool:
        keywords = ['handler', 'view', 'controller', 'request', 'response', 'api']
        return any(k in name.lower() for k in keywords)

    def _check_entry_point(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                try:
                    if (isinstance(node.test, ast.Compare) and 
                        isinstance(node.test.left, ast.Name) and 
                        node.test.left.id == "__name__"):
                        return True
                except: pass
        return False

    def _empty_result(self, path, reason=""):
        return FileRoleResult(path, Role.UNKNOWN, 0.0, [], [], reason)

    # === Insight 5: 混合智能 (LLM Assist) ===
    
    def llm_assist_prompt(self, file_result: FileRoleResult) -> str:
        """
        生成发给 LLM 的 Prompt。
        仅当 primary_confidence 低于阈值(如 0.4)时调用。
        """
        # 提取关键摘要，而不是发送全部代码
        summary = []
        for e in file_result.entities[:5]: # 只取权重最大的前5个实体
            summary.append(f"- {e.name} (Type: {e.role.name}, Weight: {e.weight}, IO_Calls: {e.features.get('io_calls', 0)})")
        
        summary_text = "\n".join(summary)
        
        prompt = f"""
        I am analyzing a Python source file: {os.path.basename(file_result.file_path)}.
        Static analysis is ambiguous. Here is the structural summary of top entities:
        
        {summary_text}
        
        Based on standard software architecture patterns (DDD, MVC, Hexagonal), 
        what is the most likely role of this file?
        Choose one: ADAPTER, LOGIC, SCHEMA, UTIL, CONFIG.
        """
        return prompt

# === CLI 运行演示 ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAACS 4.0 Entity-First Classifier")
    parser.add_argument("path", help="Path to analyze")
    parser.add_argument("--details", action="store_true", help="Show entity details")
    args = parser.parse_args()

    classifier = CodeRoleClassifierV4()
    
    # 颜色定义
    C = {
        Role.LOGIC: '\033[92m', Role.ADAPTER: '\033[93m', 
        Role.SCHEMA: '\033[94m', Role.TEST: '\033[90m',
        Role.UNKNOWN: '\033[0m'
    }
    R = '\033[0m'

    targets = [args.path] if os.path.isfile(args.path) else [
        os.path.join(r, f) for r, _, fs in os.walk(args.path) for f in fs if f.endswith('.py')
    ]

    print(f"{'File':<50} | {'Primary':<15} | {'Conf':<6} | {'Structure'}")
    print("-" * 100)

    for path in sorted(targets):
        res = classifier.analyze_file(path)
        
        # 构造结构描述字符串 (e.g., "Logic(80%) + Schema(20%)")
        structure_desc = []
        if res.entities:
            # 简化的分布描述
            roles = Counter([e.role.name for e in res.entities])
            total = len(res.entities)
            structure_desc = [f"{k}({v})" for k, v in roles.most_common(2)]
        
        role_str = f"{C.get(res.primary_role, '')}{res.primary_role.name}{R}"
        fname = os.path.relpath(path, args.path) if os.path.isdir(args.path) else os.path.basename(path)
        
        print(f"{fname[:50]:<50} | {role_str:<24} | {res.primary_confidence:.2f}   | {', '.join(structure_desc)}")

        if args.details and res.entities:
            for e in res.entities:
                print(f"  └─ {e.name:<30} -> {e.role.name:<10} (w={e.weight}, io={e.features.get('io_calls')})")