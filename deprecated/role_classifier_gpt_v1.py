import ast
import os
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

# ============================================================
#                      ROLE ENUM
# ============================================================

class Role(Enum):
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

# ============================================================
#                STRUCTURAL PATTERN (CLEANED)
# ============================================================

@dataclass
class StructuralPattern:
    """
    所有结构强度分数输出为 0~1 soft-score
    """
    # === Interaction Patterns ===
    io_score: float = 0.0              # IO Boundary intensity
    orchestrator_score: float = 0.0     # 调用多、逻辑少 → 编排强度

    # === Abstraction & Interface ===
    abstraction_ratio: float = 0.0      # 空方法/抽象方法比例 (0~1)

    # === Data Definition ===
    data_definition_ratio: float = 0.0  # AnnAssign 数量 / (AnnAssign+Methods)

    # === Computation / Control Flow ===
    compute_score: float = 0.0          # 高计算密度 + 控制流强度

    # === Testing ===
    assertion_density: float = 0.0      # assert 密度
    fixture_ratio: float = 0.0          # fixture 方法比例

    # === Config ===
    constant_ratio: float = 0.0         # 大写常量比例
    is_pure_declaration: bool = False

    # === Script ===
    has_entry_point: bool = False
    imperative_ratio: float = 0.0       # expr_count / defs

# ============================================================
#                   ROLE SCORE
# ============================================================

@dataclass
class RoleScore:
    primary_role: Role
    primary_score: float
    secondary_roles: List[Tuple[Role, float]] = field(default_factory=list)
    reasoning: str = ""

@dataclass
class EntityRole:
    name: str
    type: str        # 'class' or 'function'
    role: Role
    score: float
    lineno: int
    pattern: StructuralPattern

@dataclass
class FileAnalysis:
    file_path: str
    role_score: RoleScore
    entity_roles: List[EntityRole]
    pattern: StructuralPattern
    features: Dict[str, Any]

# ============================================================
#             MAIN CLASSIFIER
# ============================================================

class CodeRoleClassifier:

    PATH_HINTS = {
        Role.TEST: ['test', 'tests', '__tests__', 'spec'],
        Role.CONFIG: ['config', 'settings'],
        Role.SCRIPT: ['scripts', 'bin', 'tools', 'cli'],
        Role.ADAPTER: ['api', 'controller', 'handler', 'endpoint'],
        Role.SCHEMA: ['models', 'schemas', 'entities'],
        Role.UTIL: ['utils', 'helpers', 'common']
    }

    # ========================================================
    #                  PIPELINE ENTRY
    # ========================================================

    def __init__(self, debug=False, use_llm=False):
        self.debug = debug
        self.use_llm = use_llm

    def analyze_file(self, file_path: str) -> FileAnalysis:
        if not os.path.exists(file_path):
            return FileAnalysis(file_path, RoleScore(Role.UNKNOWN, 0.0), [], StructuralPattern(), {})

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            return self.analyze_source(code, file_path=file_path)
        except Exception as e:
            if self.debug:
                print(f"[Error] {e}")
            return FileAnalysis(file_path, RoleScore(Role.UNKNOWN, 0.0), [], StructuralPattern(), {})

    def analyze_source(self, code: str, file_path="") -> FileAnalysis:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return FileAnalysis(file_path, RoleScore(Role.UNKNOWN, 0.0), [], StructuralPattern(), {})

        features = self._extract_basic_features(tree, file_path)
        pattern = self._analyze_structural_pattern(tree, features)

        # === Entities ===
        entities = self._analyze_entities(tree)

        # === Score by pattern ===
        score = self._score_by_pattern(pattern, features, file_path)

        return FileAnalysis(file_path, score, entities, pattern, features)

    # ========================================================
    #         BASIC STATIC COUNTS (NO FRAMEWORK DEP)
    # ========================================================

    def _extract_basic_features(self, tree, file_path) -> Dict[str, Any]:
        f = defaultdict(int)

        f['filename'] = os.path.basename(file_path).lower()
        f['dir'] = os.path.dirname(file_path).lower()

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

        # pure __init__
        f['is_pure_init'] = (f['filename'] == "__init__.py"
                             and f['function_count'] == 0
                             and f['class_count'] == 0
                             and f['control_flow_count'] == 0)

        return f

    # ========================================================
    #                 STRUCTURAL PATTERN (SOFT)
    # ========================================================

    def _analyze_structural_pattern(self, tree, features) -> StructuralPattern:
        p = StructuralPattern()

        total_funcs = 0
        empty_funcs = 0
        data_fields = 0
        data_methods = 0
        asserts = 0
        fixtures = 0
        io_hits = 0
        orchestrations = 0
        constants = 0

        for node in ast.walk(tree):
            # === Classes ===
            if isinstance(node, ast.ClassDef):
                fields = sum(1 for n in node.body if isinstance(n, ast.AnnAssign))
                methods = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                data_fields += fields
                data_methods += methods

            # === Functions ===
            elif isinstance(node, ast.FunctionDef):
                total_funcs += 1

                # empty/abstract
                if self._is_empty_body(node.body):
                    empty_funcs += 1

                # assert
                asserts += sum(1 for n in ast.walk(node) if isinstance(n, ast.Assert))
                if self._is_fixture(node):
                    fixtures += 1

                # io boundary
                io_hits += self._io_boundary_score(node)

                # orchestrator?
                orchestrations += self._orchestrator_score(node)

            # === Assignment ===
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        constants += 1

        # === abstraction ===
        if total_funcs > 0:
            p.abstraction_ratio = empty_funcs / total_funcs

        # === data definition ===
        if (data_fields + data_methods) > 0:
            p.data_definition_ratio = data_fields / (data_fields + data_methods)

        # === testing ===
        if total_funcs > 0:
            p.assertion_density = asserts / total_funcs
            p.fixture_ratio = fixtures / total_funcs

        # === IO / orchestrator ===
        p.io_score = min(io_hits / max(total_funcs, 1), 1.0)
        p.orchestrator_score = min(orchestrations / max(total_funcs, 1), 1.0)

        # === computation intensity ===
        if total_funcs > 0:
            p.compute_score = min(features['control_flow_count'] / total_funcs, 1.0)

        # === config ===
        p.constant_ratio = (constants / features['assign_count']) if features['assign_count'] else 0
        p.is_pure_declaration = (
            features['assign_count'] > 0
            and features['function_count'] == 0
            and features['class_count'] == 0
        )

        # === script ===
        p.has_entry_point = features['has_main_entry']
        defs = features['class_count'] + features['function_count']
        p.imperative_ratio = features['expr_count'] / (defs + 1)

        return p

    # ========================================================
    #               SOFT SCORES FOR PATTERNS
    # ========================================================

    def _io_boundary_score(self, fn: ast.FunctionDef) -> float:
        """
        Soft-score based IO boundary detection.
        """
        score = 0

        param_names = [a.arg.lower() for a in fn.args.args]
        if any(k in param_names for k in ["req", "request", "ctx", "context", "event"]):
            score += 0.4

        # return type
        if fn.returns:
            rt = self._get_name(fn.returns).lower()
            if any(k in rt for k in ["response", "json", "dict", "result"]):
                score += 0.3

        # body
        for n in ast.walk(fn):
            if isinstance(n, ast.Call):
                func_name = self._get_name(n.func).lower()
                if any(k in func_name for k in ["serialize", "json", "parse"]):
                    score += 0.2

        return min(score, 1.0)

    def _orchestrator_score(self, fn: ast.FunctionDef) -> float:
        calls = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Call))
        controls = sum(1 for n in ast.walk(fn) if isinstance(n, (ast.If, ast.For, ast.While)))
        if calls > 4 and controls < 2:
            return 0.7
        if calls > 2:
            return 0.4
        return 0.0

    # ========================================================
    #             ENTITY-LEVEL ANALYSIS (FUNCTION + CLASS)
    # ========================================================

    def _analyze_entities(self, tree) -> List[EntityRole]:
        entities = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                p = StructuralPattern()
                fields = sum(1 for n in node.body if isinstance(n, ast.AnnAssign))
                methods = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))
                if fields + methods > 0:
                    p.data_definition_ratio = fields / (fields + methods)
                if methods > 0:
                    empty_m = sum(1 for n in node.body
                                  if isinstance(n, ast.FunctionDef) and self._is_empty_body(n.body))
                    p.abstraction_ratio = empty_m / methods

                role = self._entity_role_from_pattern(p)
                entities.append(EntityRole(node.name, "class", role, 1.0, node.lineno, p))

            elif isinstance(node, ast.FunctionDef):
                p = StructuralPattern()
                p.io_score = self._io_boundary_score(node)
                p.orchestrator_score = self._orchestrator_score(node)
                p.abstraction_ratio = 1.0 if self._is_empty_body(node.body) else 0.0
                role = self._entity_role_from_pattern(p)
                entities.append(EntityRole(node.name, "function", role, 1.0, node.lineno, p))

        return entities

    def _entity_role_from_pattern(self, p: StructuralPattern) -> Role:
        if p.abstraction_ratio > 0.7:
            return Role.INTERFACE
        if p.data_definition_ratio > 0.7:
            return Role.SCHEMA
        if p.io_score > 0.5 or p.orchestrator_score > 0.5:
            return Role.ADAPTER
        return Role.LOGIC

    # ========================================================
    #                  SCORE ROLE BY PATTERN
    # ========================================================

    def _score_by_pattern(self, p: StructuralPattern, f, file_path) -> RoleScore:
        scores = defaultdict(float)

        # TEST
        scores[Role.TEST] += p.assertion_density * 4
        scores[Role.TEST] += p.fixture_ratio * 3
        if "test" in f['filename']:
            scores[Role.TEST] += 1
        if self._path_hint(file_path, Role.TEST):
            scores[Role.TEST] += 1.5

        # SCRIPT
        if p.has_entry_point:
            scores[Role.SCRIPT] += 4
        scores[Role.SCRIPT] += min(p.imperative_ratio, 1) * 2
        if self._path_hint(file_path, Role.SCRIPT):
            scores[Role.SCRIPT] += 1.5

        # NAMESPACE
        if f['is_pure_init']:
            scores[Role.NAMESPACE] += 5

        # CONFIG
        if p.is_pure_declaration:
            scores[Role.CONFIG] += 4
        scores[Role.CONFIG] += p.constant_ratio * 2
        if self._path_hint(file_path, Role.CONFIG):
            scores[Role.CONFIG] += 1

        # INTERFACE
        scores[Role.INTERFACE] += p.abstraction_ratio * 4

        # SCHEMA
        scores[Role.SCHEMA] += p.data_definition_ratio * 4
        scores[Role.SCHEMA] += (1 - p.compute_score) * 2
        if self._path_hint(file_path, Role.SCHEMA):
            scores[Role.SCHEMA] += 1

        # ADAPTER
        scores[Role.ADAPTER] += p.io_score * 4
        scores[Role.ADAPTER] += p.orchestrator_score * 2
        if self._path_hint(file_path, Role.ADAPTER):
            scores[Role.ADAPTER] += 1.5

        # UTIL
        scores[Role.UTIL] += (1 - p.orchestrator_score) * 2
        scores[Role.UTIL] += (1 - p.io_score) * 1
        if self._path_hint(file_path, Role.UTIL):
            scores[Role.UTIL] += 1

        # LOGIC (fallback)
        scores[Role.LOGIC] += p.compute_score * 2

        # === final selection ===
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary, score = sorted_scores[0]
        threshold = score * 0.5
        secondary = [(r, s) for r, s in sorted_scores[1:] if s > threshold][:2]

        return RoleScore(primary, score, secondary)

    # ========================================================
    #                      HELPERS
    # ========================================================

    def _path_hint(self, path, role: Role):
        if role not in self.PATH_HINTS:
            return False
        path = path.lower()
        return any(h in path for h in self.PATH_HINTS[role])

    def _get_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return self._get_name(node.func)
        return ""

    def _is_fixture(self, fn):
        for d in fn.decorator_list:
            name = self._get_name(d).lower()
            if "fixture" in name or "setup" in name or "teardown" in name:
                return True
        return False

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


# ============================================================
#                             CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAACS Structural Code Classifier")
    parser.add_argument("path", help="file or directory")
    args = parser.parse_args()

    clf = CodeRoleClassifier()

    target = os.path.abspath(args.path)

    if os.path.isfile(target):
        r = clf.analyze_file(target)
        print("Primary:", r.role_score.primary_role.value, r.role_score.primary_score)
        print("Secondary:", r.role_score.secondary_roles)
    else:
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if f.endswith(".py"):
                    full = os.path.join(root, f)
                    r = clf.analyze_file(full)
                    print(f"{os.path.relpath(full, target):<60} → {r.role_score.primary_role.value}")