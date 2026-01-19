# RAACS: Role-Adaptive Architecture Classification System

## Project Overview

RAACS (Role-Adaptive Architecture Classification System) is a three-layer code role classification tool that analyzes Python codebases to identify architectural roles of files and entities. It combines AST analysis, symbol table propagation, and dependency graph analysis to classify code into architectural roles with confidence scores.

**Current Version**: v9.2
**Status**: Active development
**Language**: Python 3.8+

## What This Project Does

RAACS analyzes Python projects to automatically classify each file and entity into architectural roles:

- **TEST**: Unit tests, integration tests, mock data
- **NAMESPACE**: Package init files (imports only)
- **INTERFACE**: Abstract base classes, protocols
- **SCHEMA**: Data models, DTOs, ORM entities
- **ADAPTER**: API routes, views, controllers, I/O boundaries
- **CONFIG**: Environment variables, constants, settings
- **SCRIPT**: CLI tools, entry points, standalone scripts
- **UTIL**: Stateless helper functions, utilities
- **LOGIC**: Core business logic (default/fallback role)

### Three-Layer Analysis Architecture

```
Layer 1: AST Layer
├─ What: Internal file structure analysis
├─ How: Framework fingerprints + structural patterns + path hints
└─ Answers: "What does this file DO?"

Layer 2: Symbol Table Layer
├─ What: Cross-file inheritance relationships
├─ How: Role propagation through class hierarchies
└─ Answers: "What is this file a subclass OF?"

Layer 3: Graph Layer (NEW in v9)
├─ What: Dependency network analysis
├─ How: pydeps-based graph topology + dynamic thresholds
├─ GraphRoles: HUB, ORCHESTRATOR, BRIDGE, LEAF, SINK, ISOLATE
└─ Answers: "WHERE is this file in the architecture?"

Fusion: Weighted Role Fusion
├─ What: Combines all three layers with fusion rules
└─ Output: Final architectural role + confidence + reasoning
```

## Core Components

### Main Entry Point
- `role_classifier_v9.py` - Primary CLI and integration analyzer

### RAACS Library (`raacs/`)
- `ast_analyzer.py` (v8.1) - AST analysis + symbol table + role propagation
- `graph_analyzer.py` - Dependency graph analysis with dynamic thresholds
- `__init__.py` - Public API exports

### Key Classes

**AST Layer:**
- `CodeRoleClassifier` - Main AST analyzer
- `SymbolCollector` - Builds project-wide symbol table
- `RolePropagator` - Propagates roles through inheritance
- `ProjectSymbolTable` - Global class/function registry
- `RoleSource` (v9.2) - Tracks where role assignments come from

**Graph Layer:**
- `DependencyGraphAnalyzer` - Graph topology analyzer with dynamic thresholds
- `DependencyGraphGenerator` - pydeps wrapper for auto-generation
- `DynamicThresholds` - Repository-scale-aware threshold computation
- `RepositoryStats` - In/out-degree distribution statistics

**Fusion:**
- `IntegratedRoleAnalyzer` - Three-layer fusion orchestrator
- `IntegratedRoleResult` - Final analysis result with all metadata

### Enums
- `Role` - 9 architectural roles
- `GraphRole` - 6 graph topology roles (HUB, ORCHESTRATOR, BRIDGE, LEAF, SINK, ISOLATE)
- `ArchitecturalLayer` - 3 layers (APPLICATION, INTERFACE, INFRASTRUCTURE)

## How It Works

### Recognition Strategy (Weighted Scoring)

1. **Priority 1: Framework Fingerprints (Weight: 4.0)**
   - Explicit imports: `pytest`, `unittest`, `flask`, `fastapi`, `pydantic`, `sqlalchemy`
   - Decorators: `@fixture`, `@route`, `@dataclass`, `@api_view`
   - Base classes: `TestCase`, `BaseModel`, `ABC`, `Protocol`

2. **Priority 2: Structural Patterns (Weight: 2.5)**
   - Assert density (TEST)
   - Field-to-method ratio (SCHEMA)
   - Abstraction rate (INTERFACE)
   - I/O parameter patterns (ADAPTER)
   - Complexity metrics (LOGIC vs UTIL)

3. **Priority 3: Path Hints (Weight: 1.5)**
   - Directory names: `tests/`, `utils/`, `models/`, `api/`
   - File naming conventions: `test_*.py`, `config.py`, `__init__.py`

4. **Fallback: LOGIC**
   - Default for files with complex logic that don't match other patterns

### Dynamic Thresholds (v9.1)

RAACS adapts to repository size using statistical distribution:

| Repo Size | Modules | HUB Percentile | ORCH Percentile | APP Layer Ratio |
|-----------|---------|----------------|-----------------|-----------------|
| tiny      | <30     | P80            | P80             | 0.50            |
| small     | 30-100  | P85            | P85             | 0.55            |
| medium    | 100-300 | P90            | P90             | 0.60            |
| large     | 300-1000| P92            | P92             | 0.65            |
| huge      | >1000   | P95            | P95             | 0.70            |

### Fusion Rules

Key examples of AST + Graph → Final role:
- (LOGIC, HUB) → UTIL (high centrality suggests core utility)
- (LOGIC, ORCHESTRATOR) → LOGIC (confirms business logic)
- (LOGIC, SINK) → SCRIPT (entry point)
- (LOGIC, LEAF) → UTIL (stateless utility)
- (LOGIC, BRIDGE) → ADAPTER (adapter layer)

## Usage

### Command Line

```bash
# Basic analysis (auto-generates dependency graph)
python role_classifier_v9.py /path/to/project

# Analyze single file
python role_classifier_v9.py /path/to/project --file path/to/file.py

# Debug mode with full details
python role_classifier_v9.py /path/to/project --debug --show-fusion --show-graph

# Save dependency graph for reuse
python role_classifier_v9.py /path/to/project --save-deps deps.json

# Use pre-generated dependency graph
python role_classifier_v9.py /path/to/project --dep-map deps.json

# Disable auto-generation
python role_classifier_v9.py /path/to/project --no-auto-deps

# Version info
python role_classifier_v9.py --version
```

### As a Library

```python
from raacs import (
    CodeRoleClassifier,
    DependencyGraphAnalyzer,
    Role,
    GraphRole,
    RoleSource
)

# Integrated analysis
from role_classifier_v9 import IntegratedRoleAnalyzer

analyzer = IntegratedRoleAnalyzer(
    "/path/to/project",
    auto_generate_deps=True,
    debug=True
)

# Analyze entire project
results = analyzer.analyze_project()

# Analyze single file
result = analyzer.analyze_file("/path/to/file.py")

# Access results
print(f"AST Role: {result.ast_role.value}")
print(f"Graph Role: {result.graph_role.value}")
print(f"Final Role: {result.final_role.value} (conf={result.final_confidence:.2f})")
print(f"Reasoning: {result.fusion_reasoning}")
```

## Directory Structure

```
role_classifier_CLAUDE/
├── role_classifier_v9.py       # Main entry point
├── ppr.py                       # PageRank implementation (research)
├── raacs/                       # Core library
│   ├── __init__.py             # Public API
│   ├── ast_analyzer.py         # AST + symbol table (v8.1)
│   └── graph_analyzer.py       # Graph analysis + dynamic thresholds
├── docs/                        # Changelogs and documentation
│   ├── role_classifier_v9_changelog.md
│   └── ast_analyzer_v8.1_changelog.md
├── reference_docs/              # Design documents (Chinese)
│   ├── 代码角色识别规则.md
│   ├── 解释角色化PageRank.md
│   ├── 目标Benchmark与Baseline.md
│   └── ...
├── deprecated/                  # Older versions
│   ├── role_classifier_v7.py
│   └── role_classifier_v8.py
└── repos_to_be_examined/        # Test repositories
    ├── auto-nag/
    └── lithium/
```

## Dependencies

- Python 3.8+
- `pydeps` (optional, for auto-generating dependency graphs)

```bash
pip install pydeps
```

## Key Features Added in Recent Versions

### v9.2 (Latest)
- `RoleSource` enum for tracking role assignment origins
- Separated FQN vs Simple Name indexing in symbol table
- Weak signal override logic
- Structured `BaseInfo` for inheritance tracking

### v9.1
- Dynamic threshold system based on repository statistics
- Adaptive percentile selection by repo size
- Removed hardcoded magic numbers (0.3, 5, 10, etc.)

### v9.0
- Three-layer fusion architecture
- Automatic dependency graph generation via pydeps
- Graph topology roles (HUB, ORCHESTRATOR, BRIDGE, etc.)
- Architectural layer inference
- Fusion rules table

### v8.0
- Two-phase analysis (collection + propagation)
- Cross-file symbol table
- Role inheritance through class hierarchies

## Important Concepts

### Role Purity
A metric (0.0-1.0) indicating how "pure" a file is in its primary role:
- 1.0 = All entities have the same role
- <1.0 = Mixed roles within the file

### Confidence Scores
- AST confidence: Based on signal strength (framework/structure/path)
- Graph confidence: Based on topological features
- Final confidence: Fusion-adjusted confidence

### Reasoning Chains
Every role assignment includes human-readable reasoning:
- AST reasoning: Why the AST layer chose this role
- Graph reasoning: Why the graph layer chose this role
- Fusion reasoning: Why the final role differs (or doesn't) from AST

## Development Guidelines

### When Adding New Roles
1. Add to `Role` enum in `raacs/ast_analyzer.py`
2. Define framework fingerprints (strong signals)
3. Define structural patterns (semantic signals)
4. Define path hints (weak signals)
5. Update fusion rules in `role_classifier_v9.py`
6. Update documentation

### When Modifying Thresholds
- Use `DynamicThresholds` system - DO NOT hardcode
- Base on statistical distribution (P50, P75, P90, etc.)
- Consider repository scale impact
- Test on tiny/small/medium/large repos

### Code Style
- Type hints everywhere
- Dataclasses for structured data
- Enums for categorical values
- Clear reasoning strings for explainability
- Debug mode support in all analyzers

## Research Context

This project is part of research on:
- Automated code understanding
- Architecture recovery from source code
- Context compression for LLM-based code analysis
- Role-based PageRank (PPR) for minimal context construction

See `reference_docs/` for research notes (in Chinese).

## Testing Repositories

Sample projects in `repos_to_be_examined/`:
- `auto-nag/` - Bugzilla automation tool (complex business logic)
- `lithium/` - Fuzzing test case reducer (scripting/testing focus)

## Known Limitations

1. **Python-only**: Currently only analyzes Python codebases
2. **pydeps dependency**: Graph layer requires pydeps (installable via pip)
3. **Import-based**: Relies on import statements for dependency graph
4. **Static analysis**: Cannot detect runtime behaviors
5. **Framework coverage**: Best results with mainstream frameworks (Django, Flask, FastAPI, Pydantic, SQLAlchemy)

## Color Coding (CLI Output)

- TEST: Gray (90)
- NAMESPACE: Cyan (36)
- INTERFACE: Magenta (35)
- SCHEMA: Blue (34)
- ADAPTER: Yellow (33)
- CONFIG: White (37)
- SCRIPT: Red (31)
- UTIL: Light Cyan (96)
- LOGIC: Green (32)
- UNKNOWN: Default (0)

## When to Use This Tool

**Good for:**
- Understanding unfamiliar codebases quickly
- Validating architectural assumptions
- Identifying misplaced files (e.g., logic in adapter layer)
- Code review assistance
- Documentation generation
- Refactoring planning

**Not suitable for:**
- Real-time analysis (AST parsing is slow)
- Non-Python projects
- Codebases with heavy metaprogramming
- Projects without clear architectural patterns

## Future Work (from reference docs)

- Role-based PageRank (PPR) integration
- LLM-based context compression
- Benchmark against SWE-bench
- Multi-language support
- Real-time analysis optimization

---

**Note for AI Assistants**: This project emphasizes explainability - every classification includes reasoning chains. When modifying code, preserve this transparency. All thresholds should be dynamic, not hardcoded. Chinese documentation in `reference_docs/` contains important design rationale.
