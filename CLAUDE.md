# RAACS: Role-Adaptive Architecture Classification System

## Project Overview

RAACS (Role-Adaptive Architecture Classification System) is a three-layer code role classification tool that analyzes Python codebases to identify architectural roles of files and entities. It combines AST analysis, symbol table propagation, and dependency graph analysis to classify code into architectural roles with confidence scores.

**Current Version**: v9.3
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

Layer 3: Graph Layer
├─ What: Dependency network analysis
├─ How: Static import scanning + graph topology + dynamic thresholds
├─ GraphRoles: MEGA_HUB, HUB, ORCHESTRATOR, BRIDGE, LEAF, SINK, ISOLATE
└─ Answers: "WHERE is this file in the architecture?"

Layer 4: Spectral Layer (NEW in v9.3)
├─ What: Laplacian Eigenvector analysis
├─ How: Graph spectral decomposition + PPR integration
└─ Answers: "Which modules belong to the same functional cluster?"

Fusion: Weighted Role Fusion
├─ What: Combines all layers with fusion rules
└─ Output: Final architectural role + confidence + reasoning
```

## Core Components

### Main Entry Point
- `role_classifier_v9.py` - Primary CLI and integration analyzer

### RAACS Library (`raacs/`)
- `ast_analyzer.py` (v8.1) - AST analysis + symbol table + role propagation
- `graph_analyzer.py` - Dependency graph analysis with dynamic thresholds
- `spectral_ppr.py` (v9.3) - Laplacian Eigenvector + PageRank integration
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
- `DynamicThresholds` - Repository-scale-aware threshold computation
- `RepositoryStats` - In/out-degree distribution statistics

**Spectral Layer (v9.3):**
- `SpectralAnalyzer` - Laplacian eigenvector computation and clustering
- `SpectralPPR` - Spectral-weighted Personalized PageRank
- `SpectralEmbedding` - Node embeddings from Fiedler vector

**Fusion:**
- `IntegratedRoleAnalyzer` - Three-layer fusion orchestrator
- `IntegratedRoleResult` - Final analysis result with all metadata

### Enums
- `Role` - 9 architectural roles
- `GraphRole` - 7 graph topology roles (MEGA_HUB, HUB, ORCHESTRATOR, BRIDGE, LEAF, SINK, ISOLATE)
- `ArchitecturalLayer` - 3 layers (APPLICATION, INTERFACE, INFRASTRUCTURE)

## Usage

### Command Line (role_classifier_v9.py)

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

# Version info
python role_classifier_v9.py --version
```

### Spectral PPR (NEW in v9.3)

```bash
# Install dependencies
pip install numpy scipy networkx
```

```python
from raacs.spectral_ppr import SpectralAnalyzer, SpectralPPR, create_spectral_ppr

# 1. Prepare dependency map (from static import scanner or other source)
dep_map = {
    'core.models': {'imports': ['core.base', 'utils.helpers'], 'path': '/path/to/core/models.py'},
    'core.base': {'imports': [], 'path': '/path/to/core/base.py'},
    'core.services': {'imports': ['core.models', 'core.base'], 'path': '/path/to/core/services.py'},
    'utils.helpers': {'imports': [], 'path': '/path/to/utils/helpers.py'},
    'api.views': {'imports': ['core.services', 'core.models'], 'path': '/path/to/api/views.py'},
    'api.routes': {'imports': ['api.views'], 'path': '/path/to/api/routes.py'},
}

# 2. Spectral Analysis (Laplacian Eigenvector)
analyzer = SpectralAnalyzer(dep_map)
embedding = analyzer.compute_embedding()

# Get Fiedler vector (for module partitioning)
print(f"Fiedler values: {embedding.fiedler_vector}")
# Positive/negative values indicate different functional clusters

# Get spectral clustering
clusters = analyzer.get_clusters(n_clusters=3)
print(f"Clusters: {clusters}")
# Output: {'core.models': 0, 'core.base': 0, 'api.views': 1, ...}

# Get spectral centrality ranking
centrality = analyzer.rank_by_centrality()
print(f"Most central modules: {centrality[:5]}")

# 3. Spectral-Weighted PPR (Context Window)
ppr = SpectralPPR(dep_map, sigma=1.0)

# Get context window for a target module
target = 'api.views'
context = ppr.run_ppr(target, top_k=10, alpha=0.85)
print(f"Context window for {target}:")
for module, score in context:
    print(f"  {module}: {score:.4f}")

# With cluster boost (same-cluster modules get higher scores)
context_boosted = ppr.run_ppr_with_cluster_boost(target, top_k=10, cluster_boost=1.5)

# 4. Filter important nodes (for visualization)
important_nodes = ppr.get_filtered_nodes(max_nodes=50)
print(f"Important nodes: {important_nodes}")
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
│   ├── graph_analyzer.py       # Graph analysis + dynamic thresholds
│   └── spectral_ppr.py         # Laplacian + PPR (v9.3)
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

### Core (no external dependencies)
- Python 3.8+

### Optional (for specific features)

```bash
# For Spectral PPR (v9.3)
pip install numpy scipy networkx

# For visualization
pip install pyvis matplotlib
```

## Key Features Added in Recent Versions

### v9.3 (Latest)
- **Spectral PPR**: Laplacian Eigenvector + PageRank integration
- **SpectralAnalyzer**: Graph spectral decomposition for clustering
- **Fiedler vector**: Module partitioning based on graph cuts
- **Spectral-weighted edges**: PPR prefers same-cluster propagation
- **Node filtering**: `get_filtered_nodes()` for visualization
- **Tiered HUB**: MEGA_HUB (P95) + HUB (P80) for better granularity
- **Zero-inflation handling**: Uses non-zero percentiles when >50% nodes have degree=0
- **Graph density correction**: Sparse graphs get lower thresholds (factor 0.7-0.85)

### v9.2
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
- Automatic dependency graph generation
- Graph topology roles (HUB, ORCHESTRATOR, BRIDGE, etc.)
- Architectural layer inference
- Fusion rules table

## How Spectral PPR Works

### Laplacian Eigenvector

The Laplacian matrix `L = D - A` captures graph structure:
- **Fiedler vector** (2nd eigenvector): Optimal graph bipartition
- **Positive values**: One functional cluster
- **Negative values**: Another functional cluster
- **Near zero**: Bridge modules (adapters/interfaces)

### Spectral-Weighted PPR

```
edge_weight(A → B) = ast_weight × (1 + spectral_similarity(A, B))

spectral_similarity(A, B) = exp(-||v(A) - v(B)||² / (2σ²))
```

**Effect**: PPR propagates faster within the same functional cluster, producing more focused context windows.

| Edge Type | Traditional Weight | Spectral-Weighted |
|-----------|-------------------|-------------------|
| Same-cluster import | 1.0 | ~1.5 ↑ |
| Cross-cluster import | 1.0 | ~1.1 |
| Same-cluster inheritance | 3.0 | ~4.5 ↑ |
| Cross-cluster inheritance | 3.0 | ~3.3 |

## Recognition Strategy (Weighted Scoring)

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

## Dynamic Thresholds (v9.3 Enhanced)

RAACS adapts to repository characteristics using three mechanisms:

### 1. Tiered HUB Classification
| Role | Percentile | Description |
|------|------------|-------------|
| MEGA_HUB | P95 | Super core modules (極少數) |
| HUB | P80 | Core modules (被廣泛依賴) |

### 2. Repository Size Adaptation
| Repo Size | Modules | APP Layer Ratio |
|-----------|---------|-----------------|
| tiny      | <30     | 0.50            |
| small     | 30-100  | 0.55            |
| medium    | 100-300 | 0.60            |
| large     | 300-1000| 0.65            |
| huge      | >1000   | 0.70            |

### 3. Graph Density Correction
| Density Category | Range | Factor | Effect |
|------------------|-------|--------|--------|
| very_sparse | <1% | 0.70 | Thresholds reduced 30% |
| sparse | 1-5% | 0.85 | Thresholds reduced 15% |
| moderate | 5-15% | 1.00 | No change |
| dense | >15% | 1.10 | Thresholds increased 10% |

### 4. Zero-Inflation Handling
When >50% of nodes have in-degree = 0:
- Uses non-zero percentiles (P75 of non-zero values)
- Prevents threshold collapse to 0

**Example output**:
```
[DynamicThresholds] Repository stats:
  Modules: 115 (medium)
  Density: 0.0187 (sparse), edges=245
  Zero-inflation: 52.2% (in-degree zeros)
  In-degree:  min=0, max=23, P80=2.0, P95=7.0
    (nonzero: n=55, mean=3.7, P75=5.0)

DynamicThresholds( [density_factor=0.85]
  MEGA_HUB: in_degree >= 10.4 (P95.0)
  HUB: in_degree >= 7.0 (P80.0)
  ORCHESTRATOR: out_degree >= 5.8 (P80.0)
)
```

## Fusion Rules

Key examples of AST + Graph → Final role:
- (LOGIC, HUB) → UTIL (high centrality suggests core utility)
- (LOGIC, ORCHESTRATOR) → LOGIC (confirms business logic)
- (LOGIC, SINK) → SCRIPT (entry point)
- (LOGIC, LEAF) → UTIL (stateless utility)
- (LOGIC, BRIDGE) → ADAPTER (adapter layer)

## Known Limitations

1. **Python-only**: Currently only analyzes Python codebases
2. **Static analysis**: Cannot detect runtime behaviors
3. **Import-based**: Relies on import statements for dependency graph
4. **Framework coverage**: Best results with mainstream frameworks (Django, Flask, FastAPI, Pydantic, SQLAlchemy)

## Research Context

This project is part of research on:
- Automated code understanding
- Architecture recovery from source code
- Context compression for LLM-based code analysis
- Role-based PageRank (PPR) for minimal context construction
- Spectral graph theory for code clustering

See `reference_docs/` for research notes (in Chinese).

---

**Note for AI Assistants**: This project emphasizes explainability - every classification includes reasoning chains. When modifying code, preserve this transparency. All thresholds should be dynamic, not hardcoded. Chinese documentation in `reference_docs/` contains important design rationale.
