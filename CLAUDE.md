# CLAUDE.md - Project Guide for AI Assistants

## Project Overview

**RAACS (Role Analysis and Classification System)** - A Python code role classification system that automatically identifies and categorizes the architectural roles of Python modules through multi-layer signal fusion.

This is a research project for software engineering (ASE'26 conference).

## Architecture

### Core Components

```
├── role_classifier.py      # Main entry - Integrated role analyzer with fusion logic
├── ppr.py                  # Personalized PageRank for dependency analysis & visualization
├── main.py                 # CLI entry point (placeholder)
└── raacs/                  # Core library
    ├── ast_analyzer.py     # AST-based role classification & symbol table
    └── graph_analyzer.py   # Dependency graph analysis & architectural layers
```

### Analysis Pipeline

1. **AST Analysis** (`raacs/ast_analyzer.py`)
   - Framework fingerprint detection (Flask, FastAPI, Django, Pydantic, etc.)
   - Decorator/inheritance-based role inference
   - Symbol table construction & role propagation

2. **Graph Analysis** (`raacs/graph_analyzer.py`)
   - Module dependency graph features (in/out degree, centrality)
   - Architectural layer inference (Infrastructure, Domain, Application, Interface)
   - Dynamic threshold system adapting to repository size

3. **Fusion** (`role_classifier.py`)
   - Combines AST roles + Graph roles via fusion rules
   - Auto-generates dependency graph using `pydeps`

## Code Roles (9 Types)

| Role | Description |
|------|-------------|
| `TEST` | Test code |
| `NAMESPACE` | Package `__init__.py` |
| `INTERFACE` | Abstract classes, Protocols |
| `SCHEMA` | Data models (Pydantic, dataclass, ORM) |
| `ADAPTER` | Controllers, views, API handlers |
| `CONFIG` | Configuration modules |
| `SCRIPT` | Entry points, CLI scripts |
| `UTIL` | Utility functions, helpers |
| `LOGIC` | Business logic |

## Key Classes

- `IntegratedRoleAnalyzer` - Main analyzer combining all layers
- `CodeRoleClassifier` - AST-based classifier
- `DependencyGraphAnalyzer` - Graph structure analyzer
- `ProjectSymbolTable` - Cross-file symbol resolution
- `RolePropagator` - Inheritance-based role propagation

## Dependencies

- `pydeps` - For generating dependency graphs
- `networkx` - Graph algorithms (PPR)
- `pyvis` - Interactive visualization

## Usage

```python
from role_classifier import IntegratedRoleAnalyzer

analyzer = IntegratedRoleAnalyzer("/path/to/project", debug=True)
results = analyzer.analyze_project()

for path, result in results.items():
    print(f"{result.module_name}: {result.final_role.value}")
```

## Development Notes

- All thresholds are dynamically computed based on repository statistics
- Role source tracking (`RoleSource`) enables confidence scoring
- Fusion rules are defined as `(AST_Role, Graph_Role) -> (Final_Role, confidence, reasoning)`
