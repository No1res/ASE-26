# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RAACS** (Role-Aware Automated Code System) is a Python-based static analysis tool that classifies code files into semantic roles. It analyzes Python repositories to determine the architectural purpose of each file (e.g., TEST, SCHEMA, ADAPTER, LOGIC) using a multi-signal fusion approach combining AST analysis, dependency graph analysis, and role propagation.

### Core Concepts

- **Roles**: 9 semantic categories (TEST, NAMESPACE, INTERFACE, SCHEMA, ADAPTER, CONFIG, SCRIPT, UTIL, LOGIC, UNKNOWN) that correspond to different "Morphing" strategies
- **Three-layer Signal Fusion**: Combines AST-based classification, graph structure analysis, and role propagation to determine final roles
- **Role Source Tracking**: Tracks how each role was determined (framework detection, decorator, name patterns, inheritance, propagation, etc.)

## Project Structure

```
raacs/                    # Main package
├── __init__.py          # Package exports (Role, CodeRoleClassifier, etc.)
├── ast_analyzer.py      # AST analysis implementation (71k lines)
├── graph_analyzer.py    # Graph analysis implementation (32k lines)
├── core/                # Core analysis components
│   ├── roles.py         # Role enum, source tracking, compatibility matrix
│   ├── ast.py           # Re-exports from ast_analyzer.py
│   ├── graph.py         # Re-exports from graph_analyzer.py
│   ├── fusion.py        # Multi-signal fusion logic (IntegratedRoleAnalyzer)
│   └── diffusion.py     # PPR implementation (CodeGraphBuilder, GraphVisualizer)
├── adapters/            # External tool integrations
│   ├── pydeps.py        # pydeps dependency extraction
│   └── viz.py           # Visualization adapter
├── pipeline/            # Analysis orchestration
│   └── analyze.py       # Main analysis pipeline
└── utils/               # Utility functions

cli/                     # Command-line interface
├── raacs_analyze.py     # Main analysis CLI
└── raacs_context.py     # Context generation CLI

main.py                  # Entry point
```

## Common Commands

### Running Analysis

```bash
# Analyze a repository
python -m cli.raacs_analyze /path/to/repo --out results.json

# With debug output
python -m cli.raacs_analyze /path/to/repo --out results.json --debug
```

### Development

```bash
# Create/activate virtual environment (using uv)
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Run the main entry point
python main.py
```

## Architecture Notes

### Analysis Pipeline Flow

1. **Dependency Extraction**: Uses `pydeps` to build module dependency graph
2. **AST Analysis**: First pass - collects symbols, analyzes code structure
3. **Role Propagation**: Second pass - propagates roles through inheritance/imports
4. **Graph Analysis**: Analyzes structural position in dependency graph
5. **Signal Fusion**: Combines all signals with weighted confidence scores

### Key Classes

- `Role`: Enum defining the 9 code roles
- `RoleSource`: Tracks origin of role classification
- `CodeRoleClassifier`: AST-based file analysis (in `raacs/ast_analyzer.py`)
- `DependencyGraphAnalyzer`: Graph structure analysis (in `raacs/graph_analyzer.py`)
- `IntegratedRoleAnalyzer`: Final fusion of all signals (in `raacs/core/fusion.py`)
- `RolePropagator`: Propagates roles through symbol relationships
- `CodeGraphBuilder`: PPR graph builder (in `raacs/core/diffusion.py`, requires networkx)
- `GraphVisualizer`: Interactive visualization (in `raacs/core/diffusion.py`, requires pyvis)

### Import Patterns

```python
# Core imports (no external dependencies beyond stdlib)
from raacs import Role, RoleSource, CodeRoleClassifier, DependencyGraphAnalyzer
from raacs import IntegratedRoleAnalyzer, IntegratedRoleResult

# PPR/Visualization (requires networkx, pyvis, matplotlib)
from raacs.core.diffusion import run_ppr, CodeGraphBuilder, GraphVisualizer
```

### Signal Weights (from `SignalWeight`)

- FRAMEWORK: 4.0 (highest - framework detection like pytest, pydantic)
- INHERITANCE: 3.5 (base class relationships)
- STRUCTURE: 2.5 (code structure patterns)
- PATH_HINT: 1.5 (directory/file naming)
- NAME_HINT: 1.0 (identifier naming patterns)

## Code Style

- Python 3.12+
- Type hints throughout
- Docstrings in Chinese (项目文档使用中文)
- Uses dataclasses and enums extensively
- Follows standard Python project layout

## Dependencies

- Core: Python 3.12+
- External: pydeps (for dependency graph extraction)
- Optional: networkx, pyvis, matplotlib (for PPR and visualization)
- Package manager: uv
