# Contributing

Thank you for your interest in **TopoCircuit-SAE**.  
This project is a capstone + mechanistic interpretability research submission. Contributions are welcome.

---

## Ground Rules

- **One author per commit.** Commits must be authored by Farhan Aaqil Durrani.  
  Do not add co-authors or modify `git config user.*` settings.
- **Single branch model.** All work targets the `main` branch via pull requests from `master`.
- **CI must be green before merging.** The `CI` workflow (lint + tests) must pass on every PR.

---

## Development Setup

```bash
# Clone
git clone https://github.com/MuzraFatima/NeuroLens.git
cd NeuroLens

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install with dev extras
pip install -e ".[dev]"
```

---

## Running the Tests

```bash
# Core research tests (Lemma 1, routing, MPAbs, DR-SAE)
python -m pytest tests/test_topocircuit.py -v

# Production engineering tests (SQLite pool, ETag, gzip, SAELens)
python -m pytest tests/test_production_readiness.py -v

# All tests
python -m pytest -v
```

---

## Lint

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for lint issues
ruff check .

# Auto-fix safe issues
ruff check --fix .

# Format check
ruff format --check .

# Auto-format
ruff format .
```

Lint and format settings are defined in `pyproject.toml` under `[tool.ruff]`.

---

## Reproducing Results

```bash
python scripts/run_all_experiments.py
```

This single command reproduces all Phase 1–7 experimental results, statistical tests,
and the 5 publication figures in `results/figures/`.

---

## Pull Request Flow

1. Branch from `main` (or from `master` if doing phased delivery).
2. Make your changes; commit after each logical unit of work.
3. Open a PR targeting `main`.
4. Ensure CI is green — the `CI` check (lint + tests) must pass.
5. Merge only after the CI badge shows ✅.
