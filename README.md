# WORK IN PROGRESS!!
# MCT: Markov Chain Deriver / Generator / Tester

[![Tests](https://github.com/xoth42/MCT/actions/workflows/tests.yml/badge.svg)](https://github.com/xoth42/MCT/actions/workflows/tests.yml)
[![Lint & Format](https://github.com/xoth42/MCT/actions/workflows/lint.yml/badge.svg)](https://github.com/xoth42/MCT/actions/workflows/lint.yml)


A Python toolkit for **deriving Markov-chain transition matrices from quantum noise models** and **generating fast sampling code** from analytic expressions.

## TL;DR: Quick Commands

```bash
# Setup dev environment
uv sync --dev

# Install package (editable)
uv pip install -e .

# Run tests (unittest or pytest)
uv run python -m unittest tests -v
uv run pytest tests -v

# Run demo notebook (choose kernel from .venv)
uv run python -m jupyter notebook examples/Demo.ipynb
```

## Quick Start

```python
import sympy as sp
import MCT

# Define your Kraus operators with symbols
lam = sp.symbols('lambda', real=True, positive=True)
K0 = sp.Matrix([[1, 0], [0, sp.sqrt(1-lam)]])
K1 = sp.Matrix([[0, sp.sqrt(lam)], [0, 0]])

# Get analytic transition matrix for a basis state
basis = MCT.computational_basis  # [|0⟩, |1⟩]
analytical_result = MCT.superoperator(
    (K0, K1),                 # Kraus operators
    basis[0],                 # |0⟩ state
    basis=basis,
    qubits=1
)

# Generate and optionally save sampling code
MCT.markov_chain(
    analytical_result,
    to_file="generated/t1_sampling.py"
)
```

