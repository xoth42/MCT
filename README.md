# MCT: Markov Chain Deriver / Generator / Tester

[![Tests](https://github.com/xoth42/MCT/actions/workflows/tests.yml/badge.svg)](https://github.com/xoth42/MCT/actions/workflows/tests.yml)
[![Lint & Format](https://github.com/xoth42/MCT/actions/workflows/lint.yml/badge.svg)](https://github.com/xoth42/MCT/actions/workflows/lint.yml)

A Python toolkit for **deriving Markov-chain transition matrices from quantum noise models** and **generating fast sampling code** from analytic expressions.

## Vision

MCT enables quantum researchers to:

1. **Define quantum channels** using Kraus operators with SymPy symbols
2. **Derive analytic transition probabilities** symbolically from any quantum channel
3. **Generate fast sampling code** using if/elif/else patterns and cumulative probabilities
4. **Test noise models** with comprehensive validation suites

All while preserving **analytic structure**—you get symbolic expressions that work for any parameter values, not just numeric approximations.

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

## Core Concepts

### Markov Chain Representation
- **Column-stochastic matrix**: P[i, j] = Prob(state i | from state j)
- **Analytic expressions**: Parametrized by noise parameters (λ, λ₁, λ₂, etc.)
- **Verified**: Stochasticity checks at all parameter values

### Workflow

```
Quantum Channel (Kraus form)
        ↓
  [Symbolic Derivation]
        ↓
Transition Matrix P(λ, λ₁, ...)
        ↓
  [Code Generation]
        ↓
Sampling Function (if/elif/else)
        ↓
  [Testing & Validation]
        ↓
Production-Ready Code
```

## API Design

### `MCT.superoperator(kraus_ops, rho, basis, qubits)`
Derive the transition matrix for a quantum channel applied to a density matrix.

**Parameters:**
- `kraus_ops` (tuple): Sympy matrices representing Kraus operators
- `rho` (sympy.Matrix): Input density matrix (basis state)
- `basis` (list): Basis states (e.g., `MCT.computational_basis`, `MCT.bell_basis`)
- `qubits` (int): Number of qubits (1, 2, etc.)

**Returns:**
- `P` (sympy.Matrix): Symbolic column-stochastic transition matrix
- `metadata` (dict): Symbols, edge cases, Kraus operators, etc.

### `MCT.markov_chain(analytical_result, to_file=None)`
Generate sampling code from an analytical transition matrix.

**Parameters:**
- `analytical_result` (tuple): (P_symbolic, metadata) from `superoperator()`
- `to_file` (str, optional): Path to write generated Python code

**Returns:**
- `code` (str): Python sampling function as string (and writes to file if specified)

### Predefined Bases

```python
MCT.computational_basis   # [|0⟩, |1⟩]
MCT.bell_basis            # [|Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩]
MCT.get_basis(name, qubits)  # Generic accessor
```

## Examples

### Single-Qubit T1 (Amplitude Damping)

```python
import sympy as sp
import MCT

lam = sp.symbols('lambda', real=True, positive=True)
K0 = sp.Matrix([[1, 0], [0, sp.sqrt(1-lam)]])
K1 = sp.Matrix([[0, sp.sqrt(lam)], [0, 0]])

# Derive for ground state |0⟩
P_0, meta = MCT.superoperator(
    (K0, K1),
    MCT.computational_basis[0],
    basis=MCT.computational_basis,
    qubits=1
)

# Generate sampling code
code = MCT.markov_chain((P_0, meta), to_file="t1_ground.py")
print(code)  # Prints the generated function
```

### Two-Qubit Bell Basis with Asymmetric T1

```python
import sympy as sp
import MCT

lam1, lam2 = sp.symbols('lambda1 lambda2', real=True, positive=True)

# Tensor product of single-qubit T1 channels
K0 = sp.eye(4) - sp.diag(lam1, lam1, lam2, lam2)  # Simplified
K1 = sp.diag(sp.sqrt(lam1), 0, sp.sqrt(lam2), 0)

# Derive for Bell state |Φ+⟩
P_bell, meta = MCT.superoperator(
    (K0, K1),
    MCT.bell_basis[0],
    basis=MCT.bell_basis,
    qubits=2
)

# Generate and save
MCT.markov_chain((P_bell, meta), to_file="bell_t1_asymmetric.py")
```

## Architecture

```
mct/
├── __init__.py              # Public API exports
├── markov_chain.py          # Core MarkovChain class & validation
├── symbolic_derivation.py   # Symbolic channel derivation
├── sampling.py              # Sampling code generation
├── noise_models.py          # T1/T2 numeric utilities
└── bases.py                 # Basis state definitions
```

## Development

This project uses `uv` for reproducible Python environment management.

```bash
# Setup
uv sync --dev

# Run tests
uv run pytest tests -v

# Develop
source .venv/bin/activate
# (or) uv run python script.py
```

## Development

### Running Tests Locally

```bash
# Install dependencies (requires uv)
uv sync --dev

# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=mct --cov-report=html

# Run specific test file
uv run pytest tests/test_symbolic_derivation.py -v
```

### Code Quality

```bash
# Format code
uv run ruff format mct/ tests/

# Check linting
uv run ruff check mct/ tests/
```

### Continuous Integration

This project uses GitHub Actions for:
- **Tests**: Runs pytest on Python 3.9-3.12, macOS and Linux
- **Lint**: Checks code style with ruff on each PR
- **Coverage**: Generates coverage reports (uploaded to Codecov on main branch)

See `.github/workflows/` for configuration.

## References

Core patterns and physics models are derived from:

- **markov_chain_generator_full_code_jan18.pdf**: Code generation patterns
- **twiling_T1_Jan18.pdf**: Bell-basis symbolic derivations
- **T1 T2 noise.pdf** & **CS648QML_error_kraus.py.pdf**: Kraus operator definitions
- **qubit_guide.pdf** (Ch. 9): Quantum channels and operator-sum form

See `references/` folder and `.github/copilot-instructions.md` for detailed guidance.

## Design Principles

1. **Symbolic First**: Always preserve analytic structure; numeric values only at the end
2. **Separation of Concerns**: Physics, derivation, code generation, testing are distinct
3. **Reuse Patterns**: Build on existing derivations rather than reimplementing
4. **Tested & Verified**: Every function has corresponding tests; stochasticity is non-negotiable
5. **User-Focused**: Simple, intuitive API that hides internal complexity

## License

See LICENSE file.
