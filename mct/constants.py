"""Predefined quantum channel constants and basis definitions."""

from sympy import Matrix, sqrt, symbols

# ============================================================================
# Channel Definitions (Kraus Operators)
# ============================================================================

# Parameterized Kraus operators (use with symbols)
lam = symbols('lambda', real=True, positive=True)
AMPLITUDE_DAMPING_KRAUS = (
    Matrix([[1, 0], [0, sqrt(1 - lam)]]),
    Matrix([[0, sqrt(lam)], [0, 0]])
)

# Two-qubit ADC parameters
lam1, lam2 = symbols('lambda_1 lambda_2', real=True, positive=True)
TWO_QUBIT_ADC_KRAUS_PARAMS = (lam1, lam2)


# ============================================================================
# Basis State Definitions
# ============================================================================

# Single qubit: {|0⟩, |1⟩}
computational_basis = [
    Matrix([[1, 0], [0, 0]]),
    Matrix([[0, 0], [0, 1]]),
]

# Two qubit Bell basis
bell_basis = [
    (1/2) * Matrix([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
    (1/2) * Matrix([[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]),
    (1/2) * Matrix([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]),
    (1/2) * Matrix([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]]),
]


def get_basis(name: str, qubits: int = None):
    """Get predefined basis by name ('computational', 'bell')."""
    name_lower = name.lower()

    if name_lower in ['computational', 'comp']:
        return computational_basis
    elif name_lower in ['bell']:
        return bell_basis
    else:
        raise ValueError(
            f"Unknown basis '{name}'. "
            f"Available: 'computational', 'bell'"
        )
