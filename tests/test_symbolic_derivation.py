"""Tests for symbolic derivation of Markov chain transition matrices."""

import numpy as np
import pytest
from sympy import Matrix, simplify, sqrt, symbols

from mct.constants import computational_basis
from mct.main import (
    apply_channel,
    derive_transition_matrix,
    substitute_and_verify,
)
from mct.markov_chain import MarkovChain


class TestChannelApplication:
    """Test quantum channel application (Kraus form)."""

    def test_apply_channel_to_ground_state(self):
        """Apply amplitude damping to |0⟩⟨0|."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])

        rho_0 = Matrix([[1, 0], [0, 0]])
        rho_0_out = apply_channel([E0, E1], rho_0)

        rho_0_out_simplified = simplify(rho_0_out)
        assert rho_0_out_simplified == rho_0

    def test_apply_channel_to_excited_state(self):
        """Apply amplitude damping to |1⟩⟨1|."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])

        rho_1 = Matrix([[0, 0], [0, 1]])
        rho_1_out = apply_channel([E0, E1], rho_1)

        for lam_val in [0.0, 0.1, 0.5, 0.9, 1.0]:
            rho_1_numeric = rho_1_out.subs(lam, lam_val)
            rho_1_np = np.array(rho_1_numeric, dtype=float)

            assert rho_1_np[0, 0] == pytest.approx(lam_val)
            assert rho_1_np[1, 1] == pytest.approx(1.0 - lam_val)


class TestGeneralizedDerivation:
    """Test generalized transition matrix derivation."""

    def test_derive_transition_matrix_single_qubit_t1(self):
        """Derive P for single-qubit T1."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])
        kraus_ops = [E0, E1]

        P, metadata = derive_transition_matrix(kraus_ops, computational_basis, channel_name="T1")

        assert hasattr(P, 'shape')
        assert P.shape == (2, 2)
        assert isinstance(metadata, dict)
        assert metadata['channel_name'] == "T1"
        assert metadata['n_states'] == 2

    def test_stochasticity_at_parameter_values(self):
        """Verify P is stochastic at various λ."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])
        kraus_ops = [E0, E1]

        P, _ = derive_transition_matrix(kraus_ops, computational_basis)

        for lambda_val in [0.0, 0.1, 0.5, 0.9, 1.0]:
            P_numeric = P.subs(lam, lambda_val)
            P_np = np.array(P_numeric, dtype=float)

            col_sums = np.sum(P_np, axis=0)
            assert np.allclose(col_sums, 1.0)
            assert np.all((P_np >= -1e-10) & (P_np <= 1.0 + 1e-10))

    def test_markov_chain_creation_from_symbolic(self):
        """Verify symbolic matrix can create MarkovChain."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])
        kraus_ops = [E0, E1]

        P, _ = derive_transition_matrix(kraus_ops, computational_basis)

        P_numeric = P.subs(lam, 0.3)
        P_np = np.array(P_numeric, dtype=float)

        mc = MarkovChain(P_np, state_labels=['|0⟩', '|1⟩'], name='T1_test')
        result = mc.validate_stochasticity()
        assert result['is_valid']


class TestSubstituteAndVerify:
    """Test utility function for numeric substitution."""

    def test_verify_at_lambda_half(self):
        """Test verification at λ=0.5."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])

        P, _ = derive_transition_matrix([E0, E1], computational_basis)
        P_np, is_valid = substitute_and_verify(P, lam, 0.5)

        assert is_valid
        assert P_np.shape == (2, 2)
        assert np.allclose(np.sum(P_np, axis=0), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
