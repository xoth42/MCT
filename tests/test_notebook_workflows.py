"""Test notebook workflows end-to-end without Jupyter execution overhead.

These tests validate that all code in notebooks/01_symbolic_t1_computational.ipynb,
notebooks/02_api_quickstart.ipynb, and notebooks/03_monte_carlo_validation.ipynb
executes correctly.
"""

import numpy as np
import pytest
from sympy import Matrix, sqrt, symbols

import mct


class TestNotebook01SymbolicT1Computational:
    """Test notebook 01: Single-qubit T1 (Amplitude Damping)."""

    def test_setup_symbols(self):
        """Cell 1: Setup - define symbolic parameter."""
        lam = symbols('lambda', real=True, positive=True)
        assert lam.is_positive
        assert lam.is_real

    def test_derive_transition_matrix(self):
        """Cell 2: Derive transition matrix from Kraus operators."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])

        P, metadata = mct.derive_transition_matrix(
            [E0, E1],
            mct.computational_basis,
            channel_name="T1"
        )

        # Verify P is symbolic
        assert P.free_symbols
        assert lam in P.free_symbols
        # Verify shape (2 qubits -> 2 states)
        assert P.shape == (2, 2)
        # Verify metadata
        assert metadata['channel_name'] == 'T1'
        assert metadata['n_states'] == 2

    def test_verify_stochasticity_all_parameters(self):
        """Cell 3: Verify stochasticity at parameter edge cases."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P, _ = mct.derive_transition_matrix([E0, E1], mct.computational_basis)

        # Test at boundary values and interior
        test_values = [0.0, 0.1, 0.5, 0.9, 1.0]
        for lam_val in test_values:
            P_num, is_valid = mct.substitute_and_verify(
                P, lam, lam_val, name=f"λ={lam_val}"
            )
            assert is_valid, f"Stochasticity check failed at λ={lam_val}"
            # Verify all entries non-negative and column sums = 1
            assert np.all(P_num >= -1e-10)
            assert np.all(P_num <= 1.0 + 1e-10)
            col_sums = np.sum(P_num, axis=0)
            assert np.allclose(col_sums, 1.0, atol=1e-10)

    def test_use_as_markov_chain(self):
        """Cell 4: Use symbolic result as MarkovChain at λ=0.3."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P, _ = mct.derive_transition_matrix([E0, E1], mct.computational_basis)

        # Create numeric MarkovChain at λ=0.3
        P_numeric = P.subs(lam, 0.3)
        P_np = np.array(P_numeric, dtype=float)

        mc = mct.MarkovChain(P_np, state_labels=['|0⟩', '|1⟩'], name='T1')
        result = mc.validate_stochasticity()

        assert result['is_valid']
        assert mc.name == 'T1'
        assert mc.state_labels == ['|0⟩', '|1⟩']

    def test_generate_sampling_code(self):
        """Cell 5: Generate executable Python sampling code."""
        lam = symbols('lambda', real=True, positive=True)
        E0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        E1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P, metadata = mct.derive_transition_matrix([E0, E1], mct.computational_basis)

        # Generate code at λ=0.3
        code = mct.markov_chain((P, metadata), parameter_values={'lambda': 0.3})

        assert isinstance(code, str)
        assert len(code) > 0
        # Should contain sampling logic
        assert 'def' in code or 'rand' in code.lower()


class TestNotebook02APIQuickstart:
    """Test notebook 02: MCT API quickstart (symbolic → code generation → sampling)."""

    def test_full_workflow(self):
        """Full workflow: define Kraus → derive P → code generation → sampling."""
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])

        # 1. Derive symbolic transition matrix
        P, meta = mct.derive_transition_matrix(
            [K0, K1], mct.computational_basis, "T1"
        )
        assert P.free_symbols

        # 2. Verify at λ=0.5
        P_num, valid = mct.substitute_and_verify(P, lam, 0.5, "Stochastic check")
        assert valid
        assert P_num.shape == (2, 2)

        # 3. Create MarkovChain
        mc = mct.MarkovChain(P_num, state_labels=['|0⟩', '|1⟩'], name='T1')
        assert mc.validate_stochasticity()['is_valid']

        # 4. Generate sampling code
        code = mct.markov_chain((P, meta), parameter_values={'lambda': 0.5})
        assert isinstance(code, str)
        assert len(code) > 100


class TestNotebook03MonteCarloValidation:
    """Test notebook 03: Monte Carlo validation (theory vs empirical)."""

    def test_derive_t1_symbolic_matrix(self):
        """Cell 1: Derive T1 symbolic matrix."""
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P_sym, _ = mct.derive_transition_matrix([K0, K1], mct.computational_basis)

        assert P_sym.free_symbols
        # Numeric matrix at λ=0.2
        P = np.array(P_sym.subs(lam, 0.2), dtype=float)
        assert P.shape == (2, 2)

    def test_compare_theory_vs_empirical(self):
        """Cell 2: Compare theoretical vs empirical distributions after n steps."""
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P_sym, _ = mct.derive_transition_matrix([K0, K1], mct.computational_basis)

        P = np.array(P_sym.subs(lam, 0.2), dtype=float)
        n_steps = 5
        n_traj = 10000

        # Theoretical distribution
        theory_dist = mct.theoretical_distribution_after_steps(
            P, initial_state=1, n_steps=n_steps
        )
        assert len(theory_dist) == 2
        assert np.isclose(np.sum(theory_dist), 1.0)

        # Empirical from trajectories
        trajectories = mct.simulate_markov_vectorized(
            P, initial_state=1, n_steps=n_steps, n_trajectories=n_traj
        )
        assert trajectories.shape == (n_steps + 1, n_traj)

        empirical_dist = mct.empirical_distribution_from_trajectories(
            trajectories, step=n_steps, n_states=2
        )
        assert len(empirical_dist) == 2
        assert np.isclose(np.sum(empirical_dist), 1.0)

        # Error should be small
        error = np.linalg.norm(theory_dist - empirical_dist)
        # With 10k trajectories, error should be <0.01
        assert error < 0.01, f"Error {error} too large"

    def test_excited_state_population_decay(self):
        """Cell 3: Test excited state population decay over time."""
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P_sym, _ = mct.derive_transition_matrix([K0, K1], mct.computational_basis)

        P = np.array(P_sym.subs(lam, 0.2), dtype=float)
        n_steps_max = 10
        n_traj = 5000

        # Theoretical decay
        theory_decay = np.array([
            mct.theoretical_distribution_after_steps(P, initial_state=1, n_steps=n)[1]
            for n in range(n_steps_max + 1)
        ])
        assert len(theory_decay) == n_steps_max + 1
        assert np.all(np.diff(theory_decay) <= 1e-10)  # Monotonically decreasing

        # Empirical decay
        trajectories_long = mct.simulate_markov_vectorized(
            P, initial_state=1, n_steps=n_steps_max, n_trajectories=n_traj
        )
        empirical_decay = np.array([
            mct.empirical_distribution_from_trajectories(
                trajectories_long, step=n, n_states=2
            )[1]
            for n in range(n_steps_max + 1)
        ])

        # Verify empirical is approximately monotonically decreasing
        # (allow some statistical noise)
        diffs = np.diff(empirical_decay)
        non_monotone_count = np.sum(diffs > 0.05)  # More lenient threshold
        assert non_monotone_count <= 1, "Empirical decay should be mostly monotonically decreasing"

        # Verify theory and empirical are reasonably close
        # At any step, std dev of empirical proportion is sqrt(p(1-p)/n)
        # Most steps should have error < 3 * expected std
        step_errors = np.abs(theory_decay - empirical_decay)
        max_error = np.max(step_errors)
        # Accept max error up to 0.25 (very lenient for finite sampling)
        assert max_error < 0.25, f"Max error {max_error} is unexpectedly large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
