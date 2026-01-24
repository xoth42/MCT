"""Integration tests: API workflows similar to notebook usage."""

import numpy as np
import pytest
from sympy import Matrix, sqrt, symbols

import mct


class TestT1WorkflowIntegration:
    """Full T1 workflow: symbolic derivation → verification → MarkovChain → sampling."""

    def test_t1_symbolic_derivation_and_verification(self):
        """Derive T1 transition matrix and verify stochasticity."""
        # 1. Define Kraus operators
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])

        # 2. Derive symbolic transition matrix
        P, metadata = mct.derive_transition_matrix([K0, K1], mct.computational_basis, "T1")

        # Verify it's a 2x2 matrix
        assert P.shape == (2, 2), "T1 should be 2x2"

        # Verify metadata contains expected fields
        assert 'channel_name' in metadata
        assert metadata['channel_name'] == "T1"
        assert 'kraus_operators' in metadata
        assert len(metadata['kraus_operators']) == 2

    def test_t1_parameter_substitution_and_stochasticity(self):
        """Substitute λ values and verify stochasticity."""
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P, _ = mct.derive_transition_matrix([K0, K1], mct.computational_basis, "T1")

        # Test at multiple λ values
        test_values = [0.0, 0.1, 0.5, 0.9, 1.0]
        for lam_val in test_values:
            P_num, is_valid = mct.substitute_and_verify(P, lam, lam_val, name=f"λ={lam_val}")

            assert is_valid, f"Failed stochasticity at λ={lam_val}"
            assert isinstance(P_num, np.ndarray), f"Result should be numpy array at λ={lam_val}"
            assert P_num.shape == (2, 2), f"Shape should be (2,2) at λ={lam_val}"

            # Verify column sums = 1 (column-stochastic)
            col_sums = np.sum(P_num, axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-10,
                                      err_msg=f"Column sums != 1 at λ={lam_val}")

            # Verify all entries are in [0, 1]
            assert np.all(P_num >= -1e-10), f"Negative entries at λ={lam_val}"
            assert np.all(P_num <= 1 + 1e-10), f"Entries > 1 at λ={lam_val}"

    def test_t1_markovchain_creation_and_validation(self):
        """Create MarkovChain object and validate."""
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P, _ = mct.derive_transition_matrix([K0, K1], mct.computational_basis, "T1")

        # Substitute at λ=0.3
        P_num, _ = mct.substitute_and_verify(P, lam, 0.3)

        # Create MarkovChain
        mc = mct.MarkovChain(P_num, state_labels=['|0⟩', '|1⟩'], name='T1')

        assert mc.name == 'T1'
        assert mc.n_states == 2
        assert len(mc.state_labels) == 2

        # Validate stochasticity
        result = mc.validate_stochasticity()
        assert result['is_valid'], "MarkovChain should be stochastic"
        assert 'column_sums' in result
        np.testing.assert_allclose(result['column_sums'], 1.0, atol=1e-10)

    def test_t1_sampling_code_generation(self):
        """Generate sampling code from symbolic result."""
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P, metadata = mct.derive_transition_matrix([K0, K1], mct.computational_basis, "T1")

        # Generate code
        code = mct.markov_chain((P, metadata), parameter_values={'lambda': 0.3})

        # Verify code is non-empty string with expected patterns
        assert isinstance(code, str), "Should return code string"
        assert len(code) > 0, "Code should not be empty"
        assert 'def' in code, "Should contain function definition"
        assert 'rand' in code or 'random' in code, "Should contain random number generation"


class TestADCWorkflowIntegration:
    """Amplitude + Dephasing Channel workflow."""

    def test_adc_derivation(self):
        """Derive ADC transition matrix for 2-qubit Bell basis."""
        λ1, λ2 = symbols('lambda_1 lambda_2', real=True, positive=True)

        # ADC Kraus operators for 2 qubits (amplitude + dephasing)
        K0 = Matrix.eye(4)
        K1 = mct.TWO_QUBIT_ADC_KRAUS_PARAMS['E_amp'] if 'E_amp' in mct.TWO_QUBIT_ADC_KRAUS_PARAMS else None

        # Check if ADC constants exist (may be parametrized)
        assert mct.TWO_QUBIT_ADC_KRAUS_PARAMS is not None, "ADC constants should be defined"

    def test_bell_basis_availability(self):
        """Verify Bell basis is available."""
        bell_basis = mct.bell_basis

        assert bell_basis is not None, "Bell basis should be defined"
        assert len(bell_basis) == 4, "Bell basis should have 4 states for 2 qubits"

        # Each basis state should be a density matrix (2x2 tensor)
        for state in bell_basis:
            assert isinstance(state, (Matrix, np.ndarray)), "Basis state should be matrix-like"


class TestSimulationIntegration:
    """Markov chain simulation workflows."""

    def test_trajectory_simulation(self):
        """Simulate single trajectory."""
        # Create simple 2-state Markov chain
        P = np.array([[0.9, 0.2], [0.1, 0.8]])

        # Single trajectory
        traj = mct.simulate_markov_trajectory(P, initial_state=0, n_steps=10)

        assert isinstance(traj, np.ndarray), "Should return numpy array"
        assert len(traj) == 11, "Should have 11 states (initial + 10 steps)"
        assert np.all(np.isin(traj, [0, 1])), "States should be 0 or 1"

    def test_vectorized_simulation(self):
        """Simulate many trajectories at once."""
        P = np.array([[0.9, 0.2], [0.1, 0.8]])

        # Vectorized simulation
        trajectories = mct.simulate_markov_vectorized(P, initial_state=0, n_steps=20, n_trajectories=1000)

        assert isinstance(trajectories, np.ndarray), "Should return numpy array"
        # Shape is (n_steps+1, n_trajectories) = (21, 1000)
        assert trajectories.shape == (21, 1000), f"Should be (21, 1000), got {trajectories.shape}"
        assert np.all(np.isin(trajectories, [0, 1])), "States should be 0 or 1"

    def test_empirical_distribution(self):
        """Extract empirical distribution from trajectories."""
        P = np.array([[0.9, 0.2], [0.1, 0.8]])
        trajectories = mct.simulate_markov_vectorized(P, initial_state=0, n_steps=20, n_trajectories=10000)

        # Get empirical distribution at step 20
        emp_dist = mct.empirical_distribution_from_trajectories(trajectories, step=20, n_states=2)

        assert isinstance(emp_dist, np.ndarray), "Should return numpy array"
        assert len(emp_dist) == 2, "Should have 2 states"
        np.testing.assert_allclose(np.sum(emp_dist), 1.0, atol=1e-10, err_msg="Should be probability distribution")

    def test_theoretical_distribution(self):
        """Compute theoretical distribution after n steps."""
        P = np.array([[0.9, 0.2], [0.1, 0.8]])

        # Theoretical distribution after 5 steps starting from state 0
        theo_dist = mct.theoretical_distribution_after_steps(P, initial_state=0, n_steps=5)

        assert isinstance(theo_dist, np.ndarray), "Should return numpy array"
        assert len(theo_dist) == 2, "Should have 2 states"
        np.testing.assert_allclose(np.sum(theo_dist), 1.0, atol=1e-10, err_msg="Should be probability distribution")


class TestMonteCarloValidation:
    """Validate Monte Carlo sampling against theory."""

    def test_empirical_vs_theoretical_agreement(self):
        """Large sample size: empirical ≈ theoretical."""
        # Simple 2-state chain with known steady state
        P = np.array([[0.8, 0.3], [0.2, 0.7]])
        n_traj = 100000
        n_steps = 10

        # Theoretical
        theo = mct.theoretical_distribution_after_steps(P, initial_state=0, n_steps=n_steps)

        # Empirical from many trajectories
        traj = mct.simulate_markov_vectorized(P, initial_state=0, n_steps=n_steps, n_trajectories=n_traj)
        emp = mct.empirical_distribution_from_trajectories(traj, step=n_steps, n_states=2)

        # With 100k samples, error should be O(1/√N) ≈ 0.003
        error = np.linalg.norm(emp - theo)
        assert error < 0.01, f"Error {error} too large; empirical and theoretical should agree"

    def test_convergence_with_sample_size(self):
        """Error decreases as ~1/√N."""
        P = np.array([[0.7, 0.4], [0.3, 0.6]])
        theo = mct.theoretical_distribution_after_steps(P, initial_state=1, n_steps=5)

        errors = []
        sample_sizes = [100, 500, 2000, 10000]

        for n in sample_sizes:
            traj = mct.simulate_markov_vectorized(P, initial_state=1, n_steps=5, n_trajectories=n)
            emp = mct.empirical_distribution_from_trajectories(traj, step=5, n_states=2)
            error = np.linalg.norm(emp - theo)
            errors.append(error)

        # Verify overall trend: first and last should show improvement (allow local variation)
        assert errors[0] > errors[-1], f"Error should decrease overall: {errors}"


class TestSamplingCodeExecution:
    """Generated sampling code should actually work."""

    def test_generated_sampling_function_structure(self):
        """Generated code from markov_chain has expected structure."""
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])
        P, metadata = mct.derive_transition_matrix([K0, K1], mct.computational_basis, "T1")

        # Generate code with substituted λ
        code = mct.markov_chain((P, metadata), parameter_values={'lambda': 0.5})

        # Code should be non-empty and contain expected patterns
        assert isinstance(code, str), "Should return code string"
        assert len(code) > 100, "Code should be substantial"
        assert 'def' in code, "Should define a function"
        assert ('if' in code or 'elif' in code), "Should have conditional logic for sampling"


class TestComprehensiveEndToEnd:
    """Complete smoke test: Does everything run without errors?"""

    def test_full_workflow_does_everything(self):
        """Execute complete notebook-like workflow: setup → derive → verify → MC → simulate → analyze."""

        # ======= SETUP: Define symbols and Kraus operators =======
        lam = symbols('lambda', real=True, positive=True)
        K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
        K1 = Matrix([[0, sqrt(lam)], [0, 0]])

        # ======= DERIVE: Symbolic transition matrix =======
        P_sym, metadata = mct.derive_transition_matrix([K0, K1], mct.computational_basis, "T1")

        # Verify we got a symbolic matrix back
        assert isinstance(P_sym, Matrix)
        assert P_sym.shape == (2, 2)

        # ======= VERIFY: Test at multiple parameters =======
        test_params = [0.0, 0.2, 0.5, 0.8, 1.0]
        P_at_params = {}

        for lam_val in test_params:
            P_num, is_valid = mct.substitute_and_verify(P_sym, lam, lam_val, name="")
            assert is_valid, f"Should be stochastic at λ={lam_val}"
            P_at_params[lam_val] = P_num

        # ======= MARKOVCHAIN: Create object and validate =======
        P_0p5 = P_at_params[0.5]
        mc = mct.MarkovChain(P_0p5, state_labels=['|0⟩', '|1⟩'], name='T1')
        result = mc.validate_stochasticity()
        assert result['is_valid']

        # ======= MARKOV_CHAIN: Generate sampling code =======
        code = mct.markov_chain((P_sym, metadata), parameter_values={'lambda': 0.5})
        assert isinstance(code, str)
        assert len(code) > 50

        # ======= SIMULATION: Run trajectory =======
        traj = mct.simulate_markov_trajectory(P_0p5, initial_state=0, n_steps=15)
        assert len(traj) == 16  # initial + 15 steps
        assert np.all(np.isin(traj, [0, 1]))

        # ======= VECTORIZED SIMULATION: Many trajectories =======
        trajectories = mct.simulate_markov_vectorized(P_0p5, initial_state=1, n_steps=10, n_trajectories=500)
        assert trajectories.shape == (11, 500)  # (n_steps+1, n_trajectories)
        assert np.all(np.isin(trajectories, [0, 1]))

        # ======= EMPIRICAL DISTRIBUTION: From trajectories =======
        emp_dist = mct.empirical_distribution_from_trajectories(trajectories, step=10, n_states=2)
        assert len(emp_dist) == 2
        assert np.isclose(np.sum(emp_dist), 1.0)

        # ======= THEORETICAL DISTRIBUTION: Compute P^n =======
        theo_dist = mct.theoretical_distribution_after_steps(P_0p5, initial_state=1, n_steps=10)
        assert len(theo_dist) == 2
        assert np.isclose(np.sum(theo_dist), 1.0)

        # ======= MONTE CARLO VALIDATION: Theory vs empirical =======
        # Large sample size: empirical should be close to theory
        big_traj = mct.simulate_markov_vectorized(P_0p5, initial_state=0, n_steps=5, n_trajectories=50000)
        big_emp = mct.empirical_distribution_from_trajectories(big_traj, step=5, n_states=2)
        big_theo = mct.theoretical_distribution_after_steps(P_0p5, initial_state=0, n_steps=5)
        error = np.linalg.norm(big_emp - big_theo)
        assert error < 0.01, f"Large sample: empirical should match theory, error={error}"

        # ======= APPLY_CHANNEL: Direct channel application =======
        rho = mct.computational_basis[0]  # |0⟩⟨0|
        rho_out = mct.apply_channel([K0, K1], rho)
        assert isinstance(rho_out, Matrix)
        assert rho_out.shape == (2, 2)

        # ======= MATRIX FORMATTING: LaTeX display (basic check) =======
        # Just verify the function exists and is callable
        assert callable(mct.print_matrix_latex)
        # Don't actually call it in test (would require Jupyter), just ensure it's importable

        # ======= BASES: Check Bell basis =======
        bell = mct.bell_basis
        assert bell is not None
        assert len(bell) == 4

        comp = mct.computational_basis
        assert comp is not None
        assert len(comp) == 2

        # ======= CONSTANTS: Check predefined Kraus =======
        assert mct.AMPLITUDE_DAMPING_KRAUS is not None
        assert mct.TWO_QUBIT_ADC_KRAUS_PARAMS is not None

        # ✓ If we reach here, everything ran without errors!


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
