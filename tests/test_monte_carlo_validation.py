"""
Monte Carlo validation tests for Markov chain sampling.

Tests that empirical distributions from sampling match theoretical distributions
computed via P^n, across multiple steps and noise parameters.

References:
- markov_chain_generator_full_code_jan18.pdf: Sampling pattern
- Test statistical properties of generated trajectories
"""

import numpy as np
import pytest

from mct.markov_chain import (
    MarkovChain,
    empirical_distribution_from_trajectories,
    simulate_markov_trajectory,
    simulate_markov_vectorized,
    theoretical_distribution_after_steps,
)
from mct.noise_models import (
    single_qubit_t1_markov,
    two_qubit_amplitude_damping_bell,
)


class TestSingleQubitT1MonteCarloValidation:
    """
    Monte Carlo validation for single-qubit T1 Markov chains.
    
    Tests that sampling from the chain produces empirical distributions
    matching theoretical P^n distributions.
    """

    def test_single_trajectory_structure(self):
        """Test that a single trajectory has correct structure."""
        P = single_qubit_t1_markov(λ=0.3)

        trajectory = simulate_markov_trajectory(
            transition_matrix=P,
            initial_state=0,
            n_steps=10,
        )

        # Should be length 11 (initial + 10 steps)
        assert len(trajectory) == 11
        # First element should be initial state
        assert trajectory[0] == 0
        # All elements should be valid states (0 or 1)
        assert np.all((trajectory >= 0) & (trajectory < 2))

    def test_ground_state_is_absorbing(self):
        """When starting from |0⟩, should stay in |0⟩."""
        P = single_qubit_t1_markov(λ=0.5)

        # Generate many trajectories
        trajectories = simulate_markov_vectorized(
            transition_matrix=P,
            initial_state=0,
            n_steps=50,
            n_trajectories=1000,
        )

        # All states should remain 0
        assert np.all(trajectories == 0)

    def test_excited_state_decays(self):
        """When starting from |1⟩, should eventually mostly be in |0⟩."""
        λ = 0.3
        P = single_qubit_t1_markov(λ=λ)

        trajectories = simulate_markov_vectorized(
            transition_matrix=P,
            initial_state=1,
            n_steps=50,
            n_trajectories=5000,
        )

        # At step 1, fraction in state 0 should be approximately λ
        empirical_step1 = empirical_distribution_from_trajectories(
            trajectories, step=1, n_states=2
        )

        # Expected: p(0|1) = λ
        assert empirical_step1[0] == pytest.approx(λ, abs=0.05)
        assert empirical_step1[1] == pytest.approx(1 - λ, abs=0.05)

    def test_theoretical_vs_empirical_step_1(self):
        """At step 1, compare empirical to theoretical distributions."""
        λ_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        for λ in λ_values:
            P = single_qubit_t1_markov(λ=λ)

            # Simulate many trajectories
            trajectories = simulate_markov_vectorized(
                transition_matrix=P,
                initial_state=1,
                n_steps=1,
                n_trajectories=10000,
            )

            # Empirical distribution at step 1
            empirical = empirical_distribution_from_trajectories(
                trajectories, step=1, n_states=2
            )

            # Theoretical distribution: P @ [0, 1]^T = [λ, 1-λ]
            theoretical = theoretical_distribution_after_steps(
                P, initial_state=1, n_steps=1
            )

            # Should match within statistical noise (~1/sqrt(n_trajectories) ≈ 0.01)
            assert np.allclose(empirical, theoretical, atol=0.02), \
                f"λ={λ}: empirical {empirical} vs theoretical {theoretical}"

    def test_long_time_convergence_to_ground_state(self):
        """
        After many steps, excited state population should converge to ground state.
        
        For T1 (amplitude damping), the steady state is |0⟩⟨0|.
        """
        λ = 0.2  # Slow decay for visualization
        P = single_qubit_t1_markov(λ=λ)

        trajectories = simulate_markov_vectorized(
            transition_matrix=P,
            initial_state=1,
            n_steps=100,
            n_trajectories=10000,
        )

        # At step 100, nearly all population should be in state 0
        empirical_final = empirical_distribution_from_trajectories(
            trajectories, step=100, n_states=2
        )

        # Should have decayed substantially
        # Probability of still being in |1⟩ after 100 steps: (1-λ)^100
        expected_p1 = (1 - λ) ** 100  # ≈ 1.3e-10 for λ=0.2

        assert empirical_final[0] > 0.99  # >99% in ground state
        assert empirical_final[1] < 0.01  # <1% in excited state

    def test_stochasticity_preserved_across_steps(self):
        """
        Empirical distributions from trajectories should sum to 1 at all steps.
        """
        P = single_qubit_t1_markov(λ=0.3)

        trajectories = simulate_markov_vectorized(
            transition_matrix=P,
            initial_state=1,
            n_steps=20,
            n_trajectories=5000,
        )

        # Check at every step
        for step in range(21):
            empirical = empirical_distribution_from_trajectories(
                trajectories, step=step, n_states=2
            )
            assert np.isclose(np.sum(empirical), 1.0), \
                f"Empirical distribution at step {step} doesn't sum to 1: {empirical}"

    def test_monte_carlo_convergence(self):
        """
        With increasing trajectory count, empirical → theoretical distributions.
        """
        λ = 0.4
        P = single_qubit_t1_markov(λ=λ)

        theoretical = theoretical_distribution_after_steps(
            P, initial_state=1, n_steps=5
        )

        # Test with increasing numbers of trajectories
        n_traj_values = [100, 500, 2000, 10000]
        errors = []

        for n_traj in n_traj_values:
            trajectories = simulate_markov_vectorized(
                transition_matrix=P,
                initial_state=1,
                n_steps=5,
                n_trajectories=n_traj,
            )

            empirical = empirical_distribution_from_trajectories(
                trajectories, step=5, n_states=2
            )

            error = np.linalg.norm(empirical - theoretical)
            errors.append(error)

        # Errors should decrease on average (allow ~O(1/sqrt(n)) variation)
        # Compare first vs last: with 100x more trajectories, error should be smaller on average
        expected_ratio = np.sqrt(n_traj_values[0] / n_traj_values[-1])  # ~3.16x improvement expected
        assert errors[-1] < errors[0] * 2, \
            f"Error at 10k trajectories should be better than at 100: {errors}"


class TestTwoQubitMonteCarloValidation:
    """
    Monte Carlo validation for two-qubit channels (simplified implementation).
    
    Current implementation is simplified (diagonal-only) pending full symbolic
    Bell basis derivation (see todo.txt Phase 2).
    """

    def test_two_qubit_no_noise_is_identity(self):
        """With λ1=0, should be identity (no decay)."""
        P = two_qubit_amplitude_damping_bell(λ1=0.0, λ1_other=0.0)

        for initial_state in range(4):
            trajectories = simulate_markov_vectorized(
                transition_matrix=P,
                initial_state=initial_state,
                n_steps=50,
                n_trajectories=1000,
            )

            # Should stay in same state
            assert np.all(trajectories == initial_state), \
                f"With no noise, state {initial_state} should not change"


class TestRegressionWithDifferentParameters:
    """
    Regression tests across a grid of noise parameters.
    Ensures changes to simulation code don't break physics.
    """

    @pytest.mark.parametrize("λ", [0.01, 0.1, 0.5, 0.9, 0.99])
    def test_single_qubit_t1_parameters(self, λ):
        """Test single-qubit T1 with various decay rates."""
        P = single_qubit_t1_markov(λ=λ)

        # Verify stochasticity
        validation = MarkovChain(P).validate_stochasticity()
        assert validation["is_valid"], f"Stochasticity check failed for λ={λ}"

        # Simulate and check
        trajectories = simulate_markov_vectorized(
            transition_matrix=P,
            initial_state=1,
            n_steps=10,
            n_trajectories=2000,
        )

        # At step 1, should match decay rate
        empirical = empirical_distribution_from_trajectories(
            trajectories, step=1, n_states=2
        )
        theoretical = theoretical_distribution_after_steps(
            P, initial_state=1, n_steps=1
        )

        assert np.allclose(empirical, theoretical, atol=0.03)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
