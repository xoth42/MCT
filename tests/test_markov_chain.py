"""
Tests for MarkovChain core functionality.
"""

import numpy as np
import pytest

from mct.markov_chain import MarkovChain


class TestMarkovChainBasics:
    """Test basic MarkovChain initialization and validation."""

    def test_identity_matrix(self):
        """Test that identity matrix is a valid (trivial) Markov chain."""
        P = np.eye(3)
        mc = MarkovChain(P, name="Identity")

        assert mc.n_states == 3
        assert mc.state_labels == ["0", "1", "2"]

        # Validate
        result = mc.validate_stochasticity()
        assert result["is_valid"]

    def test_simple_2state(self):
        """Test a simple 2-state Markov chain."""
        # States 0 <-> 1
        # From state 0: 80% stay, 20% go to 1
        # From state 1: 30% go to 0, 70% stay
        P = np.array([
            [0.8, 0.3],
            [0.2, 0.7]
        ])
        mc = MarkovChain(P, state_labels=["A", "B"], name="TwoState")

        assert mc.n_states == 2
        assert mc.state_labels == ["A", "B"]

        result = mc.validate_stochasticity()
        assert result["is_valid"]
        assert np.allclose(result["column_sums"], [1.0, 1.0])

    def test_invalid_not_square(self):
        """Test that non-square matrices are rejected."""
        P = np.array([[0.5, 0.5, 0.5]])

        with pytest.raises(ValueError, match="square"):
            MarkovChain(P)

    def test_invalid_entries_out_of_bounds(self):
        """Test detection of entries outside [0, 1]."""
        P = np.array([
            [1.2, 0.3],
            [-0.2, 0.7]
        ])
        mc = MarkovChain(P)  # Creation is allowed

        result = mc.validate_stochasticity()
        assert not result["is_valid"]
        assert "entries_out_of_bounds" in result["violations"]

    def test_invalid_columns_not_sum_to_one(self):
        """Test detection of columns not summing to 1."""
        P = np.array([
            [0.5, 0.3],
            [0.4, 0.5]
        ])
        mc = MarkovChain(P)

        result = mc.validate_stochasticity()
        assert not result["is_valid"]
        assert "columns_do_not_sum_to_one" in result["violations"]


class TestMarkovChainDynamics:
    """Test time evolution of Markov chains."""

    def test_apply_steps_identity(self):
        """Test that identity chain preserves state distribution."""
        P = np.eye(3)
        mc = MarkovChain(P)

        # Start in state 0
        initial = np.array([1.0, 0.0, 0.0])

        # After any number of steps, should stay in state 0
        for n_steps in [1, 5, 100]:
            result = mc.apply_steps(initial, n_steps)
            assert np.allclose(result, initial)

    def test_apply_steps_convergence(self):
        """Test convergence to stationary distribution."""
        # Simple 2-state chain
        P = np.array([
            [0.9, 0.2],
            [0.1, 0.8]
        ])
        mc = MarkovChain(P)

        # Start in state 0
        initial = np.array([1.0, 0.0])

        # Apply many steps
        final = mc.apply_steps(initial, 1000)

        # Should converge to stationary distribution
        # For this P: π = [2/3, 1/3]
        expected = np.array([2/3, 1/3])
        assert np.allclose(final, expected, atol=1e-8)

    def test_apply_steps_uniformity(self):
        """Test a chain with uniform stationary distribution."""
        # 3-state fully connected (each state equally likely)
        P = np.ones((3, 3)) / 3
        mc = MarkovChain(P)

        initial = np.array([1.0, 0.0, 0.0])

        # After one step, should be uniform
        result = mc.apply_steps(initial, 1)
        expected = np.array([1/3, 1/3, 1/3])
        assert np.allclose(result, expected)


class TestMarkovChainPhysics:
    """Test physically motivated Markov chains."""

    def test_amplitude_damping_like(self):
        """
        Test a 2-state chain representing amplitude damping.
        
        State |0> relaxes to |0>, state |1> has probability λ to relax to |0>.
        This is the simplest Bell-space simplification.
        """
        λ = 0.1  # relaxation probability per step
        P = np.array([
            [1.0, λ],      # prob(0 | 0), prob(0 | 1)
            [0.0, 1 - λ]   # prob(1 | 0), prob(1 | 1)
        ])
        mc = MarkovChain(P, state_labels=["|0>", "|1>"], name="AmplitudeDamping")

        # Validate
        result = mc.validate_stochasticity()
        assert result["is_valid"]

        # After many steps starting in |1>, should be close to |0>
        initial = np.array([0.0, 1.0])
        final = mc.apply_steps(initial, 1000)
        assert final[0] > 0.99  # mostly in |0>
        assert final[1] < 0.01  # mostly out of |1>

    def test_4state_bell_chain(self):
        """Test a 4-state chain on Bell basis {|Φ+>, |Φ->, |Ψ+>, |Ψ->}."""
        # 4-state Bell basis chain with decay and leakage
        # Each Bell state decays to ground (leakage) at different rates
        decay_rates = [0.05, 0.10, 0.08, 0.12]

        # Create stochastic matrix: diagonal stays in state,
        # off-diagonals (uniform leakage to ground state 0) ensure columns sum to 1
        P = np.zeros((4, 4))
        for j in range(4):
            P[j, j] = 1.0 - decay_rates[j]  # Stay in same Bell state
            P[0, j] += decay_rates[j]  # Leak to ground state

        mc = MarkovChain(
            P,
            state_labels=["|Φ+>", "|Φ->", "|Ψ+>", "|Ψ->"],
            name="BellBasis"
        )

        result = mc.validate_stochasticity()
        assert result["is_valid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
