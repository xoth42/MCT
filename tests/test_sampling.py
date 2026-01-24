"""
Tests for sampling code generation.
"""

import numpy as np
import pytest

from mct.markov_chain import generate_sampling_function, generate_sampling_vectorized


class TestSamplingGeneration:
    """Test sampling code generation."""

    def test_generate_simple_2state(self):
        """Test code generation for 2-state chain."""
        P = np.array([
            [0.8, 0.3],
            [0.2, 0.7]
        ])

        code = generate_sampling_function(
            P,
            state_labels=["A", "B"],
            function_name="sample_2state"
        )

        # Check that code is valid Python
        assert "def sample_2state(current_state: int) -> int:" in code
        assert "if current_state == 0:" in code
        assert "if current_state == 1:" in code
        assert "random.random()" in code

        # Code should be executable
        namespace = {}
        exec(code, namespace)
        assert "sample_2state" in namespace

    def test_generated_function_samples_correctly(self):
        """Test that generated function produces correct distribution."""
        # Simple identity chain
        P = np.eye(3)
        code = generate_sampling_function(P, function_name="sample_identity")

        namespace = {}
        exec(code, namespace)
        sample_fn = namespace["sample_identity"]

        # From each state, should return same state
        for state in range(3):
            next_state = sample_fn(state)
            assert next_state == state

    def test_generated_function_respects_probabilities(self):
        """Test that generated function respects transition probabilities."""
        # 2-state chain: always stay
        P = np.eye(2)
        code = generate_sampling_function(P, function_name="sample_stay")

        namespace = {}
        exec(code, namespace)
        sample_fn = namespace["sample_stay"]

        # Many samples should all stay
        samples = [sample_fn(0) for _ in range(100)]
        assert all(s == 0 for s in samples)

    def test_generate_vectorized(self):
        """Test vectorized code generation."""
        P = np.array([
            [0.5, 0.3],
            [0.5, 0.7]
        ])

        code = generate_sampling_vectorized(P, function_name="sample_traj")

        # Check code validity
        assert "def sample_traj(initial_state: int, n_steps: int) -> list:" in code
        assert "np.random.choice" in code
        assert "trajectory = [initial_state]" in code

        # Should be executable
        namespace = {"np": np}
        exec(code, namespace)
        assert "sample_traj" in namespace

    def test_vectorized_function_produces_trajectory(self):
        """Test that vectorized function produces correct length trajectory."""
        P = np.eye(2)
        code = generate_sampling_vectorized(P, function_name="sample_traj")

        namespace = {"np": np}
        exec(code, namespace)
        traj_fn = namespace["sample_traj"]

        # Generate trajectory
        traj = traj_fn(initial_state=0, n_steps=10)

        # Check length
        assert len(traj) == 11  # initial + 10 steps

        # For identity, all should be 0
        assert all(s == 0 for s in traj)


class TestCodeGeneration:
    """Test code generation properties."""

    def test_cumulative_probabilities_correct(self):
        """Verify cumulative probability computation in generated code."""
        P = np.array([
            [0.6, 0.4],
            [0.4, 0.6]
        ])

        code = generate_sampling_function(P, function_name="test_fn")

        # Cumulative should be [0.6, 1.0] for column 0 and [0.4, 1.0] for column 1
        # These values should appear in the generated code
        assert "0.6" in code or "0.600" in code
        assert "1.0" in code or "1.000" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
