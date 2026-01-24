"""
Tests for noise_models module.
Comprehensive tests for T1/T2 noise models, two-qubit channels, and Markov chains.
"""

import math

import numpy as np
import pytest

from mct.noise_models import (
    single_qubit_combined_t1t2,
    single_qubit_t1_markov,
    single_qubit_t2_markov,
    thermal_relaxation_error_rate,
    two_qubit_adc_bell,
    two_qubit_amplitude_damping_bell,
)


class TestThermalRelaxation:
    """Test T1/T2 thermal relaxation computation against physics formulas."""

    def test_zero_idle_time(self):
        """With zero idle time, probabilities should be zero."""
        λ1, λ_φ = thermal_relaxation_error_rate(T1=100.0, T2=150.0, idle_time=0.0)
        assert λ1 == 0.0
        assert λ_φ == 0.0

    def test_t1_formula_accuracy(self):
        """Test T1 relaxation against exact exponential formula: λ1 = 1 - exp(-t/T1)."""
        T1 = 100.0  # μs
        idle_time = 10.0  # μs

        λ1, _ = thermal_relaxation_error_rate(T1=T1, T2=200.0, idle_time=idle_time)

        expected_λ1 = 1.0 - math.exp(-idle_time / T1)
        assert np.isclose(λ1, expected_λ1), \
            f"T1 formula mismatch: got {λ1}, expected {expected_λ1}"

    def test_t2_formula_with_realistic_params(self):
        """
        Test T2 (pure dephasing) with realistic transmon parameters.
        
        Formula: λ_φ = 0.5 * (1 - exp(-idle_time * dephasing_rate))
        where dephasing_rate = 1/T2 - 1/(2*T1)
        
        Physics: Typical transmon has T2 < 2*T1, dephasing_rate > 0.
        """
        T1 = 85.0  # μs (typical)
        T2 = 120.0  # μs (typical, constraint: T2 < 2*T1 = 170)
        idle_time = 5.0  # μs

        λ1, λ_φ = thermal_relaxation_error_rate(T1=T1, T2=T2, idle_time=idle_time)

        # Verify against formula
        expected_λ1 = 1.0 - math.exp(-idle_time / T1)
        dephasing_rate = (1.0 / T2) - (1.0 / (2.0 * T1))
        expected_λ_φ = 0.5 * (1.0 - math.exp(-idle_time * dephasing_rate))

        assert np.isclose(λ1, expected_λ1), f"T1 mismatch: {λ1} vs {expected_λ1}"
        assert np.isclose(λ_φ, expected_λ_φ), f"T2 mismatch: {λ_φ} vs {expected_λ_φ}"

    def test_t2_limit_case_t2_equals_2t1(self):
        """
        When T2 = 2*T1, pure dephasing rate → 0, so λ_φ → 0.
        This is the limit where there is no pure dephasing (only amplitude damping).
        """
        T1 = 100.0
        T2 = 2 * T1  # Edge case
        idle_time = 1.0

        λ1, λ_φ = thermal_relaxation_error_rate(T1=T1, T2=T2, idle_time=idle_time)

        # λ1 should be nonzero
        assert λ1 > 0.0, "T1 effect should be present"
        # λ_φ should be zero or very small (dephasing_rate ≤ 0)
        assert λ_φ == 0.0 or λ_φ < 1e-10, \
            f"When T2=2*T1, λ_φ should be ~0, got {λ_φ}"

    def test_invalid_t1(self):
        """Negative or zero T1 should raise."""
        with pytest.raises(ValueError):
            thermal_relaxation_error_rate(T1=0.0, T2=150.0, idle_time=1.0)

    def test_invalid_t2(self):
        """Negative or zero T2 should raise."""
        with pytest.raises(ValueError):
            thermal_relaxation_error_rate(T1=100.0, T2=0.0, idle_time=1.0)

    def test_invalid_idle_time(self):
        """Negative idle time should raise."""
        with pytest.raises(ValueError):
            thermal_relaxation_error_rate(T1=100.0, T2=150.0, idle_time=-1.0)


class TestSingleQubitT1:
    """
    Test single-qubit T1 (amplitude damping) Markov chain.
    
    Kraus operators (from CS648QML_error_kraus.py):
      E0 = [[1, 0], [0, sqrt(1-λ)]]
      E1 = [[0, sqrt(λ)], [0, 0]]
    
    Transition matrix P[i,j] = Prob(i|j) where states are |0>, |1>.
    Physical meaning: |1> → |0> with probability λ, |0> absorbing state.
    """

    def test_zero_decay_is_identity(self):
        """λ=0 means no decay: P should be identity."""
        P = single_qubit_t1_markov(λ=0.0)
        expected = np.eye(2)
        assert np.allclose(P, expected)

    def test_complete_decay_to_ground_state(self):
        """
        λ=1 means complete decay: |1> → |0> with 100% probability.
        
        Physical: All population in |1> decays to |0>.
        Kraus: E1 = [[0, 1], [0, 0]] → |1>⟨0| (all probability flows here)
        """
        P = single_qubit_t1_markov(λ=1.0)

        # Column 0 (from |0>): stays in |0>
        assert P[0, 0] == 1.0, "|0> must be absorbing state"
        assert P[1, 0] == 0.0

        # Column 1 (from |1>): all → |0>
        assert P[0, 1] == 1.0, "All population in |1> decays to |0>"
        assert P[1, 1] == 0.0

    def test_partial_decay_markov_chain(self):
        """
        λ=0.3 means |1> → |0> with 30% probability per step.
        
        After Kraus application:
          ρ_out = E0 ρ E0† + E1 ρ E1†
        
        For ρ = |1>⟨1|:
          ρ_out[0,0] = λ = 0.3
          ρ_out[1,1] = (1-λ) = 0.7
        """
        λ = 0.3
        P = single_qubit_t1_markov(λ)

        # From |1>: transition probabilities
        assert P[0, 1] == pytest.approx(λ), "Prob(|0>||1>) should be λ"
        assert P[1, 1] == pytest.approx(1.0 - λ), "Prob(|1>||1>) should be 1-λ"

        # From |0>: no transition (absorbing)
        assert P[0, 0] == pytest.approx(1.0)
        assert P[1, 0] == pytest.approx(0.0)

    def test_column_stochasticity(self):
        """Each column must sum to 1 (proper Markov chain)."""
        for λ in [0.0, 0.25, 0.5, 0.75, 1.0]:
            P = single_qubit_t1_markov(λ)
            col_sums = np.sum(P, axis=0)
            assert np.allclose(col_sums, 1.0), f"Failed for λ={λ}"
            # All entries non-negative
            assert np.all(P >= 0.0)


class TestSingleQubitT2:
    """
    Test single-qubit T2 (pure dephasing) Markov chain.
    
    Kraus operators (from CS648QML_error_kraus.py):
      K0 = sqrt(1-λ_φ) * I
      K1 = sqrt(λ_φ) * Z
    
    Channel: ρ ↦ (1-λ_φ)ρ + λ_φ*Z*ρ*Z†
    
    Physical meaning: Dephasing destroys coherences (ρ_01, ρ_10) but preserves
    populations (ρ_00, ρ_11). In computational basis Markov chain: identity matrix.
    """

    def test_t2_is_identity_all_values(self):
        """
        Pure dephasing in computational basis doesn't change populations.
        Result: identity matrix for all λ_φ.
        """
        for λ_φ in [0.0, 0.3, 0.7, 1.0]:
            P = single_qubit_t2_markov(λ_φ)
            expected = np.eye(2)
            assert np.allclose(P, expected), \
                f"T2 dephasing should be identity in computational basis, got λ_φ={λ_φ}"

    def test_t2_column_stochasticity(self):
        """T2 matrix must be column-stochastic (sum to 1)."""
        for λ_φ in [0.0, 0.25, 0.5, 0.75, 1.0]:
            P = single_qubit_t2_markov(λ_φ)
            col_sums = np.sum(P, axis=0)
            assert np.allclose(col_sums, 1.0), f"Failed for λ_φ={λ_φ}"
            assert np.all(P >= 0.0)


class TestSingleQubitCombinedT1T2:
    """
    Test single-qubit combined T1 + T2 noise.
    
    Physically: Both amplitude damping (T1) and pure dephasing (T2) occur.
    In this simplified model: T1 dominates relaxation; T2 modulates dephasing.
    """

    def test_no_noise_gives_identity(self):
        """λ1=0, λ_φ=0 should give identity."""
        P = single_qubit_combined_t1t2(λ1=0.0, λ_φ=0.0)
        expected = np.eye(2)
        assert np.allclose(P, expected)

    def test_pure_t1_matches_t1_model(self):
        """When λ_φ=0, combined should match T1-only Markov chain."""
        λ1 = 0.4
        P_combined = single_qubit_combined_t1t2(λ1, λ_φ=0.0)
        P_t1 = single_qubit_t1_markov(λ1)
        assert np.allclose(P_combined, P_t1), \
            "Combined T1+T2 with λ_φ=0 should equal T1-only model"

    def test_t1_dominates(self):
        """T1 transitions should dominate over T2."""
        λ1 = 0.5
        λ_φ = 0.1
        P = single_qubit_combined_t1t2(λ1, λ_φ)

        # Column 1 (from |1>): transitions should be driven by λ1
        assert P[0, 1] == pytest.approx(λ1, abs=0.01), \
            "Prob(|0>||1>) should be dominated by T1"
        assert P[1, 1] == pytest.approx(1.0 - λ1, abs=0.01)

    def test_column_stochasticity(self):
        """Combined model must be column-stochastic."""
        for λ1 in [0.0, 0.3, 0.8]:
            for λ_φ in [0.0, 0.2, 0.6]:
                P = single_qubit_combined_t1t2(λ1, λ_φ)
                col_sums = np.sum(P, axis=0)
                assert np.allclose(col_sums, 1.0), \
                    f"Failed for λ1={λ1}, λ_φ={λ_φ}"
                assert np.all(P >= 0.0)


class TestTwoQubitAmplitudeDampingBell:
    """
    Test two-qubit T1 in Bell basis.
    
    Physics: Bell pairs decay under T1 relaxation. Each qubit has decay rate λ.
    In simplified model: diagonal with average decay across qubits.
    """

    def test_zero_decay_is_identity(self):
        """λ1=0 should give identity (no decay)."""
        P = two_qubit_amplitude_damping_bell(λ1=0.0, λ1_other=0.0)
        expected = np.eye(4)
        assert np.allclose(P, expected)

    def test_symmetric_decay_is_uniform_diagonal(self):
        """Symmetric decay (both qubits same λ) gives uniform diagonal."""
        λ = 0.3
        P = two_qubit_amplitude_damping_bell(λ1=λ)

        # All diagonal entries should be 1-λ (with λ1_other defaulting to λ)
        assert np.allclose(np.diag(P), [1.0 - λ] * 4), \
            f"Symmetric decay should give uniform diagonal, got {np.diag(P)}"

    def test_asymmetric_decay_uses_average(self):
        """Asymmetric qubits use average decay."""
        P = two_qubit_amplitude_damping_bell(λ1=0.2, λ1_other=0.4)
        avg_decay = (0.2 + 0.4) / 2.0
        expected_diag = [1.0 - avg_decay] * 4
        assert np.allclose(np.diag(P), expected_diag)


class TestTwoQubitADCBell:
    """
    Test two-qubit Amplitude + Dephasing Channel (ADC) in Bell basis.
    
    Physics: Combined T1 and T2 on Bell pair. T1 dominates relaxation;
    T2 modulates via multiplicative decay: combined = λ1 + (1-λ1)*λ2.
    See twiling_T1_Jan18.pdf for full density matrix derivation.
    """

    def test_no_noise_is_identity(self):
        """λ1=0, λ2=0 should give identity."""
        P = two_qubit_adc_bell(λ1=0.0, λ2=0.0)
        expected = np.eye(4)
        assert np.allclose(P, expected)

    def test_t1_only_matches_amplitude_damping(self):
        """When λ2=0, ADC should match pure amplitude damping."""
        λ1 = 0.3
        P_adc = two_qubit_adc_bell(λ1=λ1, λ2=0.0)
        P_amp_damp = two_qubit_amplitude_damping_bell(λ1=λ1)
        assert np.allclose(P_adc, P_amp_damp), \
            "ADC with λ2=0 should equal amplitude damping"

    def test_combined_decay_formula(self):
        """
        Test combined decay follows formula:
        combined = λ1 + (1-λ1)*λ2, clamped to [0,1]
        """
        λ1 = 0.3
        λ2 = 0.2
        P = two_qubit_adc_bell(λ1, λ2)

        expected_decay = λ1 + (1.0 - λ1) * λ2
        expected_decay = min(1.0, expected_decay)
        expected_diag = [1.0 - expected_decay] * 4

        assert np.allclose(np.diag(P), expected_diag, atol=1e-10), \
            f"Combined decay formula mismatch: got {np.diag(P)}, expected {expected_diag}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
