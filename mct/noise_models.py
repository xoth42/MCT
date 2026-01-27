# """T1/T2 noise models. All times in microseconds (μs)."""

# import math
# from typing import Optional, Tuple

# import numpy as np


# def thermal_relaxation_error_rate(
#     T1: float,
#     T2: float,
#     idle_time: float,
# ) -> Tuple[float, float]:
#     """Compute λ1 = 1 - exp(-t/T1), λ_φ = 0.5(1 - exp(-t·dephasing_rate))."""
#     if T1 <= 0 or T2 <= 0 or idle_time < 0:
#         raise ValueError(f"T1={T1}, T2={T2}, idle_time={idle_time} invalid")

#     if idle_time == 0.0:
#         return 0.0, 0.0

#     λ1 = 1.0 - math.exp(-idle_time / T1)
#     dephasing_rate = (1.0 / T2) - (1.0 / (2.0 * T1))
#     λ_φ = 0.5 * (1.0 - math.exp(-idle_time * dephasing_rate)) if dephasing_rate > 0 else 0.0

#     return max(0.0, min(1.0, λ1)), max(0.0, min(1.0, λ_φ))


# def two_qubit_amplitude_damping_bell(
#     λ1: float,
#     λ1_other: Optional[float] = None,
# ) -> np.ndarray:
#     """
#     Two-qubit amplitude damping channel in Bell basis.

#     Represents the effect of T1 relaxation on a Bell pair in the Bell basis
#     {|Φ+>, |Φ->, |Ψ+>, |Ψ->} ≡ {state 0, 1, 2, 3}.
    
#     When one qubit relaxes (|1> → |0>), Bell entanglement is lost.
#     A Bell state decays with rate λ1 (single-qubit) to computational basis states.
    
#     For simplicity (matching markov_chain_generator patterns), we model:
#     - Each Bell state i has self-decay (stays in computational basis with prob 1-λ1).
#     - Transitions out are uniform to other computational states (if needed).
    
#     This is a simplified 4×4 "Bell-to-computational" transition approximation.
    
#     Parameters
#     ----------
#     λ1 : float
#         Amplitude damping probability per step for qubit 1.
#     λ1_other : float, optional
#         Amplitude damping probability for qubit 2. If None, use same as λ1.
    
#     Returns
#     -------
#     np.ndarray
#         4×4 transition matrix in Bell basis.
#         P[i, j] = Prob(outcome i | initial Bell state j)
#     """
#     if λ1_other is None:
#         λ1_other = λ1

#     # Simplified 4-state Bell decohering model:
#     # Each Bell state has probability λ of decaying (staying in Bell basis with prob 1-λ)
#     # For detailed analysis, see twiling_T1_Jan18.pdf

#     # Placeholder: diagonal decay for each Bell state
#     # (More detailed: would account for different decay rates per Bell state)
#     avg_decay = (λ1 + λ1_other) / 2.0

#     P = np.diag([1.0 - avg_decay] * 4)

#     return P


# def two_qubit_adc_bell(
#     λ1: float,
#     λ2: float,
#     λ1_other: Optional[float] = None,
#     λ2_other: Optional[float] = None,
# ) -> np.ndarray:
#     """
#     Two-qubit Amplitude + Dephasing Channel (ADC) in Bell basis.

#     Combined T1 and T2 (pure dephasing) effects on Bell pair.
#     See twiling_T1_Jan18.pdf for symbolic derivation.
    
#     Parameters
#     ----------
#     λ1 : float
#         T1 (amplitude damping) probability for qubit 1.
#     λ2 : float
#         T2 (dephasing) probability for qubit 1.
#     λ1_other : float, optional
#         T1 for qubit 2. Defaults to λ1.
#     λ2_other : float, optional
#         T2 for qubit 2. Defaults to λ2.
    
#     Returns
#     -------
#     np.ndarray
#         4×4 transition matrix combining T1 and T2 effects.
#     """
#     if λ1_other is None:
#         λ1_other = λ1
#     if λ2_other is None:
#         λ2_other = λ2

#     # Simplified model: independent T1/T2 on each qubit
#     # More detailed: see reference PDF for full density matrix calculation

#     # For now, combine via product (multiplicative dephasing and relaxation effects)
#     avg_λ1 = (λ1 + λ1_other) / 2.0
#     avg_λ2 = (λ2 + λ2_other) / 2.0

#     # T1 dominates; T2 modulates
#     combined_decay = avg_λ1 + (1.0 - avg_λ1) * avg_λ2
#     combined_decay = min(1.0, combined_decay)  # clamp

#     P = np.diag([1.0 - combined_decay] * 4)

#     return P


# def single_qubit_t1_markov(λ: float) -> np.ndarray:
#     """
#     Single-qubit T1 (amplitude damping) as 2-state Markov chain.
    
#     States: {|0>, |1>}
#     |1> relaxes to |0> with probability λ per step.
#     |0> is ground state (absorbing).
    
#     Parameters
#     ----------
#     λ : float
#         Amplitude damping probability (0 to 1).
    
#     Returns
#     -------
#     np.ndarray
#         2×2 transition matrix.
#         P[i, j] = Prob(i | j) where i, j ∈ {0, 1}.
#     """
#     λ = max(0.0, min(1.0, λ))

#     P = np.array([
#         [1.0, λ],        # Prob(stay in |0>, or relax to |0> from |1>)
#         [0.0, 1.0 - λ]   # Prob(no transition) or (stay in |1>)
#     ])

#     return P


# def single_qubit_t2_markov(λ_φ: float) -> np.ndarray:
#     """
#     Single-qubit T2 (pure dephasing) as 2×2 dephasing map.
    
#     Simplified: phase damping (dephasing without relaxation).
#     States: {|0>, |1>} (computational basis)
#     Dephasing reduces coherence in superpositions.
    
#     Parameters
#     ----------
#     λ_φ : float
#         Pure dephasing probability (0 to 1).
    
#     Returns
#     -------
#     np.ndarray
#         2×2 transition matrix for dephasing.
#     """
#     λ_φ = max(0.0, min(1.0, λ_φ))

#     # Phase damping: diagonalizes the density matrix
#     # No population change, only dephasing of coherences
#     # As a Markov chain on computational basis: diagonal (populations unchanged)
#     P = np.eye(2)

#     return P


# def single_qubit_combined_t1t2(λ1: float, λ_φ: float) -> np.ndarray:
#     """
#     Single-qubit combined T1 + T2 effect.
    
#     Combines amplitude damping (T1) and dephasing (T2).
#     Simplified approach: apply relaxation first, then account for dephasing.
    
#     Parameters
#     ----------
#     λ1 : float
#         T1 amplitude damping probability.
#     λ_φ : float
#         T2 pure dephasing probability (not typically used in computational basis directly).
    
#     Returns
#     -------
#     np.ndarray
#         2×2 transition matrix combining both effects.
#     """
#     λ1 = max(0.0, min(1.0, λ1))
#     λ_φ = max(0.0, min(1.0, λ_φ))

#     # Dominant effect is T1; T2 modulates
#     P = np.array([
#         [1.0, λ1],
#         [0.0, 1.0 - λ1]
#     ])

#     return P


# if __name__ == "__main__":
#     # Example: compute error rates for typical transmon parameters
#     T1 = 100.0  # 100 μs
#     T2 = 150.0  # 150 μs
#     idle_time = 1.0  # 1 μs

#     λ1, λ_φ = thermal_relaxation_error_rate(T1, T2, idle_time)
#     print(f"T1={T1} μs, T2={T2} μs, idle_time={idle_time} μs")
#     print(f"λ1 (amplitude damping): {λ1:.6f}")
#     print(f"λ_φ (pure dephasing):   {λ_φ:.6f}")

#     # Example: single-qubit T1 chain
#     P = single_qubit_t1_markov(λ1)
#     print(f"\nSingle-qubit T1 Markov chain:\n{P}")
