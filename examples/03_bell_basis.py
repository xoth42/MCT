"""
Bell Basis Example: Two-Qubit Channels

Demonstrates using predefined Bell basis for two-qubit quantum channels.
"""

import sympy as sp
from sympy import sqrt, symbols, Matrix
import mct
import numpy as np


def main():
    """Run Bell basis example."""
    print("=" * 70)
    print("Bell Basis Example: Two-Qubit Channels")
    print("=" * 70)

    # Step 1: Show available bases
    print("\n1. Available basis states:")
    print(f"   Computational basis (1-qubit): {len(mct.computational_basis)} states")
    print(f"   Bell basis (2-qubit): {len(mct.bell_basis)} states")
    print(f"   States: |00⟩, |10⟩, |01⟩, |11⟩ (or Bell state labels A,B,C,D)")

    # Step 2: Show single-qubit T1
    print("\n2. Single-qubit T1 on computational basis:")
    lam = symbols('lambda', real=True, positive=True)
    K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
    K1 = Matrix([[0, sqrt(lam)], [0, 0]])

    P_comp, meta_comp = mct.derive_transition_matrix(
        [K0, K1], mct.computational_basis, channel_name="T1 (1-qubit)"
    )
    print(f"   Shape: {P_comp.shape}")
    print(f"   Symbols: {meta_comp['symbols']}")

    # Step 3: Numeric evaluation
    print("\n3. Evaluate T1 at λ=0.1 (10% excited state decay per step):")
    P_numeric = np.array(P_comp.subs(lam, 0.1), dtype=float)
    print(f"   P(λ=0.1) =")
    for row in P_numeric:
        print(f"   {row}")

    # Step 4: Bell basis structure
    print("\n4. Bell basis states (2-qubit):")
    for i, state_name in enumerate(['|00⟩ (A)', '|10⟩ (B)', '|01⟩ (C)', '|11⟩ (D)']):
        print(f"   State {i}: {state_name}")

    # Step 5: Properties
    print("\n5. Markov chain properties:")
    mc = mct.MarkovChain(
        P_numeric,
        state_labels=['|0⟩', '|1⟩'],
        name="T1"
    )
    validation = mc.validate_stochasticity()
    print(f"   Valid: {validation['is_valid']}")
    print(f"   Column sums: {validation['column_sums']}")
    print(f"   Min/Max entries: {validation['min_entry']:.4f} / {validation['max_entry']:.4f}")

    # Step 6: Simulate
    print("\n6. Simulate 100 trajectories starting from |1⟩:")
    trajectories = mct.simulate_markov_vectorized(
        P_numeric, initial_state=1, n_steps=5, n_trajectories=100
    )
    print(f"   Trajectory shape: {trajectories.shape}")
    print(f"   Example trajectory (first 6 steps): {trajectories[:6, 0]}")

    print("\n" + "=" * 70)
    print("✓ Bell basis example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
