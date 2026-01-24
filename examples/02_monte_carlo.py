"""
Monte Carlo Validation Example: Theory vs Empirical

Demonstrates comparing theoretical transition probabilities with
empirical distributions from Markov chain sampling.
"""

import sympy as sp
from sympy import sqrt, symbols, Matrix
import mct
import numpy as np


def main():
    """Run Monte Carlo validation example."""
    print("=" * 70)
    print("Monte Carlo Validation: Theoretical vs Empirical")
    print("=" * 70)

    # Step 1: Derive T1 transition matrix
    print("\n1. Derive T1 transition matrix...")
    lam = symbols('lambda', real=True, positive=True)
    K0 = Matrix([[1, 0], [0, sqrt(1 - lam)]])
    K1 = Matrix([[0, sqrt(lam)], [0, 0]])

    P_sym, _ = mct.derive_transition_matrix([K0, K1], mct.computational_basis)
    print(f"   Derived symbolic matrix at λ")

    # Step 2: Evaluate at λ=0.2
    print("\n2. Evaluate at λ=0.2...")
    P = np.array(P_sym.subs(lam, 0.2), dtype=float)
    print(f"   Transition matrix P:\n{P}")

    # Step 3: Theoretical distribution after n steps
    print("\n3. Compute theoretical distribution...")
    n_steps = 5
    theory_dist = mct.theoretical_distribution_after_steps(
        P, initial_state=1, n_steps=n_steps
    )
    print(f"   After {n_steps} steps from |1⟩: {theory_dist}")

    # Step 4: Empirical distribution from trajectories
    print(f"\n4. Simulate {10000:,} trajectories...")
    trajectories = mct.simulate_markov_vectorized(
        P, initial_state=1, n_steps=n_steps, n_trajectories=10000
    )
    empirical_dist = mct.empirical_distribution_from_trajectories(
        trajectories, step=n_steps, n_states=2
    )
    print(f"   Empirical distribution: {empirical_dist}")

    # Step 5: Compare
    print(f"\n5. Compare theory vs empirical:")
    error = np.linalg.norm(theory_dist - empirical_dist)
    print(f"   Theoretical: {theory_dist}")
    print(f"   Empirical:   {empirical_dist}")
    print(f"   L2 error:    {error:.8f}")

    # Step 6: Decay over time
    print(f"\n6. Population decay over time (starting from |1⟩)...")
    n_steps_max = 10
    theory_decay = np.array([
        mct.theoretical_distribution_after_steps(P, initial_state=1, n_steps=n)[1]
        for n in range(n_steps_max + 1)
    ])

    trajectories_long = mct.simulate_markov_vectorized(
        P, initial_state=1, n_steps=n_steps_max, n_trajectories=10000
    )
    empirical_decay = np.array([
        mct.empirical_distribution_from_trajectories(
            trajectories_long, step=n, n_states=2
        )[1]
        for n in range(n_steps_max + 1)
    ])

    print(f"\n   Step | Theoretical | Empirical | Error")
    print(f"   " + "-" * 45)
    for n in range(0, n_steps_max + 1, 2):
        err = abs(theory_decay[n] - empirical_decay[n])
        print(f"   {n:4d} | {theory_decay[n]:.10f} | {empirical_decay[n]:.10f} | {err:.10f}")

    print("\n" + "=" * 70)
    print("✓ Monte Carlo validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
