"""
Quickstart Example: Define → Derive → Generate

Shows the core MCT workflow:
1. Define Kraus operators with symbolic parameters
2. Derive analytic transition matrix
3. Generate executable sampling code
"""

import sympy as sp
import mct
import os


def main():
    """Run quickstart example."""
    print("=" * 70)
    print("MCT Quickstart: T1 Amplitude Damping")
    print("=" * 70)

    # Step 1: Define Kraus operators with symbols
    print("\n1. Define Kraus operators...")
    lam = sp.symbols('lambda', real=True, positive=True)
    K0 = sp.Matrix([[1, 0], [0, sp.sqrt(1 - lam)]])
    K1 = sp.Matrix([[0, sp.sqrt(lam)], [0, 0]])
    print(f"   E0 = {K0.T}")
    print(f"   E1 = {K1.T}")

    # Step 2: Get analytic transition matrix
    print("\n2. Derive symbolic transition matrix...")
    basis = mct.computational_basis  # [|0⟩, |1⟩]
    analytical_result = mct.superoperator(
        (K0, K1),  # Kraus operators
        basis[0],  # |0⟩ state
        basis=basis,
        qubits=1
    )
    P_sym, metadata = analytical_result
    print(f"   Matrix shape: {P_sym.shape}")
    print(f"   Symbols: {metadata['symbols']}")

    # Step 3: Display LaTeX representation
    print("\n3. Symbolic transition matrix P(λ):")
    mct.print_matrix_latex(P_sym, "P(\\lambda)")

    # Step 4: Verify stochasticity at parameter values
    print("\n4. Verify stochasticity at λ = 0.1, 0.5, 0.9...")
    for lam_val in [0.1, 0.5, 0.9]:
        P_num, is_valid = mct.substitute_and_verify(
            P_sym, lam, lam_val, name=f"   λ={lam_val}"
        )
        assert is_valid, f"Stochasticity failed at λ={lam_val}"

    # Step 5: Generate sampling code
    print("\n5. Generate executable sampling code at λ=0.3...")
    os.makedirs("generated", exist_ok=True)
    code = mct.markov_chain(
        analytical_result,
        to_file="generated/t1_sampling.py",
        parameter_values={'lambda': 0.3}
    )
    print(f"   Generated {len(code)} characters")
    print(f"   Saved to: generated/t1_sampling.py")
    print(f"   First 200 chars:\n{code[:200]}...")

    print("\n" + "=" * 70)
    print("✓ Quickstart complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
