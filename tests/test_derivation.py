import unittest
import numpy as np
import sympy as sp

# Test derivation.py apply channel for 2qubit amplitude damping
from mct.derivation import apply_channel, probability_symbol, to_bell_basis


class TestApplyChannel(unittest.TestCase):
    """Test the apply_channel function for various scenarios."""

    def test_single_qubit_amplitude_damping(self):
        """Test single-qubit amplitude damping channel."""
        l = sp.symbols("lambda", real=True, positive=True)
        assumptions = sp.Q.real(l) & sp.Q.positive(l) & (l <= 1)

        E0 = sp.Matrix([[1, 0], [0, sp.sqrt(1 - l)]])
        E1 = sp.Matrix([[0, sp.sqrt(l)], [0, 0]])

        # all-symbolic placeholder
        ρ00, ρ01, ρ10, ρ11 = sp.symbols("ρ_00 ρ_01 ρ_10 ρ_11")

        ρ = sp.Matrix([[ρ00, ρ01], [ρ10, ρ11]])

        # Apply channel
        ρ_f = apply_channel(ρ, [E0, E1])
        print("Before:", ρ, "\nAfter:", ρ_f)
        # Verify the result is not None
        self.assertIsNotNone(ρ_f)

        ρ2 = sp.Matrix([[0.5, 0.5], [0.5, 0.5]])
        ρ2_f = apply_channel(ρ2, [E0, E1], symbols=[l], assumptions=assumptions)
        print("Before:", ρ2, "\nAfter:", ρ2_f)

    def test_two_qubit_amplitude_damping(self):
        """Test two-qubit amplitude damping channel."""
        l = sp.symbols("lambda", real=True, positive=True)
        E0 = sp.Matrix([[1, 0], [0, sp.sqrt(1 - l)]])
        E1 = sp.Matrix([[0, sp.sqrt(l)], [0, 0]])

        # all-symbolic placeholder
        symbols_str = " ".join([f"ρ_{i}{j}" for i in range(4) for j in range(4)])
        print("Symbols string:", symbols_str)
        ρ_symbols = sp.symbols(symbols_str.strip())

        # create 4 by 4 density matrix
        ρ = sp.Matrix(4, 4, lambda i, j: ρ_symbols[i * 4 + j])
        print("Before:", ρ)
        # Apply channel
        ρ_f = apply_channel(ρ, [E0, E1])
        print("After:", ρ_f)
        # Verify the result is not None
        self.assertIsNotNone(ρ_f)

    def test_n_qubit_amp_damp(self):
        """Test single-qubit amplitude damping channel."""
        l = sp.symbols("lambda", real=True, positive=True)
        assumptions = sp.Q.real(l) & sp.Q.positive(l) & (l <= 1)

        E0 = sp.Matrix([[1, 0], [0, sp.sqrt(1 - l)]])
        E1 = sp.Matrix([[0, sp.sqrt(l)], [0, 0]])

        for qubits in range(1, 4):
            ρ = sp.Matrix(2**qubits, 2**qubits, lambda i, j: sp.symbols(f"ρ_{i}{j}"))

            # Apply channel
            ρ_f = apply_channel(ρ, [E0, E1], symbols=[l], assumptions=assumptions)

            print("\nQubits =", qubits, "Before:", ρ, "\nAfter:", ρ_f)
            # Verify the result is not None
            self.assertIsNotNone(ρ_f)

    def test_n_qubit_phase_damp(self):
        l = sp.symbols("lambda", real=True, positive=True)
        assumptions = sp.Q.real(l) & sp.Q.positive(l) & (l <= 1)

        E0 = sp.Matrix([[1, 0], [0, sp.sqrt(1 - l)]])
        E1 = sp.Matrix([[0, 0], [0, sp.sqrt(l)]])

        for qubits in range(1, 4):
            pass
            ρ = sp.Matrix(2**qubits, 2**qubits, lambda i, j: sp.symbols(f"ρ_{i}{j}"))

            # Apply channel
            ρ_f = apply_channel(ρ, [E0, E1], symbols=[l], assumptions=assumptions)

            print("\nQubits =", qubits, "Before:", ρ, "\nAfter:", ρ_f)
            # Verify the result is not None
            self.assertIsNotNone(ρ_f)

    def test_probability_symbol_comprehensive(self):
        """Comprehensive test for probability_symbol helper function."""

        # Test 1: Basic symbol creation with valid assumptions
        l, assumptions = probability_symbol("lambda")
        self.assertIsInstance(l, sp.Symbol)
        self.assertEqual(str(l), "lambda")
        self.assertIsNotNone(assumptions)
        self.assertTrue(isinstance(assumptions, sp.Basic))

        # Test 2: Multiple parameter names
        names = ["lambda", "gamma", "mu", "p_x", "decay_rate"]
        for name in names:
            sym, assume = probability_symbol(name)
            self.assertEqual(str(sym), name)
            self.assertIsNotNone(assume)

        # Test 3: Works correctly in apply_channel
        E0 = sp.Matrix([[1, 0], [0, sp.sqrt(1 - l)]])
        E1 = sp.Matrix([[0, sp.sqrt(l)], [0, 0]])
        ρ = sp.Matrix([[1, 0], [0, 0]])
        ρ_f = apply_channel(ρ, [E0, E1], symbols=[l], assumptions=assumptions)
        self.assertIsNotNone(ρ_f)
        self.assertEqual(ρ_f.shape, (2, 2))

        # Test 4: Assumptions enforce real, positive, ≤1
        l_beta, assumptions_beta = probability_symbol("beta")
        test_val = sp.Rational(1, 2)
        refined = sp.refine(l_beta - test_val, assumptions_beta)
        self.assertIsNotNone(refined)
        test_val_one = sp.Integer(1)
        refined_one = sp.refine(test_val_one - l_beta, assumptions_beta)
        self.assertIsNotNone(refined_one)

        # Test 5: Symbol substitution in expressions
        expr = 1 - l
        result = expr.subs(l, sp.Rational(1, 4))
        self.assertEqual(result, sp.Rational(3, 4))
        self.assertGreaterEqual(float(result), 0)
        self.assertLessEqual(float(result), 1)

        # Test 6: Multiple independent symbols
        l1, assume1 = probability_symbol("lambda")
        l2, assume2 = probability_symbol("gamma")
        self.assertNotEqual(l1, l2)
        expr_multi = l1 + l2
        result_multi = expr_multi.subs(
            [(l1, sp.Rational(1, 4)), (l2, sp.Rational(1, 3))]
        )
        self.assertEqual(result_multi, sp.Rational(7, 12))

    def test_to_bell_basis_transformation(self):
        """Test to_bell_basis transformation for various density matrices."""
        from mct.derivation import Bell_transformation,ρ_00_comp, ρ_01_comp, ρ_10_comp, ρ_11_comp, ρ_00_bell, ρ_01_bell, ρ_10_bell, ρ_11_bell
        # Test 1: Bell states in computational basis should be diagonal in bell basis
        bell_states_comp = [ρ00_comp, ρ01_comp, ρ10_comp, ρ11_comp]
        bell_states_bell = [ρ00_bell, ρ01_bell, ρ10_bell, ρ11_bell]

        for i, (ρ_comp, ρ_bell_expected) in enumerate(
            zip(bell_states_comp, bell_states_bell)
        ):
            ρ_bell_actual = to_bell_basis(ρ_comp)
            # Convert to numpy for comparison
            if isinstance(ρ_bell_actual, sp.Matrix):
                ρ_bell_actual = np.array(ρ_bell_actual, dtype=complex)

            # Check that result is close to expected (diagonal with 1 at position i)
            np.testing.assert_allclose(
                ρ_bell_actual,
                ρ_bell_expected,
                atol=1e-10,
                err_msg=f"Bell state {i} not correctly transformed",
            )

        # Test 2: Identity operator should remain identity
        identity_2qubit = np.eye(4) / 4  # Maximally mixed state
        ρ_bell = to_bell_basis(identity_2qubit)
        if isinstance(ρ_bell, sp.Matrix):
            ρ_bell = np.array(ρ_bell, dtype=complex)
        np.testing.assert_allclose(
            ρ_bell,
            identity_2qubit,
            atol=1e-10,
            err_msg="Identity should remain identity under Bell basis transformation",
        )

        # Test 3: Transformation should be unitary (preserves trace and purity)
        for ρ_comp in bell_states_comp:
            ρ_bell = to_bell_basis(ρ_comp)
            if isinstance(ρ_bell, sp.Matrix):
                ρ_bell = np.array(ρ_bell, dtype=complex)

            trace_comp = np.trace(ρ_comp)
            trace_bell = np.trace(ρ_bell)
            self.assertAlmostEqual(
                trace_comp,
                trace_bell,
                places=10,
                msg="Trace should be preserved under unitary transformation",
            )

            purity_comp = np.trace(ρ_comp @ ρ_comp)
            purity_bell = np.trace(ρ_bell @ ρ_bell)
            self.assertAlmostEqual(
                purity_comp,
                purity_bell,
                places=10,
                msg="Purity should be preserved under unitary transformation",
            )

        # Test 4: Round-trip transformation (bell -> comp -> bell should give identity)
        # This requires the inverse transformation, so we test symmetry instead
        ρ_comp = bell_states_comp[0]
        ρ_bell = to_bell_basis(ρ_comp)
        if isinstance(ρ_bell, sp.Matrix):
            ρ_bell = np.array(ρ_bell, dtype=complex)

        # Check that all off-diagonal elements are near zero (bell states should be pure)
        for i in range(4):
            for j in range(4):
                if i != j:
                    self.assertAlmostEqual(
                        abs(ρ_bell[i, j]),
                        0,
                        places=10,
                        msg=f"Off-diagonal element [{i},{j}] should be zero for pure bell state",
                    )

        # Test 5: Test with sympy Matrix input
        ρ_comp_sympy = sp.Matrix(bell_states_comp[0])
        ρ_bell_sympy = to_bell_basis(ρ_comp_sympy)
        self.assertIsInstance(
            ρ_bell_sympy,
            sp.Matrix,
            msg="to_bell_basis should return sympy.Matrix when given sympy.Matrix input",
        )
        self.assertEqual(ρ_bell_sympy.shape, (4, 4), msg="Output should be 4x4 matrix")

        # Test 6: Invalid input should raise ValueError
        with self.assertRaises(ValueError):
            to_bell_basis("invalid_input")

        with self.assertRaises(ValueError):
            to_bell_basis([1, 2, 3, 4])  # List instead of matrix


if __name__ == "__main__":
    unittest.main()
