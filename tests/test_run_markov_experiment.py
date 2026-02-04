
import unittest
import numpy as np
import sympy as sp
from mct import apply_channel, probability_symbol, get_density_basis, apply_channel_basis, run_markov_experiment, transition_matrix_from_sympy

class TestRunMarkovExperiment(unittest.TestCase):
    def test_pop_history_not_empty(self):
        # Setup for 2-qubit amplitude damping
        λ, assumptions = probability_symbol('λ')
        E0 = sp.Matrix([[1, 0], [0, sp.sqrt(1 - λ)]])
        E1 = sp.Matrix([[0, sp.sqrt(λ)], [0, 0]])
        basis_states = get_density_basis(2)
        rho_finals = apply_channel_basis(basis_states, [E0, E1], symbols=[λ], assumptions=assumptions)
        dt = 0.2
        t1 = 100
        lambda_val = 1 - sp.exp(-dt / t1)
        kraus_ops = [E0.subs(λ, lambda_val), E1.subs(λ, lambda_val)]
        trans_matrix = transition_matrix_from_sympy(rho_finals, lambda_val)
        initial_state = 4
        n_trajectories = 1000
        keep_traj = 10
        total_time = 2
        steps = int(total_time / dt)
        kept_trajs, pop_history, asymptotic_pops = run_markov_experiment(
            name="test", initial_state=initial_state, dt=dt, steps=steps,
            n_trajectories=n_trajectories, keep_traj=keep_traj, fit_state=initial_state,
            state_count=4, transition_matrix=trans_matrix, save_path=None,
            compare_to_analytical_kraus=kraus_ops,
            analytical_init_state=basis_states[initial_state-1],
            analytical_pts=steps, analytical_symbols={λ: lambda_val}
        )
        # Test that pop_history is not empty and has expected shape
        self.assertIsNotNone(pop_history)
        self.assertIsInstance(pop_history, np.ndarray)
        self.assertEqual(pop_history.shape[0], steps + 1)  # initial + steps
        self.assertEqual(pop_history.shape[1], 4)
        # Should not be all zeros
        self.assertTrue(np.any(pop_history != 0))
    
    def test_transition_matrix_with_symbol_dict_keys(self):
        # Test transition_matrix_from_sympy with SymPy symbols as dict keys
        p_x, assumptions_x = probability_symbol('p_x')
        p_y, assumptions_y = probability_symbol('p_y')
        p_z, assumptions_z = probability_symbol('p_z')
        
        # Simple test: identity channel (all noise params = 0)
        I = sp.Matrix([[1, 0], [0, 1]])
        X = sp.Matrix([[0, 1], [1, 0]])
        Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
        Z = sp.Matrix([[1, 0], [0, -1]])
        
        p_sigma = 1 - p_x - p_y - p_z
        K0 = sp.sqrt(p_sigma) * I
        K1 = sp.sqrt(p_x) * X
        K2 = sp.sqrt(p_y) * Y
        K3 = sp.sqrt(p_z) * Z
        kraus_ops = [K0, K1, K2, K3]
        
        basis_states = get_density_basis(1)
        assumptions = assumptions_x & assumptions_y & assumptions_z
        rho_finals = apply_channel_basis(basis_states, kraus_ops, symbols=[p_x, p_y, p_z], assumptions=assumptions)
        
        # All zero noise: should be identity
        p_x_val = 0.0
        p_y_val = 0.0
        p_z_val = 0.0
        
        trans_matrix = transition_matrix_from_sympy(rho_finals, {p_x: p_x_val, p_y: p_y_val, p_z: p_z_val})
        
        # Should be 2x2 identity (no change)
        self.assertEqual(trans_matrix.shape, (2, 2))
        np.testing.assert_allclose(trans_matrix, np.eye(2), atol=1e-10)
