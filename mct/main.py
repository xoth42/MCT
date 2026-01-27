# """Core MCT API: symbolic derivation and high-level sampling."""

# from typing import List, Optional, Tuple

# import numpy as np
# from sympy import Matrix

# def derive_transition_matrix(kraus_ops: List[Matrix], basis_states: List[Matrix],
#                              channel_name: str = "") -> Tuple[Matrix, dict]:
#     """Derive transition matrix P from Kraus operators and basis states.
    
#     P[i,j] = Prob(state i | state j), extracted as diagonal of output density matrices.
#     """
#     from sympy import simplify

#     n_states = len(basis_states)

#     # Apply channel to each basis state; extract diagonals
#     output_densities = []
#     for rho in basis_states:
#         rho_out = apply_channel(kraus_ops, rho)
#         output_densities.append(rho_out)

#     # Build transition matrix: P[i, j] = output_densities[j][i, i]
#     P = Matrix(n_states, n_states,
#                lambda i, j: simplify(output_densities[j][i, i], rational=True))

#     metadata = {
#         'channel_name': channel_name,
#         'n_states': n_states,
#         'kraus_operators': kraus_ops,
#         'basis_states': basis_states,
#         'output_densities': output_densities,
#         'symbols': P.free_symbols,
#     }

#     return P, metadata


# def substitute_and_verify(P_symbolic, lam_symbol, lam_value, name=""):
#     """Substitute parameter value into symbolic P, verify stochasticity."""
#     P_numeric = P_symbolic.subs(lam_symbol, lam_value)
#     P_np = np.array(P_numeric, dtype=float)

#     # Check stochasticity
#     col_sums = np.sum(P_np, axis=0)
#     is_stochastic = np.allclose(col_sums, 1.0, atol=1e-10)
#     entries_valid = np.all((P_np >= -1e-10) & (P_np <= 1.0 + 1e-10))
#     is_valid = is_stochastic and entries_valid

#     if name:
#         print(f"{name}")
#         print(f"  λ = {lam_value}")
#         print(f"  P_numeric =\n{P_np}")
#         print(f"  Column sums: {col_sums}")
#         print(f"  ✓ Valid: {is_valid}\n")

#     return P_np, is_valid


# # ============================================================================
# # High-Level API
# # ============================================================================

# def superoperator(
#     kraus_ops: Tuple[Matrix, ...],
#     basis_state: Matrix,
#     basis: List[Matrix],
#     qubits: int,
# ) -> Tuple[Matrix, dict]:
#     """Apply quantum channel to basis state; extract transition matrix."""
#     n_basis = len(basis)
#     P_matrix = []

#     for j in range(n_basis):
#         rho_j = basis[j]
#         rho_j_out = apply_channel(list(kraus_ops), rho_j)

#         P_column = []
#         for i in range(n_basis):
#             p_ij = rho_j_out[i, i]
#             P_column.append(p_ij)

#         P_matrix.append(P_column)

#     P = Matrix(P_matrix).T

#     metadata = {
#         'kraus_operators': kraus_ops,
#         'basis_state': basis_state,
#         'basis': basis,
#         'basis_size': n_basis,
#         'qubits': qubits,
#         'symbols': P.free_symbols,
#     }

#     return P, metadata


# def markov_chain(
#     analytical_result: Tuple[Matrix, dict],
#     to_file: Optional[str] = None,
#     parameter_values: Optional[dict] = None,
# ) -> str:
#     """Generate sampling code from a symbolic transition matrix.
    
#     Takes the result of superoperator() and generates executable Python code
#     for fast Markov chain sampling.
#     """
#     from .markov_chain import generate_sampling_function

#     P_symbolic, metadata = analytical_result

#     # Substitute parameters if provided
#     P = P_symbolic
#     if parameter_values:
#         for symbol_name, value in parameter_values.items():
#             for sym in metadata.get('symbols', []):
#                 if str(sym) == symbol_name:
#                     P = P.subs(sym, value)
#                     break

#     # Convert to numeric for code generation
#     # First try with remaining free symbols, then with evalf
#     remaining_symbols = P.free_symbols
#     if remaining_symbols:
#         # Still has unsubstituted symbols - try to evalf with dummy values
#         subs_dict = {sym: 0.5 for sym in remaining_symbols}
#         P = P.subs(subs_dict)

#     try:
#         P_numeric = np.array(P.evalf(), dtype=float)
#     except (TypeError, AttributeError, ValueError):
#         P_numeric = np.array(P, dtype=float)

#     # Generate sampling function
#     code = generate_sampling_function(P_numeric)

#     # Write to file if requested
#     if to_file:
#         import os
#         os.makedirs(os.path.dirname(to_file) or '.', exist_ok=True)
#         with open(to_file, 'w') as f:
#             f.write(code)
#         print(f"Generated sampling code written to: {to_file}")

#     return code


