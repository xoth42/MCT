import numpy as np
# import pandas as pd
import sympy as sp
from itertools import product
# from main import transition_matrix_from_sympy

def density_matrix(state_vector):
    return np.dot(state_vector, state_vector.conj().T)

# Bell states in the computational basis (ket)
bell_00_comp = np.array([[1], [0], [0], [1]]) / np.sqrt(2)  # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
bell_01_comp = np.array([[1], [0], [0], [-1]]) / np.sqrt(2)  # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
bell_10_comp = np.array([[0], [1], [1], [0]]) / np.sqrt(2)  # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
bell_11_comp = np.array([[0], [1], [-1], [0]]) / np.sqrt(2)  # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2

# The density matrices of the bell basis (in computational basis)
ρ00_comp = density_matrix(bell_00_comp)
ρ01_comp = density_matrix(bell_01_comp)
ρ10_comp = density_matrix(bell_10_comp)
ρ11_comp = density_matrix(bell_11_comp)

# Bell states in bell basis
bell_00_bell = np.array([[1], [0], [0], [0]])
bell_01_bell = np.array([[0], [1], [0], [0]]) 
bell_10_bell = np.array([[0], [0], [1], [0]]) 
bell_11_bell = np.array([[0], [0], [0], [1]]) 

# define transformation matrix from computational to bell basis
# Bell -> Computational 
Bell_transformation = np.hstack((bell_00_comp, bell_01_comp, bell_10_comp, bell_11_comp))

"""Generate n-qubit density operators basis. Supported options:
- comp states, comp basis
- bell states, comp basis (only 2 qubits)
- bell states, bell basis"""
def get_density_basis(qubits, states='computational', basis='computational'):
    """Generate n-qubit density operators basis."""
    assert qubits >= 1, "Number of qubits must be at least 1"
    basis_  = basis.lower() 
    states_ = states.lower()   
    states = []
    comp_basis = basis_ == 'computational' or basis_ == 'comp'
    bell_basis = basis_ == 'bell'
    comp_states = states_ == 'computational' or states_ == 'comp'
    bell_states = states_ == 'bell'
    
    # Normal computational case (bell states in bell basis become comp basis)
    if (bell_basis and bell_states) or (comp_basis and comp_states):
        for i in range(2**qubits):
            state = np.zeros((2**qubits, 1))
            state[i, 0] = 1
            states.append(density_matrix(state))
        return states
    elif bell_basis and comp_states:
        raise NotImplementedError("Bell basis density matrices from computational states not implemented yet")
    elif comp_basis and bell_states:
        if qubits != 2:
            raise NotImplementedError("Only 2-qubit Bell states implemented")
        states = [ρ00_comp, ρ01_comp, ρ10_comp, ρ11_comp]
        return states        
    else:
        raise ValueError(f"Unsupported basis ({basis}) and states ({states}) combination, {qubits} qubits")

"""
Apply n-qubit channel given single qubit Kraus operators [E0, E1...Ek] 
Infer the amount of qubits from the initial ρ.
kraus operators are sympy matricies that may contain symbols. If symbols and assumptions are provided, we will try to simplify with them.
Example assumptions: assumptions = sp.Q.real(l) & sp.Q.positive(l) & (l <= 1)"""
def apply_channel(ρ, kraus_ops, symbols=None, assumptions=None):
    # Sanity checks
    assert len(kraus_ops) > 0, "At least one Kraus operator required"
    for array in [ρ, *kraus_ops]:
        assert isinstance(array, (np.ndarray, sp.Matrix)), "ρ must be a numpy array or sympy Matrix"

    qubits = int(np.log2(ρ.shape[0]))
    if qubits != int(qubits) or qubits < 1:
        raise ValueError("ρ must be a square density operator with dimension 2^n")

    # Basis transformed kraus ops (n-qubit ops instead of single qubit ops)
    # 2qubit example:
    # K_ij = Ei ⊗ Ej 
    #  Kn = [
    #     sp.kronecker_product() for Ei in kraus_ops for Ej in kraus_ops
    # ]

    # N qubit example:
    # K_n = E_a0 ⊗ E_a1 ⊗ ... ⊗ E_an
    Kn = []
    
    # Generate all n-qubit tensor products
    # For n qubits and k single-qubit Kraus ops, we have k^n combinations
    for op_indices in product(range(len(kraus_ops)), repeat=qubits):
        # Build tensor product: E_i0 ⊗ E_i1 ⊗ ... ⊗ E_in
        combined_op = kraus_ops[op_indices[0]]
        for idx in op_indices[1:]:
            combined_op = sp.kronecker_product(combined_op, kraus_ops[idx])
        Kn.append(combined_op)
    
    assert len(Kn) == len(kraus_ops) ** qubits, "Number of n-qubit Kraus operators incorrect"
    assert all(
        Ki.shape == (2**qubits, 2**qubits) for Ki in Kn
    ), "Each n-qubit Kraus operator must have correct dimensions"

    if isinstance(ρ, np.ndarray):
        ρ = sp.Matrix(ρ)
    
    # ρ_f = Σ_i K_i ρ K_i†
    ρ_applied = sum((Ki @ ρ @ Ki.H for Ki in Kn), sp.zeros(ρ.shape[0], ρ.shape[1]))

    # Simplify if symbols provided
    if symbols is not None and assumptions is not None:
        ρ_applied = sp.simplify(sp.refine(ρ_applied, assumptions))
        # ρ_applied = ρ_applied.applyfunc(lambda x: )

        substitutions = []
        for sym in symbols:
            # Is this symbol real-positive and <= 1?
            if sp.Q.real(sym) & sp.Q.positive(sym) & (sym <= 1):
                # Set up common substitutions
                substitutions.append((sp.conjugate(sp.sqrt(1 - sym)), sp.sqrt(1 - sym)))
                substitutions.append((sp.conjugate(sp.sqrt(2 - sym)), sp.sqrt(2 - sym)))
        
        def sub(x):
            for sub in substitutions:
                x = x.subs(sub[0], sub[1])
            return x

        ρ_applied = sp.simplify(
            ρ_applied.applyfunc(sub) # substitute
        ) 
        

    return ρ_applied

"""Helper to create probability symbols for kraus ops"""
def probability_symbol(name):
    l = sp.symbols(name, real=True, positive=True)
    assumptions = sp.Q.real(l) & sp.Q.positive(l) & (l <= 1)
    return l, assumptions  

def to_bell_basis(ρ_comp):
    """Transform density matrix from computational basis to bell basis"""
    if isinstance(ρ_comp, sp.Matrix):
        return Bell_transformation.T.conj() @ ρ_comp @ Bell_transformation
    elif isinstance(ρ_comp, np.ndarray):
        return Bell_transformation.conj().T * ρ_comp * Bell_transformation
    else:
        raise ValueError("ρ_comp must be a numpy array or sympy Matrix")
    
def density_operator_simulator(kraus_ops, initial_state, symbols_values, steps = 100):
    # Substitute symbols in Kraus ops
    substituted_ops = [op.subs(symbols_values) for op in kraus_ops]

    # Determine system size
    dim = initial_state.shape[0]
    n_qubits = int(np.log2(dim))
    # If ops are 2x2, expand to n-qubit via tensor product
    if substituted_ops[0].shape == (2, 2) and n_qubits > 1:
        # Build all tensor products for n qubits
        from itertools import product
        single_qubit_ops = substituted_ops
        substituted_ops = []
        for idxs in product(range(len(single_qubit_ops)), repeat=n_qubits):
            op = single_qubit_ops[idxs[0]]
            for i in idxs[1:]:
                op = sp.Matrix(np.kron(np.array(op).astype(np.complex128), np.array(single_qubit_ops[i]).astype(np.complex128)))
            substituted_ops.append(op)

    def step(ρ):
        ρ_next = sp.zeros(ρ.shape[0], ρ.shape[1])
        for K in substituted_ops:
            ρ_next += K @ ρ @ K.H
        return ρ_next

    ρ = initial_state
    history = [ρ]
    for _ in range(steps):
        ρ = step(ρ)
        history.append(ρ)
    return history
