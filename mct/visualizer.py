from sympy import latex, Matrix, init_printing
from IPython.display import Math, display
from .derivation import apply_channel
init_printing()

"""Pretty latex matrix result"""
def lmatrix(matrix):
    # Handle list or array of matrices
    if isinstance(matrix, (list, tuple)):
        return Math(''.join([latex(m) + r'\\ ' for m in matrix]))
    else:
        return Math(latex(Matrix(matrix)))

"""Apply channel to given basis, show results if show=True, default"""
def apply_channel_basis(basis,ops,symbols=None,assumptions=None,show=True):
    results = []
    i = 0
    for state in basis:
        show and print(f"State {i}")
        res = apply_channel(state, ops, symbols=symbols, assumptions=assumptions)
        show and display(lmatrix(res))
        results.append(res)
        i += 1
    
    return results
    

# """Population decay plots and LaTeX matrix formatting."""

# import numpy as np

# try:
#     import matplotlib.pyplot as plt
#     HAS_MATPLOTLIB = True
# except ImportError:
#     HAS_MATPLOTLIB = False


# def print_matrix_latex(matrix, name="Matrix"):
#     """Print SymPy matrix in LaTeX format."""
#     try:
#         from IPython.display import Markdown, display
#         latex_str = f"${name} = {matrix._repr_latex_()[1:-1]}$"
#         display(Markdown(latex_str))
#     except ImportError:
#         print(f"{name} =\n{matrix}")


# def plot_populations(P, n_steps=10, initial_state=1, title="Population Evolution"):
#     """Plot population decay from initial state over n steps."""
#     if not HAS_MATPLOTLIB:
#         raise ImportError("matplotlib required for plot_populations(); install with: pip install matplotlib")

#     n_states = P.shape[0]

#     # Compute theoretical distributions
#     initial_dist = np.zeros(n_states)
#     initial_dist[initial_state] = 1.0

#     populations = [initial_dist.copy()]
#     dist = initial_dist.copy()
#     for _ in range(n_steps):
#         dist = P @ dist
#         populations.append(dist.copy())

#     populations = np.array(populations)

#     # Plot
#     fig, ax = plt.subplots(figsize=(8, 4))
#     for state in range(n_states):
#         ax.plot(populations[:, state], marker='o', label=f'State {state}')
#     ax.set_xlabel('Step')
#     ax.set_ylabel('Population')
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     return fig, ax
