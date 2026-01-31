from sympy import latex, Matrix, init_printing
from IPython.display import Math, display
from .derivation import apply_channel
init_printing()
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import os
from .main import _exp_fit
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
    

def plot_markov_results(kept_trajs, pop_history, dt, fit_state=1, tail_fraction=0.25, name="", save_path=None):
    """Combined plotting: population dynamics and excitation decay.
    
    Optimized for Agg backend with:
    - LineCollection for batch trajectory rendering
    - Aggressive downsampling
    - Disabled anti-aliasing
    
    Parameters:
    - save_path: If provided, saves the figure to this path (e.g., "plots/my_plot.png")
    """
    steps = pop_history.shape[0] - 1
    times = np.arange(steps + 1) * dt
    
    # Aggressive downsampling for plotting
    max_plot_points = 1000
    if len(times) > max_plot_points:
        stride = len(times) // max_plot_points
        plot_times = times[::stride]
        plot_pop = pop_history[::stride]
        plot_trajs = kept_trajs[:, ::stride] if kept_trajs.size > 0 else kept_trajs
    else:
        plot_times = times
        plot_pop = pop_history
        plot_trajs = kept_trajs
    
    # Disable anti-aliasing for speed
    # plt.rcParams['lines.antialiased'] = False
    # plt.rcParams['patch.antialiased'] = False
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Use LineCollection for trajectories (MUCH faster than individual lines)
    n_show = min(kept_trajs.shape[0], 10)
    if n_show > 0:
        segments = [np.column_stack([plot_times, (plot_trajs[i] == fit_state).astype(float)]) 
                   for i in range(n_show)]
        lc = LineCollection(segments, colors='gray', alpha=0.3, linewidths=0.5)
        ax1.add_collection(lc)
        ax1.autoscale()
    
    # Plot population dynamics
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#000000"]
    linestyles = ["--", "-", "--", "-", (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (5, 2, 1, 2))]
    
    for s in range(plot_pop.shape[1]):
        ax1.plot(plot_times, plot_pop[:, s], color=colors[s % len(colors)],
                linestyle=linestyles[s % len(linestyles)], linewidth=2, label=f"P({s+1})")
    
    # Fit exponential to selected state population (use full data for accuracy)
    fit = _exp_fit(times, pop_history[:, fit_state - 1], tail_fraction=tail_fraction, verbose=False)
    if fit is not None:
        a, tau, c = fit
        ax1.plot(plot_times, c + a * np.exp(-plot_times / tau), color="blue", linestyle=":", 
                linewidth=2, label=f"τ={tau:.1f} μs")
    
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("time (μs)")
    ax1.set_ylabel("Population")
    ax1.legend(loc="upper right")
    ax1.set_title(name if name else "Markov Monte Carlo")
    ax1.grid(False)
    
    # Use LineCollection for excitation decay trajectories
    n_traj_show = min(plot_trajs.shape[0], 10)
    if n_traj_show > 0:
        segments2 = [np.column_stack([plot_times, plot_trajs[i]]) for i in range(n_traj_show)]
        lc2 = LineCollection(segments2, colors='gray', alpha=0.2, linewidths=0.5)
        ax2.add_collection(lc2)
    
    avg_state = sum((s + 1) * plot_pop[:, s] for s in range(plot_pop.shape[1]))
    ax2.plot(plot_times, avg_state, color="#0072B2", linewidth=2, label="Average state")
    
    fit = _exp_fit(times, sum((s + 1) * pop_history[:, s] for s in range(pop_history.shape[1])), 
                   tail_fraction=tail_fraction, verbose=False)
    if fit is not None:
        a, tau, c = fit
        ax2.plot(plot_times, c + a * np.exp(-plot_times / tau), color="#E69F00", linestyle="--",
                linewidth=2, label=f"τ={tau:.1f} μs")
    
    ax2.set_ylim(0.5, pop_history.shape[1] + 0.5)
    ax2.set_xlabel("time (μs)")
    ax2.set_ylabel("State")
    ax2.set_yticks([1, 2, 3, 4])
    ax2.legend(loc="upper right")
    ax2.set_title("Excitation Decay")
    ax2.grid(False)
    # ax2.autoscale_view()
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
    
    # For Agg backend, display in notebook via IPython
    # display(fig)
    # plt.close(fig)  # Close to free memory
    
    return fig


def plot_mk_analytic_comparison(analytical_pops_history, mk_pops_history,dt,name="",save_path=None):
    # Create a n-vertically stacked plot comparing each population's analytical result to each mk result, in the same style as plot_markov_results
    steps = mk_pops_history.shape[0] - 1
    assert steps == len(analytical_pops_history) - 1, "Analytical and MK population histories must have the same number of steps"
    times = np.arange(steps + 1) * dt
    states = mk_pops_history.shape[1]
    assert states == len(analytical_pops_history[0])
    # "Number of states in analytical and MK results must match"
    
    fig, axs = plt.subplots(states, 1, figsize=(10, 4 * states))
    
    
    # Plot population dynamics
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#000000"]

    linestyles = ["--", "-", "--", "-", (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (5, 2, 1, 2))]
    
    for (i,ax) in enumerate(axs):
        ax.set_xlabel("time (μs)")
        ax.set_ylabel("Population")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(False)
        ax.plot(analytical_pops_history[:, i], color=colors[0], linestyle=linestyles[0], linewidth=2, label=f"Analytical P({i+1})")
        ax.plot(mk_pops_history[:, i], color=colors[1], linestyle=linestyles[1], linewidth=2, label=f"MK P({i+1})")
        ax.legend(loc="best")
    # fig.tight_layout() 
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")
        
    return fig
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
