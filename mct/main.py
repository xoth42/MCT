import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from scipy.optimize import curve_fit
import time
import os
from matplotlib.collections import LineCollection
from IPython.display import display
import sympy as sp



# Optional numba import with fallback
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorators if numba not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args, **kwargs) 

def load_markov_function(module_path, function_name):
    """Dynamically load a Markov chain function from a Python module."""
    spec = importlib.util.spec_from_file_location("markov_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

def compute_analytical_transition_matrix(density_matrices, state_count=None):
    """
    Compute transition matrix analytically from density matrices.
    
    Parameters:
    - density_matrices: dict mapping state -> list of (density_matrix, measurement_result) tuples.
                        Each density matrix should be a 2D numpy array (or convertible to one).
                        measurement_result is the resulting state (1-indexed).
    - state_count: Number of states. If None, inferred from max state in density_matrices.
    
    Returns:
    - trans_matrix: (state_count, state_count) numpy array where trans_matrix[i,j] is
                    the probability of transitioning from state i+1 to state j+1.
    
    Example usage:
        density_matrices = {
            1: [(rho_1_to_1, 1), (rho_1_to_2, 2)],  # From state 1
            2: [(rho_2_to_1, 1), (rho_2_to_2, 2)],  # From state 2
        }
        trans = compute_analytical_transition_matrix(density_matrices)
    """
    if state_count is None:
        state_count = max(density_matrices.keys())
    
    trans_matrix = np.zeros((state_count, state_count), dtype=np.float64)
    
    for from_state, outcomes in density_matrices.items():
        from_idx = from_state - 1  # Convert to 0-indexed
        for density_matrix, to_state in outcomes:
            to_idx = to_state - 1
            # Probability is the trace of the density matrix
            dm = np.asarray(density_matrix, dtype=np.complex128)
            prob = np.real(np.trace(dm))
            trans_matrix[from_idx, to_idx] += prob
        
        # Normalize row (should already sum to 1, but ensure numerical stability)
        row_sum = trans_matrix[from_idx].sum()
        if row_sum > 0:
            trans_matrix[from_idx] /= row_sum
    
    return trans_matrix

def transition_matrix_from_sympy(density_matrices_list, lambda_val, state_count=None):
    """
    Compute numerical transition matrix from list of SymPy density matrices.
    
    Parameters:
    - density_matrices_list: List of SymPy Matrix objects, one per state.
                            Each matrix is the density matrix after the channel.
                            Diagonal elements give measurement probabilities.
    - lambda_val: Either:
                  - A single numeric value (substitutes ALL symbols with this value)
                  - A dict mapping symbol names to values, e.g.:
                    {'lambda': 0.01} or {'lambda_1': 0.01, 'lambda_2': 0.02}
    - state_count: Number of states. If None, inferred from list length.
    
    Returns:
    - trans_matrix: (state_count, state_count) numpy array
    
    The transition probability from state i to state j is the j-th diagonal
    element of the i-th density matrix.
    
    Example usage:
        # Single lambda (amplitude damping):
        trans = transition_matrix_from_sympy(matrices, 0.01)
        
        # Multiple lambdas (ADC channel):
        trans = transition_matrix_from_sympy(matrices, {'lambda_1': 0.01, 'lambda_2': 0.02})
    """
    from sympy import N
    
    if state_count is None:
        state_count = len(density_matrices_list)
    
    # Normalize lambda_val to a dict
    if isinstance(lambda_val, dict):
        lambda_dict = lambda_val
        # Check if keys are SymPy symbols or strings
        first_key = next(iter(lambda_dict.keys()))
        from sympy import Symbol
        keys_are_symbols = isinstance(first_key, Symbol)
    else:
        # Single value - will substitute all free symbols
        lambda_dict = None
        keys_are_symbols = False
        single_val = lambda_val
    
    trans_matrix = np.zeros((state_count, state_count), dtype=np.float64)
    
    for i, dm in enumerate(density_matrices_list):
        for j in range(state_count):
            # Get diagonal element
            elem = dm[j, j]
            
            # Substitute symbols
            if lambda_dict is not None:
                if keys_are_symbols:
                    # Dict with SymPy symbol keys: {p_x: value, p_y: value}
                    for sym, val in lambda_dict.items():
                        elem = elem.subs(sym, val)
                else:
                    # Dict with string keys: {'lambda_1': value, 'lambda_2': value}
                    for sym in elem.free_symbols:
                        sym_name = str(sym)
                        if sym_name in lambda_dict:
                            elem = elem.subs(sym, lambda_dict[sym_name])
            else:
                # Single value - substitute all free symbols
                for sym in elem.free_symbols:
                    elem = elem.subs(sym, single_val)
            
            prob = float(N(elem))
            trans_matrix[i, j] = prob
    
    return trans_matrix

# =============================================================================
# NUMBA JIT-Compiled Monte Carlo Simulation 
# =============================================================================

# Number of samples to estimate transition matrix.
N_SAMPLES_DEFAULT = 10000000

# -----------------------------------------------------------------------------
# Numba-accelerated core simulation kernel
# -----------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _simulate_numba_core(cum_probs, initial_state, steps, n_trajectories, keep_count, rands):
    """
    Numba JIT-compiled simulation core.
    
    This is the hot loop that benefits most from JIT compilation.
    Runs 50-100x faster than pure Python.
    """
    state_count = cum_probs.shape[0]
    
    # Initialize states (0-indexed)
    states = np.full(n_trajectories, initial_state, dtype=np.int32)
    
    # Population history
    pop_history = np.zeros((steps + 1, state_count), dtype=np.float64)
    pop_history[0, initial_state] = 1.0
    
    # Kept trajectories
    kept_states = np.zeros((keep_count, steps + 1), dtype=np.int32)
    kept_states[:, 0] = initial_state
    
    # Main simulation loop
    for step in range(steps):
        # Update each trajectory
        for traj in range(n_trajectories):
            r = rands[step, traj]
            current = states[traj]
            
            # Find next state using cumulative probabilities
            for new_state in range(state_count):
                if r < cum_probs[current, new_state]:
                    states[traj] = new_state
                    break
        
        # Count populations
        for s in range(state_count):
            count = 0
            for traj in range(n_trajectories):
                if states[traj] == s:
                    count += 1
            pop_history[step + 1, s] = count / n_trajectories
        
        # Store kept trajectories
        for k in range(keep_count):
            kept_states[k, step + 1] = states[k]
    
    return kept_states, pop_history


@njit(parallel=True, cache=True, fastmath=True)
def _simulate_numba_parallel(cum_probs, initial_state, steps, n_trajectories, keep_count, rands):
    """
    Numba JIT-compiled PARALLEL simulation.
    
    Uses prange for parallel trajectory updates. Best for large n_trajectories.
    """
    state_count = cum_probs.shape[0]
    
    # Initialize states (0-indexed)
    states = np.full(n_trajectories, initial_state, dtype=np.int32)
    
    # Population history
    pop_history = np.zeros((steps + 1, state_count), dtype=np.float64)
    pop_history[0, initial_state] = 1.0
    
    # Kept trajectories
    kept_states = np.zeros((keep_count, steps + 1), dtype=np.int32)
    kept_states[:, 0] = initial_state
    
    # Temporary array for population counting (thread-safe)
    pop_counts = np.zeros(state_count, dtype=np.int64)
    
    # Main simulation loop
    for step in range(steps):
        # Update each trajectory IN PARALLEL
        for traj in prange(n_trajectories):
            r = rands[step, traj]
            current = states[traj]
            
            # Find next state
            for new_state in range(state_count):
                if r < cum_probs[current, new_state]:
                    states[traj] = new_state
                    break
        
        # Count populations (reduction pattern)
        for s in range(state_count):
            pop_counts[s] = 0
        for traj in range(n_trajectories):
            pop_counts[states[traj]] += 1
        for s in range(state_count):
            pop_history[step + 1, s] = pop_counts[s] / n_trajectories
        
        # Store kept trajectories
        for k in range(keep_count):
            kept_states[k, step + 1] = states[k]
    
    return kept_states, pop_history

def _estimate_transition_matrix(markov_fn, state_count, lambda_val, n_samples, verbose=True):
    """
    Estimate transition matrix by sampling the Markov function.
    
    Optimized for large sample counts with:
    - Batched processing for memory efficiency
    - Progress reporting
    - Efficient numpy-based counting
    """
    import sys
    
    trans_matrix = np.zeros((state_count, state_count), dtype=np.float64)
    batch_size = 100_000  # Process in batches for progress updates
    
    if verbose:
        print(f"Estimating transition matrix with {n_samples:,} samples per state...")
    
    for from_state in range(1, state_count + 1):
        # Pre-allocate array for batch results
        counts = np.zeros(state_count, dtype=np.int64)
        samples_done = 0
        
        while samples_done < n_samples:
            # Determine batch size
            current_batch = min(batch_size, n_samples - samples_done)
            
            # Collect batch results into array for efficient counting
            batch_results = np.empty(current_batch, dtype=np.int32)
            
            for i in range(current_batch):
                if lambda_val is None:
                    batch_results[i] = markov_fn(from_state)
                elif isinstance(lambda_val, (list, tuple)) and len(lambda_val) == 2:
                    batch_results[i] = markov_fn(from_state, lambda_val[0], lambda_val[1])
                else:
                    batch_results[i] = markov_fn(from_state, lambda_val)
            
            # Efficient counting with numpy bincount
            batch_counts = np.bincount(batch_results - 1, minlength=state_count)
            counts += batch_counts[:state_count]
            samples_done += current_batch
            
            # Progress update
            if verbose and n_samples >= 1_000_000:
                pct = 100 * samples_done / n_samples
                print(f"\r  State {from_state}/{state_count}: {pct:5.1f}% ({samples_done:,}/{n_samples:,})", end="")
                sys.stdout.flush()
        
        trans_matrix[from_state - 1, :] = counts / n_samples
        
        if verbose:
            if n_samples >= 1_000_000:
                print()  # Newline after progress
            # Show transition probabilities (especially useful for debugging rare transitions)
            nonzero = [(i+1, p) for i, p in enumerate(trans_matrix[from_state - 1]) if p > 0]
            prob_str = ", ".join([f"→{s}: {p:.6f}" for s, p in nonzero])
            print(f"  State {from_state}: {prob_str}")
    
    if verbose:
        print("Transition matrix estimation complete.\n")
    
    return trans_matrix


def simulate_markov_vectorized(markov_fn, initial_state, steps, n_trajectories, 
                                keep_traj=100, state_count=4, lambda_val=None,
                                n_samples=N_SAMPLES_DEFAULT, transition_matrix=None):
    """
    Vectorized Markov chain Monte Carlo simulation.
    
    Estimates transition matrix by sampling, then uses fast NumPy vectorization.
    This is 10-50x faster than Python-loop simulation.
    
    Parameters:
    - n_samples: Number of samples to estimate transition matrix (default from N_SAMPLES_DEFAULT).
                 High default needed for accurate estimation of rare transitions
                 when λ is small (e.g., λ ~ 0.0002).
    - transition_matrix: Optional precomputed transition matrix. If provided, skips
                        sampling estimation (use for analytical matrices).
    """
    # Use provided matrix or estimate by sampling
    if transition_matrix is not None:
        trans_matrix = np.asarray(transition_matrix, dtype=np.float64)
    else:
        trans_matrix = _estimate_transition_matrix(markov_fn, state_count, lambda_val, n_samples)
    
    # Run vectorized simulation
    return _simulate_with_transition_matrix(trans_matrix, initial_state, steps, 
                                            n_trajectories, keep_traj, state_count)

def _simulate_with_transition_matrix(trans_matrix, initial_state, steps, 
                                      n_trajectories, keep_traj, state_count, use_parallel=True):
    """
    Core simulation using precomputed transition matrix with NUMBA JIT.
    
    Uses Numba-compiled kernels for 50-100x speedup over pure Python.
    
    Parameters:
    - use_parallel: If True, uses parallel Numba kernel (best for n_trajectories > 1000)
    """
    # Precompute cumulative probabilities
    cum_probs = np.cumsum(trans_matrix, axis=1).astype(np.float64)
    
    keep_count = min(keep_traj, n_trajectories)
    
    # Pre-generate all random numbers
    rands = np.random.rand(steps, n_trajectories).astype(np.float64)
    
    # Select Numba kernel based on problem size
    if use_parallel and n_trajectories > 1000:
        kept_states, pop_history = _simulate_numba_parallel(
            cum_probs, initial_state - 1, steps, n_trajectories, keep_count, rands
        )
    else:
        kept_states, pop_history = _simulate_numba_core(
            cum_probs, initial_state - 1, steps, n_trajectories, keep_count, rands
        )
    
    # Convert to 1-indexed states for output
    kept_trajs = kept_states + 1
    
    return kept_trajs, pop_history

def simulate_markov(markov_fn, initial_state, steps, n_trajectories, keep_traj=100, 
                    state_count=4, lambda_val=None, num_workers=None, 
                    module_path=None, function_name=None, n_samples=N_SAMPLES_DEFAULT,
                    transition_matrix=None):
    """
    Main simulation entry point.
    
    Uses vectorized simulation for efficiency (10-50x faster than Python loops).
    
    Parameters:
    - n_samples: Number of samples to estimate transition matrix (default 500000)
    - transition_matrix: Optional precomputed transition matrix. If provided, skips
                        sampling estimation (use for analytical matrices).
    """
    # Load function if not provided
    if markov_fn is None and module_path and function_name:
        markov_fn = load_markov_function(module_path, function_name)
    
    # Use vectorized simulation
    return simulate_markov_vectorized(
        markov_fn, initial_state, steps, n_trajectories,
        keep_traj, state_count, lambda_val, n_samples, transition_matrix
    )

def _exp_fit(times, y, tail_fraction=0.5, verbose=True, bounds=None, fit_fraction=None):
    """Fit y ≈ c + a * exp(-t/τ) with simple safeguards.
    - If `bounds` is None, choose sensible bounds from the data.
    - If `fit_fraction` is provided, only fit the first fraction of the data.
    Returns (a, tau, c) or None on failure/low quality.
    """

    def exp_model(t, a, tau, c):
        return c + a * np.exp(-t / tau)

    # Sanitize inputs
    times = np.asarray(times)
    y = np.asarray(y)
    mask = np.isfinite(times) & np.isfinite(y)
    t = times[mask]
    y = y[mask]
    if t.size < 3:
        if verbose:
            print("Exponential fit skipped - insufficient finite samples")
        return None

    # Optional early-window fit (often closer to single-exponential behavior)
    if fit_fraction is not None and 0 < fit_fraction < 1:
        n = max(3, int(len(t) * fit_fraction))
        t = t[:n]
        y = y[:n]

    tail_start = int((1 - tail_fraction) * len(t))
    c_guess = float(np.mean(y[tail_start:]))

    time_range = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    # Auto-bounds from data if not provided
    if bounds is None:
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        y_rng = max(1e-6, y_max - y_min)
        # Detect probability-like data and use tight c-bounds
        is_prob = (y_min >= -0.05) and (y_max <= 1.05)
        if is_prob:
            a_lo, a_hi = -1.5, 1.5
            c_lo, c_hi = -0.1, 1.1
        else:
            a_lo, a_hi = -10 * y_rng, 10 * y_rng
            c_lo, c_hi = y_min - y_rng, y_max + y_rng
        bounds = ([a_lo, 0.1, c_lo], [a_hi, max(1.0, 10 * time_range), c_hi])

    # Initial guesses clipped into bounds
    a_init = np.clip(y[0] - c_guess, bounds[0][0], bounds[1][0])
    tau_init = np.clip(max(10.0, time_range / 10), bounds[0][1], bounds[1][1])
    c_init = np.clip(c_guess, bounds[0][2], bounds[1][2])

    try:
        popt, _ = curve_fit(
            exp_model, t, y,
            p0=[a_init, tau_init, c_init],
            bounds=bounds,
            maxfev=10000,
        )
        a, tau, c = popt

        # Validate parameters
        if not np.isfinite([a, tau, c]).all():
            if verbose:
                print("Exponential fit failed - parameters are not finite")
            return None
        if tau <= 0 or tau > bounds[1][1]:
            if verbose:
                print(f"Exponential fit failed - unreasonable τ={tau:.3g}")
            return None

        # Quality gate (R^2)
        y_hat = c + a * np.exp(-t / tau)
        sse = float(np.sum((y - y_hat) ** 2))
        sst = float(np.sum((y - np.mean(y)) ** 2))
        sst = sst if sst > 1e-12 else 1e-12
        r2 = 1.0 - sse / sst
        if r2 < 0.7:
            if verbose:
                print(f"Exponential fit poor quality (R^2={r2:.2f}); skipping")
            return None

        return a, tau, c
    except Exception as e:
        if verbose:
            print(f"Exponential fit encountered error - {str(e)}")
        return None

def run_markov_experiment(markov_fn=None, module_path=None, function_name=None,
                          initial_state=1, steps=1000, n_trajectories=10000, 
                          dt=0.1, keep_traj=100, state_count=4,
                          lambda_val=None, fit_state=1, tail_fraction=0.25, name="",
                          n_samples=N_SAMPLES_DEFAULT, transition_matrix=None, save_path=None,
                          compare_to_analytical_kraus=None,analytical_pts=500,
                          # Placeholder CHANGEME
                        analytical_init_state=np.array([[0.5, 0. , 0. , 0.5],
            [0. , 0. , 0. , 0. ],
            [0. , 0. , 0. , 0. ],
            [0.5, 0. , 0. , 0.5]]),
                        analytical_symbols={sp.Symbol('λ'): 0.01}):
    """
    Run a complete Markov Monte Carlo experiment with timing and plotting.
    
    Parameters:
    - n_samples: Number of samples to estimate transition matrix (default 500000)
    - transition_matrix: Optional precomputed transition matrix. If provided, skips
                        sampling estimation (use for analytical matrices for exact results).
    - save_path: If provided, saves the plot to this path (e.g., "plots/my_plot.png")
    - compare_to_analytical_kraus: Set to kraus ops, compares simulation results to analytical solution (if available)"""
    start_time = time.time()
    
    if n_samples is None:
        n_samples = 500000
    kept_trajs, pop_history = simulate_markov(
        markov_fn=markov_fn,
        initial_state=initial_state,
        steps=steps,
        n_trajectories=n_trajectories,
        keep_traj=keep_traj,
        state_count=state_count,
        lambda_val=lambda_val,
        module_path=module_path,
        function_name=function_name,
        n_samples=n_samples,
        transition_matrix=transition_matrix,
    )
    elapsed = time.time() - start_time
    

    from .visualizer import plot_markov_results
    plot_markov_results(kept_trajs, pop_history, dt=dt, fit_state=fit_state, 
                       tail_fraction=tail_fraction, name=name, save_path=save_path)
    
    throughput = (n_trajectories * steps) / elapsed / 1e6
    print(f"Simulation completed in {elapsed:.2f}s ({throughput:.1f}M transitions/s)")
    
    # Compute asymptotic populations from final portion
    tail_start = int(0.8 * pop_history.shape[0])
    asymptotic_pops = np.mean(pop_history[tail_start:], axis=0)
    print(f"Asymptotic populations: {asymptotic_pops}")
    
    # Use density operator for analytical comparison
    if compare_to_analytical_kraus is not None:
        from .derivation import density_operator_simulator
        analytical_results = density_operator_simulator(compare_to_analytical_kraus,  steps=analytical_pts, initial_state=analytical_init_state, symbols_values=analytical_symbols)
        analytical_pops = []
        # get populations
        for dm in analytical_results:
            pops = [float(dm[i, i]) for i in range(state_count)]
            analytical_pops.append(pops)
        # Plot populations alongside mk sim
        # add 'analytical_comparison' to name
        # expect there to be .png at end
        from .visualizer import plot_mk_analytic_comparison

        save_path_analytical = None
        if save_path:
            assert save_path.endswith('.png'), "save_path must end with .png"
            save_path_analytical = save_path[:-4] + '_analytical_comparison.png'
        analytical_pops = np.array(analytical_pops)
        plot_mk_analytic_comparison(analytical_pops, pop_history, dt, name=name, save_path=save_path_analytical if save_path else None)
        
    
    return kept_trajs, pop_history, asymptotic_pops

# =============================================================================
# Numba JIT Warmup (compile kernels on first load)
# =============================================================================
if HAS_NUMBA:
    print("Warming up Numba JIT kernels...")
    _warmup_start = time.time()

    # Small warmup run to trigger JIT compilation
    _warmup_cum_probs = np.array([[0.9, 1.0], [0.1, 1.0]], dtype=np.float64)
    _warmup_rands = np.random.rand(10, 100).astype(np.float64)
    _ = _simulate_numba_core(_warmup_cum_probs, 0, 10, 100, 5, _warmup_rands)
    _ = _simulate_numba_parallel(_warmup_cum_probs, 0, 10, 100, 5, _warmup_rands)

    print(f"Numba warmup complete in {time.time() - _warmup_start:.2f}s")
else:
    print("Numba not installed; using pure Python simulation (slower). Install with: pip install numba")