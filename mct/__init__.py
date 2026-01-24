"""MCT: Derive and validate Markov-chain transition matrices from quantum noise models."""

__version__ = "0.1.0"

# Core classes
# Constants and basis definitions
from .constants import (
    AMPLITUDE_DAMPING_KRAUS,
    TWO_QUBIT_ADC_KRAUS_PARAMS,
    bell_basis,
    computational_basis,
    get_basis,
)

# Core API (symbolic derivation, sampling, code generation)
from .main import (
    apply_channel,
    derive_transition_matrix,
    substitute_and_verify,
    superoperator,
)

# Import markov_chain module for simulation classes
from .markov_chain import (
    MarkovChain,
    empirical_distribution_from_trajectories,
    generate_sampling_function,
    generate_sampling_vectorized,
    simulate_markov_trajectory,
    simulate_markov_vectorized,
    theoretical_distribution_after_steps,
)

# Now import the markov_chain generation function (must be AFTER module import)
from .main import markov_chain

# Noise models (numeric utilities)
from .noise_models import thermal_relaxation_error_rate

# Visualization
from .visualizer import plot_populations, print_matrix_latex

__all__ = [
    # Classes
    "MarkovChain",
    # Simulation functions
    "simulate_markov_trajectory",
    "simulate_markov_vectorized",
    "empirical_distribution_from_trajectories",
    "theoretical_distribution_after_steps",
    # Symbolic derivation (generalized)
    "apply_channel",
    "derive_transition_matrix",
    "substitute_and_verify",
    # Sampling
    "generate_sampling_function",
    "generate_sampling_vectorized",
    # Noise models
    "thermal_relaxation_error_rate",
    # Bases
    "computational_basis",
    "bell_basis",
    "get_basis",
    # Constants
    "AMPLITUDE_DAMPING_KRAUS",
    "TWO_QUBIT_ADC_KRAUS_PARAMS",
    # Visualization
    "plot_populations",
    "print_matrix_latex",
    # High-level API (user-facing)
    "superoperator",
    "markov_chain",
]

