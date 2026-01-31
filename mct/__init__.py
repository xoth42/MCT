# """MCT: Derive and validate Markov-chain transition matrices from quantum noise models."""

# __version__ = "0.1.0"

from .derivation import apply_channel, probability_symbol, get_density_basis, to_bell_basis, ρ00_comp, ρ01_comp, ρ10_comp, ρ11_comp
from .visualizer import lmatrix, apply_channel_basis

from .main import transition_matrix_from_sympy, run_markov_experiment

__all__ = [
    "apply_channel",                # Derivation
    "probability_symbol",
    "get_density_basis",
    "to_bell_basis",
    "lmatrix",                      # Visualization
    "apply_channel_basis",
    "transition_matrix_from_sympy", # Main mct testing
    "run_markov_experiment"
]
