# """Markov chain representation and validation."""

# from typing import List, Optional

# import numpy as np


# class MarkovChain:
#     """Finite-state Markov chain: P[i,j] = Prob(i | j), columns sum to 1."""

#     def __init__(
#         self,
#         transition_matrix: np.ndarray,
#         state_labels: Optional[List[str]] = None,
#         name: str = "MarkovChain"
#     ):
#         """Initialize with NxN transition matrix."""
#         P = np.asarray(transition_matrix, dtype=float)

#         if P.ndim != 2 or P.shape[0] != P.shape[1]:
#             raise ValueError("Transition matrix must be square")

#         self.P = P
#         self.n_states = P.shape[0]
#         self.name = name
#         self.state_labels = state_labels or [str(i) for i in range(self.n_states)]

#     def validate_stochasticity(self, tol: float = 1e-10) -> dict:
#         """Check if P is stochastic (entries in [0,1], columns sum to 1)."""
#         violations = []

#         if np.any(self.P < -tol) or np.any(self.P > 1.0 + tol):
#             violations.append("entries_out_of_bounds")

#         col_sums = np.sum(self.P, axis=0)
#         if not np.allclose(col_sums, 1.0, atol=tol):
#             violations.append("columns_do_not_sum_to_one")

#         return {
#             "is_valid": len(violations) == 0,
#             "violations": violations,
#             "column_sums": col_sums,
#             "min_entry": np.min(self.P),
#             "max_entry": np.max(self.P),
#         }

#     def apply_steps(self, initial_state_dist: np.ndarray, n_steps: int) -> np.ndarray:
#         """Apply n Markov steps: dist_{n} = P^n @ dist_0."""
#         dist = np.asarray(initial_state_dist, dtype=float)
#         for _ in range(n_steps):
#             dist = self.P @ dist
#         return dist

#     def __repr__(self) -> str:
#         return (
#             f"MarkovChain(name='{self.name}', n_states={self.n_states})\n"
#             f"States: {self.state_labels}\n"
#             f"Transition matrix shape: {self.P.shape}"
#         )

#     def __str__(self) -> str:
#         return repr(self)


# def simulate_markov_trajectory(
#     transition_matrix: np.ndarray,
#     initial_state: int,
#     n_steps: int,
#     rng: Optional[np.random.Generator] = None,
# ) -> np.ndarray:
#     """Single trajectory via cumulative probability comparisons (ref: markov_chain_generator_full_code_jan18)."""
#     if rng is None:
#         rng = np.random.default_rng()

#     n_states = transition_matrix.shape[0]
#     trajectory = np.zeros(n_steps + 1, dtype=int)
#     trajectory[0] = initial_state

#     current_state = initial_state
#     for step in range(n_steps):
#         # Get transition probabilities FROM current state
#         probs = transition_matrix[:, current_state]

#         # Draw uniform random number
#         r = rng.uniform(0, 1)

#         # Cumulative probability comparison (if/elif/else pattern)
#         cumsum = 0.0
#         next_state = n_states - 1  # Default to last state
#         for state_idx in range(n_states):
#             cumsum += probs[state_idx]
#             if r < cumsum:
#                 next_state = state_idx
#                 break

#         trajectory[step + 1] = next_state
#         current_state = next_state

#     return trajectory


# def simulate_markov_vectorized(
#     transition_matrix: np.ndarray,
#     initial_state: int,
#     n_steps: int,
#     n_trajectories: int,
#     rng: Optional[np.random.Generator] = None,
# ) -> np.ndarray:
#     """Vectorized trajectories: shape (n_steps+1, n_trajectories) via cumulative probability sums."""
#     if rng is None:
#         rng = np.random.default_rng()

#     n_states = transition_matrix.shape[0]
#     trajectories = np.zeros((n_steps + 1, n_trajectories), dtype=int)
#     trajectories[0, :] = initial_state

#     current_states = np.full(n_trajectories, initial_state, dtype=int)

#     for step in range(n_steps):
#         # Get transition probabilities for all current states
#         # shape: (n_states, n_trajectories)
#         probs = transition_matrix[:, current_states]

#         # Compute cumulative probabilities along state axis
#         # shape: (n_states, n_trajectories)
#         cumulative_probs = np.cumsum(probs, axis=0)

#         # Draw random numbers for all trajectories
#         # shape: (n_trajectories,)
#         random_draws = rng.uniform(0, 1, n_trajectories)

#         # Compare each random draw against cumulative probabilities
#         # Broadcasting: (n_states, n_trajectories) vs (n_trajectories,)
#         # Result: (n_states, n_trajectories) boolean matrix
#         comparisons = random_draws[np.newaxis, :] < cumulative_probs

#         # For each trajectory, find the first state where random < cumsum
#         # This gives the next state for each trajectory
#         next_states = np.argmax(comparisons, axis=0)

#         trajectories[step + 1, :] = next_states
#         current_states = next_states

#     return trajectories


# def empirical_distribution_from_trajectories(
#     trajectories: np.ndarray,
#     step: int,
#     n_states: int,
# ) -> np.ndarray:
#     """Empirical distribution at given step: fraction of trajectories in each state."""
#     states_at_step = trajectories[step, :]
#     empirical_dist = np.zeros(n_states)
#     for state in range(n_states):
#         empirical_dist[state] = np.sum(states_at_step == state) / len(states_at_step)
#     return empirical_dist


# def theoretical_distribution_after_steps(
#     transition_matrix: np.ndarray,
#     initial_state: int,
#     n_steps: int,
# ) -> np.ndarray:
#     """Theoretical distribution: P^n @ e_initial_state."""
#     # Start with a unit vector in the initial state
#     initial_dist = np.zeros(transition_matrix.shape[0])
#     initial_dist[initial_state] = 1.0

#     # Apply matrix multiplication n times
#     P_power = transition_matrix.copy()
#     for _ in range(n_steps - 1):
#         P_power = P_power @ transition_matrix

#     # Result is P^n @ initial_dist
#     return P_power @ initial_dist


# # ============================================================================
# # Sampling Code Generation (from sampling.py)
# # ============================================================================

# def generate_sampling_function(
#     transition_matrix: np.ndarray,
#     state_labels: Optional[List[str]] = None,
#     function_name: str = "sample_markov_step",
# ) -> str:
#     """Generate Python function code for Markov chain sampling (if/elif/else pattern)."""
#     P = np.asarray(transition_matrix, dtype=float)
#     n_states = P.shape[0]

#     if state_labels is None:
#         state_labels = [str(i) for i in range(n_states)]

#     lines = [
#         f"def {function_name}(current_state: int) -> int:",
#         '    """Sample next state using cumulative probabilities."""',
#         "    import random",
#         "    r = random.random()",
#         "",
#     ]

#     # For each possible initial state, generate a branch
#     for from_state in range(n_states):
#         label = state_labels[from_state]
#         lines.append(f"    if current_state == {from_state}:  # {label}")

#         # Extract transition probabilities from this state
#         probs = P[:, from_state]

#         # Cumulative probabilities
#         cum_probs = np.cumsum(probs)

#         # Generate if/elif/else for each destination state
#         for to_state in range(n_states):
#             to_label = state_labels[to_state]
#             cum_prob = cum_probs[to_state]

#             if to_state == 0:
#                 lines.append(f"        if r < {cum_prob:.15f}:")
#                 lines.append(f"            return {to_state}  # -> {to_label}")
#             elif to_state == n_states - 1:
#                 lines.append(f"        else:  # cumsum = {cum_prob:.15f}")
#                 lines.append(f"            return {to_state}  # -> {to_label}")
#             else:
#                 lines.append(f"        elif r < {cum_prob:.15f}:")
#                 lines.append(f"            return {to_state}  # -> {to_label}")

#         lines.append("")

#     return "\n".join(lines)


# def generate_sampling_vectorized(
#     transition_matrix: np.ndarray,
#     state_labels: Optional[List[str]] = None,
#     function_name: str = "sample_markov_trajectory",
# ) -> str:
#     """Generate Python code for vectorized sampling (NumPy-based trajectory)."""
#     P = np.asarray(transition_matrix, dtype=float)
#     n_states = P.shape[0]

#     if state_labels is None:
#         state_labels = [str(i) for i in range(n_states)]

#     lines = [
#         f"def {function_name}(initial_state: int, n_steps: int) -> list:",
#         '    """Sample a trajectory of n_steps from Markov chain starting at initial_state."""',
#         "    import numpy as np",
#         "    P = np.array([",
#     ]

#     # Add transition matrix rows
#     for i in range(n_states):
#         row_str = ", ".join([f"{P[i, j]:.15f}" for j in range(n_states)])
#         lines.append(f"        [{row_str}],")

#     lines.extend([
#         "    ])",
#         "    trajectory = [initial_state]",
#         "    current = initial_state",
#         "    for _ in range(n_steps):",
#         "        probs = P[:, current]",
#         "        next_state = np.random.choice(len(probs), p=probs)",
#         "        trajectory.append(next_state)",
#         "        current = next_state",
#         "    return trajectory",
#     ])

#     return "\n".join(lines)
