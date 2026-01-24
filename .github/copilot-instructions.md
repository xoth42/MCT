MCT – Markov Chain Deriver / Generator / Tester

References folder: references/

PROJECT PURPOSE & VISION
- Build generalized tools to:
  (1) derive Markov-chain transition matrices from physical noise models, and
  (2) generate fast sampling code plus tests.
- Primary examples: T1/T2 and ADC noise on Bell pairs and related two-qubit channels.
- Design must support arbitrary finite-state Markov processes, not just 4-state Bell systems.

TARGET USER EXPERIENCE
Users should interact with MCT like this:

```python
import sympy as sp
import MCT

# Define Kraus operators with symbols
lam = sp.symbols('lambda', real=True, positive=True)
K0 = sp.Matrix([[1, 0], [0, sp.sqrt(1-lam)]])
K1 = sp.Matrix([[0, sp.sqrt(lam)], [0, 0]])

# Derive analytical transition matrix
result = MCT.superoperator(
    (K0, K1),
    basis_state=MCT.computational_basis[0],
    basis=MCT.computational_basis,
    qubits=1
)

# Generate and save sampling code
MCT.markov_chain(result, to_file="my_chains/sampling.py")
```

This design hides complexity while exposing clear semantics:
- Kraus operators (physics)
- Basis states (representation choice)
- Symbolic derivation (analytic preservation)
- Sampling code generation (performance)

API DESIGN GOALS

ABSOLUTE PRIORITY: REUSE EXISTING CODE
- Before writing new logic, look for and reuse patterns from:
  - markov_chain_generator_full_code_jan18.pdf
    • Functions like create_markov_chain_function that:
      - Take a list of density matrices,
      - Extract diagonal entries as transition probabilities,
      - Emit a sampling function using cumulative probability comparisons (if/elif/else),
      - Optionally write the function to a .py file.<citation filename="markov_chain_generator_full_code_jan18.pdf" page="10"></citation>
    • Utilities for estimating transition matrices and simulating Markov chains with sampling and NumPy vectorization.<citation filename="markov_chain_generator_full_code_jan18.pdf" page="5"></citation>
  - twiling_T1_Jan18.pdf
    • Symbolic derivations of two-qubit amplitude damping and ADC in both computational and Bell bases; Bell-basis density matrices with explicit polynomial entries in λ, λ1, λ2 ready to be turned into Markov transitions.<citation filename="twiling_T1_Jan18.pdf" page="7"></citation><citation filename="twiling_T1_Jan18.pdf" page="10"></citation>
  - T1 T2 noise.pdf and CS648QML_error_kraus.py.pdf
    • Standard single-qubit amplitude damping and phase-damping Kraus operators and T1/T2 parameterization, including tensor-product construction for two-qubit channels.<citation filename="T1 T2 noise.pdf" page="3"></citation><citation filename="CS648QML_error_kraus.py.pdf" page="2"></citation><citation filename="CS648QML_error_kraus.py.pdf" page="3"></citation>
- Prefer:
  - Refactoring, generalizing, or wrapping these patterns
  - Over inventing new APIs or reimplementing functionality from scratch.
- Only write new code when:
  - Extending to new dimensions/state spaces,
  - Generalizing the “fixed 4-state” patterns to N-state,
  - Or adding missing tests/diagnostics.

CORE RULES
1) Markov chain as explicit primary object
   - Represent a Markov chain by a matrix P where P[i, j] = Prob(state i ← state j).
   - P must be stochastic: entries ≥ 0, each column or row sum = 1 (be consistent and document the choice).

2) Separate concerns
   - Distinguish clearly between:
     - (a) Physics/noise model: channels, Kraus operators, parameters.
     - (b) Symbolic derivation of transition probabilities.
     - (c) Construction of numeric transition matrices for fixed parameter values.
     - (d) Sampling functions (r = rand(); if/elif/else).
     - (e) Testing and diagnostics.

3) Preserve analytic structure
   - When possible, express transitions as analytic functions of parameters (e.g. λ, λ1, λ2, p_x, p_y, p_z).
   - Don’t prematurely substitute arbitrary numeric constants.
   - Use SymPy or similar where appropriate (as in twiling_T1_Jan18 and markov_chain_generator_full_code_jan18 patterns).<citation filename="markov_chain_generator_full_code_jan18.pdf" page="10"></citation><citation filename="twiling_T1_Jan18.pdf" page="7"></citation>
API / MODULE DESIGN (CANONICAL)

HIGH-LEVEL PUBLIC API:

1) MCT.superoperator(kraus_ops, basis_state, basis, qubits) → (P_symbolic, metadata)
   - Inputs:
     • kraus_ops: tuple of sympy.Matrix Kraus operators
     • basis_state: sympy.Matrix density matrix for a single basis state
     • basis: list of basis state density matrices (e.g., MCT.computational_basis)
     • qubits: int (1, 2, etc.)
   - Process:
     • Apply channel to basis_state: ρ_out = Σ_k E_k ρ_state E_k†
     • Extract diagonal entries as transition probabilities
     • Return symbolic transition matrix P(parameters)
   - Output:
     • P_symbolic: sympy.Matrix, column-stochastic, parametrized by symbols
     • metadata: dict with symbol definitions, edge cases, Kraus operators, etc.
   
2) MCT.markov_chain(analytical_result, to_file=None) → code_string
   - Input:
     • analytical_result: tuple (P_symbolic, metadata) from superoperator()
     • to_file: optional path to write generated sampling code
   - Process:
     • Substitute numeric or symbolic parameter values
     • Generate Python function with if/elif/else cumulative probability pattern
     • Write to file if specified
   - Output:
     • code_string: Python source code for fast sampling

3) MCT.computational_basis, MCT.bell_basis, MCT.get_basis(name, qubits)
   - Predefined basis state collections for common scenarios
   - Each basis is a list of sympy.Matrix density matrices

INTERNAL MODULE STRUCTURE:

mct/
├── __init__.py                 # Exports: superoperator(), markov_chain(), bases
├── symbolic_derivation.py      # Core: derive_single_qubit_t1_symbolic(), apply_channel()
├── markov_chain.py             # Core: MarkovChain class & validation
├── sampling.py                 # Core: generate_sampling_function()
├── bases.py                    # NEW: computational_basis, bell_basis definitions
├── noise_models.py             # Utilities: thermal_relaxation_error_rate(), etc.
└── integration.py              # NEW: superoperator() and markov_chain() wrappers
REFERENCE PATTERN: T1/T2 AND ADC
- Use as templates:

  1) T1 (amplitude damping)
     - Single-qubit Kraus operators:
       E0(λ) and E1(λ) as in T1 T2 noise and CS648QML_error_kraus; extend to two qubits via tensor products.<citation filename="T1 T2 noise.pdf" page="3"></citation><citation filename="CS648QML_error_kraus.py.pdf" page="2"></citation>
     - twiling_T1_Jan18 and T1 T2 noise provide explicit Bell-basis density matrices and diagonal entries as polynomials in λ; use those diagonals as transition probabilities for 4-state Bell Markov chains.<citation filename="twiling_T1_Jan18.pdf" page="9"></citation><citation filename="twiling_T1_Jan18.pdf" page="10"></citation>

  2) ADC (Amplitude + Dephasing Channel)
     - twiling_T1_Jan18 provides Bell-basis density matrices for ADC with parameters λ1, λ2; these diagonals define 4×4 transition matrices for Bell states (states 00, 10, 01, 11).<citation filename="twiling_T1_Jan18.pdf" page="7"></citation>
     - Reuse the “format_bell_markov_matrices” style: Transform density matrices into lists of 4 diagonal entries per Bell state, then feed them into create_markov_chain_function-style generators.<citation filename="twiling_T1_Jan18.pdf" page="7"></citation><citation filename="markov_chain_generator_full_code_jan18.pdf" page="10"></citation>

  3) General quantum channels
     - For new channels, follow the same recipe:
       • Construct channel with Kraus operators (see qubit_guide Ch. 9 for operator-sum form).<citation filename="qubit_guide.pdf" page="177"></citation><citation filename="qubit_guide.pdf" page="178"></citation>
       • Compute output density matrices for each chosen basis state.
       • Extract diagonals to get per-state transition probabilities.
       • Plug diagonals into a generalized create_markov_chain_function.

API / ABSTRACTION GOALS
- Provide abstractions for:
  - Finite state spaces:
    • Internal canonical indices 0..N-1 (or 1..N in Julia).
    • Optional helpers for mapping between bitstrings, Bell labels, and indices.
  - Transition specifications:
    • Declarative description: “from state j to state i: probability expression f(λ, …)”.
  - Channel-based derivation:
    • Given Kraus operators and basis, build density matrices and derive transitions automatically.
    • Direction: use tools/year patterns from twiling_T1_Jan18 for density-matrix generation, and from markov_chain_generator_full_code_jan18 for Markov-function codegen.

SYMBOLIC DERIVATION (HIGH-LEVEL)
- Inputs:
  - Either: list of density matrices ρ_j (for each basis state j) with parameters.
  - Or: a channel object (Kraus set or superoperator) + explicit basis.
- Process:
  - For each “source” basis state j:
    • ρ_j_out = channel(ρ_j).
    • Extract p(i ← j) = (ρ_j_out)_ii for each basis state i.
- Outputs:
  - Symbolic expressions for p(i ← j), collected into a matrix P(λ, …).
  - Symbolic or numeric verification that ∑_i P(i, j) = 1 for every j.

SAMPLING CODE GENERATION (REUSE PATTERNS)
- Use the pattern from create_markov_chain_function (markov_chain_generator_full_code_jan18):
  - Construct if/elif/else tree over cumulative probabilities for each initial_state.
  - Use a single rand() call per step and cumulative comparisons:
    r = rand()
    if     r < p1            → state_1
    elif   r < p1 + p2       → state_2
    ...
    else                     → state_k   # Final branch: sum = 1
  - Comment the final “else” with the total cumulative probability to show it is 1.<citation filename="markov_chain_generator_full_code_jan18.pdf" page="11"></citation>
- Generalize:
  - Allow N states, not just 4.
  - Support multiple parameters (λ1, λ2, etc.).
  - Optionally write generated code to files (like the original implementation).

TESTING REQUIREMENTS
- Every Markov chain generator must have tests that:

  1) Stochasticity
     - Check 0 ≤ P[i, j] ≤ 1 for all i, j.
     - Check ∑_i P[i, j] = 1 to numerical tolerance or symbolically (for symbolic matrices).

  2) Parameter edge cases
     - For quantum noise models:
       • λ = 0, λ1 = 0, λ2 = 0 → identity channel (P = I).
       • Maximal noise (e.g., λ = 1) → expected limiting distribution (e.g., full relaxation to ground state for T1).<citation filename="Manenti_Motta-Do_not_share.pdf" page="336"></citation>

  3) Monte Carlo validation
     - Use simulate_markov_vectorized from markov_chain_generator_full_code_jan18 or a refactored version to:
       • Generate many trajectories or one-step samples.
       • Compare empirical distributions to theoretical P^n distributions.<citation filename="markov_chain_generator_full_code_jan18.pdf" page="5"></citation>

  4) Structure checks (optional)
     - Reachability, stationary distributions, irreducibility, etc., for more complex chains.

STATE AND ENCODING CONVENTIONS
- Internal representation:
  - Use a consistent indexing convention per language (e.g., 0-based in Python, 1-based in Julia).
- For quantum-oriented examples:
  - Provide helpers mapping between:
    • computational basis labels (e.g., "00", "01", "10", "11"),
    • Bell labels (A, B, C, D or 00, 10, 01, 11),
    • and integer indices.

PERFORMANCE GUIDELINES
- Keep inner loops for sampling and simulation tight:
  - Avoid unnecessary allocation in each step.
  - Use vectorized operations (NumPy) where appropriate, as in simulate_markov_vectorized.<citation filename="markov_chain_generator_full_code_jan18.pdf" page="5"></citation>
- In Julia-style contexts (e.g., BPGates.jl patterns):
  - Use @inbounds, @simd, Val(N) where they are standard and safe.

PHYSICS / NOISE REFERENCES
- For definitions and physical constraints:
  - qubit_guide.pdf, Ch. 9: quantum channels, Kraus operator-sum form, composition, and complete positivity.<citation filename="qubit_guide.pdf" page="177"></citation><citation filename="qubit_guide.pdf" page="178"></citation><citation filename="qubit_guide.pdf" page="181"></citation>
  - Quantum Noise (Gardiner & Zoller): general Lindblad master equations and Markovian generators (background, not usually needed in code directly).<citation filename="Quantum Noise_ A Handbook of Markovian and Non-Markovian -- Crispin W Gardiner; Peter Zoller -- Springer series in synergetics, 2nd enlarged ed, 2004 -- 9783540223016.pdf" page="165"></citation>
  - Manenti_Motta-Do_not_share.pdf: practical T1/T2 relations and master-equation forms for qubits, connecting λ-like parameters to T1/T2.<citation filename="Manenti_Motta-Do_not_share.pdf" page="300"></citation><citation filename="Manenti_Motta-Do_not_share.pdf" page="336"></citation>
  - Surface code with decoherence…: use only for context/benchmarks, not for core API design.
  - BPGates.jl: a Julia reference for realistic noise and gate modeling; follow its style where it aligns with our abstractions.

WHAT COPILOT SHOULD AVOID
- Do NOT:
  - Invent new physical models or parameters unrelated to the references.
  - Re-implement existing features already in markov_chain_generator_full_code_jan18 or twiling_T1_Jan18 if a simple generalization/refactor is possible.
  - Mix derivation, sampling, and testing logic in a single large function.
- DO:
  - Favor incremental generalization and stronger tests over “new from scratch” code.
  - Keep APIs small, composable, and documented.
DEVELOPMENT WORKFLOW (STRICT RULES)
===================================

VALIDATION RULE (Non-Negotiable)
- No new features are marked complete until tests PASS
- All validation happens through tests/ folder ONLY (use pytest)
- Run pytest on affected test files before moving on
- Show test output (pass/fail) in each response

FOR EACH NEW FEATURE / FIX
1. Identify what needs to be added/changed
2. Create or modify the code
3. Create/update corresponding test in tests/
4. Execute pytest on that test file
5. Display results (failures = stop, must fix)
6. Only proceed to next feature if tests pass
7. Get user approval before proceeding to next feature

FILE RULES
✓ ALLOWED: Core logic (mct/*.py), Tests (tests/*.py), setup.py
✗ PROHIBITED: README, documentation markdown (unless updating existing docs)
✗ PROHIBITED: Standalone validation/demo scripts outside tests/
✗ PROHIBITED: Multiple independent features per response

CODE QUALITY
- Each test validates ONE specific behavior
- Tests must import from actual implementation modules (no mocks)
- Tests must execute with only pytest, numpy, sympy as dependencies
- No orphaned code: every function must have corresponding test case
- All imports must work in clean .venv or conda environment

ENVIRONMENT RULES
- Use .venv or conda environment ONLY
- Never modify global Python environment
- setup.py defines all dependencies (including pytest in extras_require["dev"])
- Install with: pip install -e . (for development) or pip install -e ".[dev]" (with test tools)

TEST EXECUTION CHECKLIST
Before marking any work as "complete":
  □ Test file exists and is in tests/
  □ All imports in test are from mct.* modules
  □ pytest runs without environment setup errors
  □ All tests in that file PASS (0 failures)
  □ Show actual pytest output in response

COMMUNICATION PROTOCOL
- Start each work section: "Working on: [module name]"
- Display pytest output verbatim (not summaries)
- End with: "Result: [X/Y tests passed]" or "Result: PASS/FAIL"
- If test fails, show error, fix code/test, re-run, then show passing output
- Ask for approval: "Feature complete. Ready for [next item]?" (wait for response)

AGENTIC ENVIRONMENT & TERMINAL BEST PRACTICES
==============================================

UV Environment Setup (Canonical Approach)
- This project uses 'uv' for Python environment and dependency management
- Initialize with: uv sync (installs core dependencies)
- Install dev tools: uv add --dev pytest
- Run commands via uv: uv run pytest tests -v
- Benefits: Reproducible, fast, deterministic, isolated from global Python

Terminal Output Integration (Non-Negotiable)
- ALWAYS use create_and_run_task tool to run pytest or any validation
- NEVER use raw run_in_terminal without capturing task output
- Agentic tools: create_and_run_task + get_task_output for visibility
- Pattern: 
  1. create_and_run_task with pytest command
  2. Inspect task output from terminal
  3. Show full output to user verbatim
  4. Act on failures (fix → re-run → verify pass)

Verification Checklist (Before Every Feature)
□ Activate venv manually if needed: source .venv/bin/activate
□ If venv is corrupted, nuke it: rm -rf .venv && uv sync --dev
□ Run via uv: uv run pytest tests -v (bypasses activation issues)
□ Capture actual pytest output in terminal/task
□ Verify all tests PASS before proceeding
□ Show output to user (not just "tests passed")

Common Issues & Fixes
- Problem: "command not found: pytest"
  Solution: Use `uv run pytest` instead of bare `pytest`
- Problem: "No module named pip" in corrupted venv
  Solution: rm -rf .venv && uv sync --dev
- Problem: Task terminal not seeing activated venv
  Solution: Always prefix commands with `uv run` to use environment automatically
- Problem: Partial test output visible
  Solution: Use create_and_run_task with full command, check terminal output in task results

TASKS.JSON HOUSEKEEPING (Non-Negotiable)
- .vscode/tasks.json is shared across all development sessions
- DO NOT add duplicate tasks (check file first before creating)
- USE EXISTING TASKS: "Run pytest", "Full test suite", "Sync with dev dependencies"
- REUSE EXISTING TASKS for validation rather than creating new ones
- MAX RULE: Keep tasks.json under 50 tasks total
- If a new validation pattern is needed that doesn't exist, create ONE reusable task (not multiple debug variants)
- Before each session: Review current tasks.json and clean up any obsolete debug tasks

Key Learning (Why This Matters)
- Agentic systems need visible I/O: agent → terminal → output → agent can see results
- UV is lightweight and reproducible, far better than ad-hoc pip installations
- Task terminal output provides ground truth for what actually happened
- Visible feedback loop enables proper test-driven development and failure recovery



TEST DISCIPLINE
- NO: Tests that don't assert anything meaningful
  BAD:    def test_something():
              result = compute_something()
              # Passes as long as no exception raised
  DO:     def test_something():
              result = compute_something()
              assert result is not None
              assert result.shape == (4, 4)
              assert abs(result.sum() - 1.0) < 1e-10

- NO: Multiple independent features tested in one test function
  WHY: Makes it hard to pinpoint failures; reduces clarity.
  DO: One test function = one behavior (stochasticity, edge case, correctness, etc.)
      Name clearly: test_stochasticity_* / test_edge_case_* / test_monte_carlo_*

- NO: Tests that pass by accident (e.g., assert True)
  WHY: False confidence; won't catch real bugs.
  DO: Ensure assertion would *fail* if code changed incorrectly.

DOCUMENTATION DISCIPLINE
- NO: Verbose docstrings that just repeat the function signature
  BAD:    def apply_channel(kraus_ops, rho):
              """Apply channel to density matrix.
              
              Args:
                  kraus_ops: Kraus operators
                  rho: Density matrix
              
              Returns:
                  Result density matrix
              """
  DO:     def apply_channel(kraus_ops, rho):
              """ρ_out = Σ_k E_k ρ E_k†; assumes E_k are sympy.Matrix."""

- NO: Docstrings for trivial functions
  WHY: Noise if the function name is self-explanatory.
  DO: Use docstrings only when intent or mathematical meaning isn't obvious from name.

- NO: Inline documentation (README expansions, design docs) unless updating existing docs
  WHY: Keeps code repo focused; design changes belong in copilot-instructions.md or references/.
  DO: All design policy lives in this file; code comments explain *only the code itself*.

CONFIGURATION DISCIPLINE
- NO: Magic numbers hardcoded in functions
  WHY: Hidden assumptions; hard to tune.
  DO:  Define at module level with clear name and meaning:
       # Numerical tolerance for stochasticity checks
       TOLERANCE_STOCHASTIC = 1e-10

