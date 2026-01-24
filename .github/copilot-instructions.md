MCT – Markov Chain Deriver / Generator / Tester

References folder: references/

PROJECT PURPOSE
- Build generalized tools to:
  (1) derive Markov-chain transition matrices from physical noise models, and
  (2) generate fast sampling code plus tests.
- Primary examples: T1/T2 and ADC noise on Bell pairs and related two-qubit channels.
- Design must support arbitrary finite-state Markov processes, not just 4-state Bell systems.

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
