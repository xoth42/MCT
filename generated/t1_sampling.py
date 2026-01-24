def sample_markov_step(current_state: int) -> int:
    """Sample next state using cumulative probabilities."""
    import random
    r = random.random()

    if current_state == 0:  # 0
        if r < 1.000000000000000:
            return 0  # -> 0
        else:  # cumsum = 1.000000000000000
            return 1  # -> 1

    if current_state == 1:  # 1
        if r < 0.300000000000000:
            return 0  # -> 0
        else:  # cumsum = 1.000000000000000
            return 1  # -> 1
