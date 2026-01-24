import numpy as np


def flip_state(state):
    """Left-right flip for (C, R, C) encoded state."""
    return np.flip(state, axis=2).copy()


def flip_policy(policy):
    """Left-right flip for column policy (cols,)."""
    return policy[::-1].copy()
