import hashlib
import struct
import numpy as np
from env.env import GreatKingdomEnv


TRANSFORMS = (
    "rot90",
    "rot180",
    "rot270",
    "flip_lr",
    "flip_ud",
    "flip_diag_main",
    "flip_diag_anti",
)

CANON_TRANSFORMS = ("identity",) + TRANSFORMS

_ACTION_INDEX_MAP = {}


def _apply_transform_array(arr, transform):
    """Apply a symmetry transform to a 2D board or 3D (C,H,W) tensor."""
    if transform == "identity":
        return arr.copy() if hasattr(arr, "copy") else arr
    if arr.ndim == 2:
        if transform == "rot90":
            return np.rot90(arr, k=-1)
        if transform == "rot180":
            return np.rot90(arr, k=2)
        if transform == "rot270":
            return np.rot90(arr, k=1)
        if transform == "flip_lr":
            return np.fliplr(arr)
        if transform == "flip_ud":
            return np.flipud(arr)
        if transform == "flip_diag_main":
            return arr.T
        if transform == "flip_diag_anti":
            return np.flipud(np.fliplr(arr.T))
    elif arr.ndim == 3:
        if transform == "rot90":
            return np.rot90(arr, k=-1, axes=(1, 2))
        if transform == "rot180":
            return np.rot90(arr, k=2, axes=(1, 2))
        if transform == "rot270":
            return np.rot90(arr, k=1, axes=(1, 2))
        if transform == "flip_lr":
            return np.flip(arr, axis=2)
        if transform == "flip_ud":
            return np.flip(arr, axis=1)
        if transform == "flip_diag_main":
            return np.transpose(arr, (0, 2, 1))
        if transform == "flip_diag_anti":
            return np.flip(np.flip(np.transpose(arr, (0, 2, 1)), axis=1), axis=2)
    raise ValueError(f"Unsupported transform or shape: {transform}, ndim={arr.ndim}")


def _transform_coord(r, c, n, transform):
    if transform == "identity":
        return r, c
    if transform == "rot90":
        return c, n - 1 - r
    if transform == "rot180":
        return n - 1 - r, n - 1 - c
    if transform == "rot270":
        return n - 1 - c, r
    if transform == "flip_lr":
        return r, n - 1 - c
    if transform == "flip_ud":
        return n - 1 - r, c
    if transform == "flip_diag_main":
        return c, r
    if transform == "flip_diag_anti":
        return n - 1 - c, n - 1 - r
    raise ValueError(f"Unknown transform: {transform}")


def _get_action_index_map(board_size, transform):
    key = (board_size, transform)
    cached = _ACTION_INDEX_MAP.get(key)
    if cached is not None:
        return cached

    n = board_size
    rows, cols = np.indices((n, n))
    if transform == "identity":
        nr, nc = rows, cols
    elif transform == "rot90":
        nr, nc = cols, n - 1 - rows
    elif transform == "rot180":
        nr, nc = n - 1 - rows, n - 1 - cols
    elif transform == "rot270":
        nr, nc = n - 1 - cols, rows
    elif transform == "flip_lr":
        nr, nc = rows, n - 1 - cols
    elif transform == "flip_ud":
        nr, nc = n - 1 - rows, cols
    elif transform == "flip_diag_main":
        nr, nc = cols, rows
    elif transform == "flip_diag_anti":
        nr, nc = n - 1 - cols, n - 1 - rows
    else:
        raise ValueError(f"Unknown transform: {transform}")

    new_idx = (nr * n + nc).astype(np.int64).ravel()
    _ACTION_INDEX_MAP[key] = new_idx
    return new_idx


_INVERSE_TRANSFORM = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_lr": "flip_lr",
    "flip_ud": "flip_ud",
    "flip_diag_main": "flip_diag_main",
    "flip_diag_anti": "flip_diag_anti",
}


def inverse_transform(transform):
    inv = _INVERSE_TRANSFORM.get(transform)
    if inv is None:
        raise ValueError(f"Unknown transform: {transform}")
    return inv


def transform_action_index(action, board_size, transform):
    if action is None:
        return None
    pass_action = board_size * board_size
    if action == pass_action:
        return pass_action
    if action < 0:
        return action
    new_idx = _get_action_index_map(board_size, transform)
    return int(new_idx[int(action)])

def transform_action_probs(action_probs, board_size, transform):
    """Transform action probabilities with the same symmetry as the board."""
    expected = board_size * board_size + 1
    if action_probs.shape[0] != expected:
        raise ValueError(f"Invalid action_probs size: {action_probs.shape[0]} != {expected}")

    new_probs = np.zeros_like(action_probs)
    pass_action = board_size * board_size
    new_probs[pass_action] = action_probs[pass_action]

    new_idx = _get_action_index_map(board_size, transform)
    new_probs[new_idx] = action_probs[:pass_action]
    return new_probs


def invert_action_probs(action_probs, board_size, transform):
    inv = inverse_transform(transform)
    return transform_action_probs(action_probs, board_size, inv)


def _normalize_last_moves(last_moves):
    if last_moves is None:
        return (-1, -1)
    if isinstance(last_moves, (list, tuple)):
        lm0 = last_moves[0] if len(last_moves) > 0 else None
        lm1 = last_moves[1] if len(last_moves) > 1 else None
        return (int(lm0) if lm0 is not None else -1, int(lm1) if lm1 is not None else -1)
    return (int(last_moves), -1)


def _transform_last_moves(last_moves, board_size, transform):
    lm0, lm1 = _normalize_last_moves(last_moves)
    lm0_t = transform_action_index(lm0, board_size, transform) if lm0 >= 0 else lm0
    lm1_t = transform_action_index(lm1, board_size, transform) if lm1 >= 0 else lm1
    return (lm0_t, lm1_t)


def transform_last_moves(last_moves, board_size, transform):
    return _transform_last_moves(last_moves, board_size, transform)


def canonicalize_state(board, player, passes, last_moves, board_size):
    """Return canonical (board, last_moves) and transform id based on lexicographic key."""
    best_key = None
    best_board = None
    best_last_moves = None
    best_transform = "identity"

    for transform in CANON_TRANSFORMS:
        t_board = _apply_transform_array(board, transform)
        t_last_moves = _transform_last_moves(last_moves, board_size, transform)
        board_bytes = t_board.tobytes()
        meta = struct.pack(
            "<b b h h",
            int(player),
            int(passes),
            int(t_last_moves[0]),
            int(t_last_moves[1])
        )
        h = hashlib.blake2b(digest_size=32)
        h.update(board_bytes)
        h.update(meta)
        key = h.digest()
        if best_key is None or key < best_key:
            best_key = key
            best_board = t_board
            best_last_moves = t_last_moves
            best_transform = transform

    return best_board, best_last_moves, best_transform, best_key


def canonicalize_encoded_state(encoded_state):
    """Canonicalize encoded (C,H,W) state for caching. Returns (state, transform, key)."""
    best_key = None
    best_state = None
    best_transform = "identity"
    for transform in CANON_TRANSFORMS:
        t_state = _apply_transform_array(encoded_state, transform)
        h = hashlib.blake2b(digest_size=32)
        h.update(t_state.tobytes())
        key = h.digest()
        if best_key is None or key < best_key:
            best_key = key
            best_state = t_state
            best_transform = transform
    return best_state, best_transform, best_key


def augment_sample(encoded_state, action_probs, value, board_size, ownership_target=None):
    """Return original + 7 symmetric samples."""
    if encoded_state.ndim != 3 or encoded_state.shape[1] != board_size or encoded_state.shape[2] != board_size:
        raise ValueError(f"Invalid encoded_state shape: {encoded_state.shape}")
    if ownership_target is not None and ownership_target.shape != (2, board_size, board_size):
        raise ValueError(f"Invalid ownership_target shape: {ownership_target.shape}")

    samples = [(encoded_state, action_probs, value, ownership_target)]
    for transform in TRANSFORMS:
        t_state = _apply_transform_array(encoded_state, transform)
        t_state = np.ascontiguousarray(t_state)
        t_probs = transform_action_probs(action_probs, board_size, transform)
        if ownership_target is not None:
            t_owner = _apply_transform_array(ownership_target, transform)
            t_owner = np.ascontiguousarray(t_owner)
        else:
            t_owner = None
        samples.append((t_state, t_probs, value, t_owner))
    return samples


def debug_symmetry_transforms():
    """Debug helper: apply transforms to a small board and print results."""
    board_size = 5
    env = GreatKingdomEnv(board_size=board_size)
    env.reset()

    moves = [(0, 0), (1, 2), (3, 4)]
    for r, c in moves:
        env.step(r * board_size + c)

    board = env.board.copy()
    print("Original board:")
    print(board)
    print()

    for name in TRANSFORMS:
        t_board = _apply_transform_array(board, name)
        print(f"{name}:")
        print(t_board)
        print()

    # Policy transform check with a single hot action.
    action_probs = np.zeros(board_size * board_size + 1, dtype=np.float32)
    test_action = 0 * board_size + 1
    action_probs[test_action] = 1.0
    print(f"Policy test action: {test_action} -> (0, 1)")
    for name in TRANSFORMS:
        t_probs = transform_action_probs(action_probs, board_size, name)
        new_idx = int(np.argmax(t_probs))
        nr, nc = divmod(new_idx, board_size)
        print(f"{name}: {new_idx} -> ({nr}, {nc})")
    print("Debug done.")


if __name__ == "__main__":
    debug_symmetry_transforms()
