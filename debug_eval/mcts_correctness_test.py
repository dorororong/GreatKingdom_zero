import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.env import GreatKingdomEnv
from mcts_alphazero import AlphaZeroMCTS, AlphaZeroNode


class _DummyNet:
    def predict(self, encoded_state):
        action_space = encoded_state.shape[1] * encoded_state.shape[2] + 1
        policy = np.ones(action_space, dtype=np.float32) / action_space
        value = 0.0
        return policy, value

    def predict_batch(self, state_tensors):
        action_space = state_tensors.shape[2] * state_tensors.shape[3] + 1
        batch = state_tensors.shape[0]
        policy = np.ones((batch, action_space), dtype=np.float32) / action_space
        value = np.zeros((batch,), dtype=np.float32)
        return policy, value


def _build_random_node(num_children=10):
    parent = AlphaZeroNode(prior=0.0)
    parent.visit_count = np.random.randint(1, 200)
    parent.player = 1
    for action in range(num_children):
        child = AlphaZeroNode(prior=np.random.random())
        child.visit_count = np.random.randint(0, 200)
        child.value_sum = np.random.uniform(-10, 10)
        child.player = 1 if np.random.random() < 0.5 else 2
        parent.children[action] = child
    return parent


def test_select_child_matches():
    env = GreatKingdomEnv(board_size=5, center_wall=False)
    net = _DummyNet()
    mcts = AlphaZeroMCTS(net, env, use_vector_puct=True)

    for seed in range(50):
        np.random.seed(seed)
        node = _build_random_node(num_children=20)
        np.random.seed(seed)
        a_vec, _ = mcts._select_child_vector(node)
        np.random.seed(seed)
        a_py, _ = mcts._select_child_py(node)
        if a_vec != a_py:
            raise AssertionError(f"Select mismatch at seed {seed}: vec={a_vec}, py={a_py}")


def test_set_env_state_copy_isolation():
    env = GreatKingdomEnv(board_size=5, center_wall=False)
    net = _DummyNet()
    mcts = AlphaZeroMCTS(net, env)

    board = env.board.copy()
    board[0, 0] = 1
    state = (board, env.current_player, env.consecutive_passes, (None, None))
    mcts._set_env_state(env, state)

    env.board[0, 0] = 2
    if board[0, 0] != 1:
        raise AssertionError("Board aliasing detected in _set_env_state")


if __name__ == "__main__":
    test_select_child_matches()
    test_set_env_state_copy_isolation()
    print("OK: MCTS correctness checks passed.")
