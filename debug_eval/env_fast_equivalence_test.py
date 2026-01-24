import os
import sys
import random
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.env import GreatKingdomEnv
from env.env_fast import GreatKingdomEnvFast


def _compare_envs(env_a, env_b, step_idx):
    if not np.array_equal(env_a.board, env_b.board):
        raise AssertionError(f"Board mismatch at step {step_idx}")
    if env_a.current_player != env_b.current_player:
        raise AssertionError(f"Current player mismatch at step {step_idx}")
    if env_a.consecutive_passes != env_b.consecutive_passes:
        raise AssertionError(f"Pass counter mismatch at step {step_idx}")

    legal_a = env_a.get_legal_moves()
    legal_b = env_b.get_legal_moves()
    if not np.array_equal(legal_a, legal_b):
        raise AssertionError(f"Legal moves mismatch at step {step_idx}")

    for player in (1, 2):
        mask_a = env_a._get_territory_mask(player)
        mask_b = env_b._get_territory_mask(player)
        if not np.array_equal(mask_a, mask_b):
            raise AssertionError(f"Territory mask mismatch at step {step_idx}, player={player}")

    if env_a.get_territory_scores() != env_b.get_territory_scores():
        raise AssertionError(f"Territory scores mismatch at step {step_idx}")

    for player in (1, 2):
        if env_a.get_kill_moves(player) != env_b.get_kill_moves(player):
            raise AssertionError(f"Kill moves mismatch at step {step_idx}, player={player}")
        if env_a.get_kill_moves_bfs(player) != env_b.get_kill_moves_bfs(player):
            raise AssertionError(f"Kill moves BFS mismatch at step {step_idx}, player={player}")


def run_test(board_size=5, center_wall=True, seeds=20, max_steps=60):
    for seed in range(seeds):
        random.seed(seed)
        np.random.seed(seed)
        env_a = GreatKingdomEnv(board_size=board_size, center_wall=center_wall)
        env_b = GreatKingdomEnvFast(board_size=board_size, center_wall=center_wall)

        _compare_envs(env_a, env_b, step_idx=0)

        for step_idx in range(1, max_steps + 1):
            legal = env_a.get_legal_moves()
            legal_idx = np.where(legal)[0]
            action = int(np.random.choice(legal_idx))

            obs_a, reward_a, done_a, info_a = env_a.step(action)
            obs_b, reward_b, done_b, info_b = env_b.step(action)

            if reward_a != reward_b or done_a != done_b or info_a != info_b:
                raise AssertionError(f"Step result mismatch at step {step_idx}")

            _compare_envs(env_a, env_b, step_idx)

            if done_a:
                break


if __name__ == "__main__":
    run_test(board_size=5, center_wall=True, seeds=30, max_steps=80)
    run_test(board_size=5, center_wall=False, seeds=30, max_steps=80)
    run_test(board_size=6, center_wall=False, seeds=20, max_steps=80)
    print("OK: env_fast matches env on randomized tests.")
