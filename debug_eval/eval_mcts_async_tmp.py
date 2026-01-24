import os
import sys
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train.train_core import AlphaZeroTrainer


def main():
    trainer = AlphaZeroTrainer(
        board_size=4,
        center_wall=False,
        use_fast_env=True,
        num_res_blocks=2,
        num_channels=48,
        num_simulations=80,
        c_puct=2.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        playout_full_prob=0.3,
        playout_full_cap=100,
        playout_fast_cap=20,
        infer_batch_size=64,
        infer_timeout=0.001,
        use_forced_playouts=True,
        use_policy_target_pruning=True,
        forced_playout_k=2.0,
        use_liberty_features=True,
        liberty_bins=1,
        use_last_moves=False
    )

    checkpoint_path = "checkpoints/board_4/center_wall_off/alphazero_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer.network.load_state_dict(checkpoint["network"])
    trainer.network.eval()

    win_rate, _ = trainer.evaluate_vs_mcts_async(num_games=20, mcts_sims=100)
    print(f"Eval vs MCTS100 (async): {win_rate*100:.1f}%")


if __name__ == "__main__":
    main()
