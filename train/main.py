import argparse
import os
import sys
import multiprocessing as mp

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.env import GreatKingdomEnv
from env.env_fast import GreatKingdomEnvFast
from network import AlphaZeroNetwork
from mcts_alphazero import AlphaZeroMCTS
from symmetry import debug_symmetry_transforms

from train.config import list_profiles, get_profile
from train.train_core import train_alphazero


def main():
    mp.freeze_support()

    if os.environ.get("AUGMENT_DEBUG") == "1":
        debug_symmetry_transforms()
        raise SystemExit(0)

    parser = argparse.ArgumentParser(description="AlphaZero Training")
    parser.add_argument(
        "--profile",
        type=str,
        default="5x5_center_off",
        choices=list_profiles(),
        help="Training profile name"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit"
    )
    args = parser.parse_args()

    if args.list_profiles:
        for name in list_profiles():
            print(name)
        raise SystemExit(0)

    profile = get_profile(args.profile)
    mcts_eval_batch_size = profile.get("mcts_eval_batch_size", 1)
    mcts_profile = profile.get("mcts_profile", False)
    mcts_profile_every = profile.get("mcts_profile_every", 0)
    temperature_schedule = profile.get("temperature_schedule")
    eval_infer_batch_size = profile.get("eval_infer_batch_size", profile.get("infer_batch_size", 64))
    eval_infer_timeout = profile.get("eval_infer_timeout", profile.get("infer_timeout", 0.01))
    komi = profile.get("komi", 0)
    network_head = profile.get("network_head", "fc")
    freeze_backbone = profile.get("freeze_backbone", False)
    freeze_backbone_blocks = profile.get("freeze_backbone_blocks")
    freeze_backbone_input = profile.get("freeze_backbone_input", True)
    init_checkpoint = profile.get("init_checkpoint")
    benchmark_interval = profile.get("benchmark_interval", 10)
    freeze_schedule = profile.get("freeze_schedule")

    # 간단 sanity check (프로필 설정 기준)
    env_cls = GreatKingdomEnvFast if profile.get("use_fast_env") else GreatKingdomEnv
    test_env = env_cls(
        board_size=profile["board_size"],
        center_wall=profile["center_wall"],
        komi=komi
    )
    test_net = AlphaZeroNetwork(
        board_size=profile["board_size"],
        num_res_blocks=profile["num_res_blocks"],
        num_channels=profile["num_channels"],
        use_liberty_features=profile["use_liberty_features"],
        liberty_bins=profile["liberty_bins"],
        use_last_moves=profile["use_last_moves"],
        head_type=network_head
    )
    test_mcts = AlphaZeroMCTS(
        test_net,
        test_env,
        num_simulations=2,
        c_puct=profile["c_puct"],
        eval_batch_size=mcts_eval_batch_size,
        use_liberty_features=profile["use_liberty_features"],
        liberty_bins=profile["liberty_bins"],
        use_last_moves=profile["use_last_moves"]
    )
    test_state = (test_env.board.copy(), test_env.current_player, test_env.consecutive_passes)
    test_action, test_probs = test_mcts.run(test_state, temperature=1.0)
    print(f"Sanity check: action={test_action}, probs_shape={test_probs.shape}")

    train_alphazero(
        num_iterations=profile["num_iterations"],
        games_per_iteration=profile["games_per_iteration"],
        batches_per_iteration=profile["batches_per_iteration"],
        eval_games=profile["eval_games"],
        num_simulations=profile["num_simulations"],
        save_interval=profile["save_interval"],
        temperature_threshold=profile["temperature_threshold"],
        c_puct=profile["c_puct"],
        dirichlet_alpha=profile["dirichlet_alpha"],
        dirichlet_epsilon=profile["dirichlet_epsilon"],
        fresh_start=profile["fresh_start"],
        board_size=profile["board_size"],
        num_res_blocks=profile["num_res_blocks"],
        num_channels=profile["num_channels"],
        batch_size=profile["batch_size"],
        buffer_size=profile["buffer_size"],
        checkpoint_dir=profile["checkpoint_dir"],
        center_wall=profile["center_wall"],
        komi=komi,
        playout_full_prob=profile["playout_full_prob"],
        playout_full_cap=profile["playout_full_cap"],
        playout_fast_cap=profile["playout_fast_cap"],
        infer_batch_size=profile["infer_batch_size"],
        infer_timeout=profile["infer_timeout"],
        eval_infer_batch_size=eval_infer_batch_size,
        eval_infer_timeout=eval_infer_timeout,
        mcts_eval_batch_size=mcts_eval_batch_size,
        mcts_profile=mcts_profile,
        mcts_profile_every=mcts_profile_every,
        temperature_schedule=temperature_schedule,
        use_forced_playouts=profile["use_forced_playouts"],
        use_policy_target_pruning=profile["use_policy_target_pruning"],
        forced_playout_k=profile["forced_playout_k"],
        use_fast_env=profile["use_fast_env"],
        use_liberty_features=profile["use_liberty_features"],
        liberty_bins=profile["liberty_bins"],
        use_last_moves=profile["use_last_moves"],
        network_head=network_head,
        freeze_backbone=freeze_backbone,
        freeze_backbone_blocks=freeze_backbone_blocks,
        freeze_backbone_input=freeze_backbone_input,
        init_checkpoint=init_checkpoint,
        benchmark_interval=benchmark_interval,
        freeze_schedule=freeze_schedule,
        selfplay_record_interval=profile.get("selfplay_record_interval", 0),
        selfplay_record_dir=profile.get("selfplay_record_dir", "logs/selfplay_records"),
        ownership_loss_weight=profile.get("ownership_loss_weight", 0.2),
        ownership_loss_weight_capture=profile.get("ownership_loss_weight_capture", 0.1)
    )


if __name__ == "__main__":
    main()
