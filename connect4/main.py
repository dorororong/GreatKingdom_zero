import argparse
import os
import sys
import multiprocessing as mp

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from connect4.config import list_profiles, get_profile
from connect4.train_core import train_alphazero


def main():
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="Connect4 AlphaZero Training")
    parser.add_argument(
        "--profile",
        type=str,
        default="c4_standard",
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
        rows=profile["rows"],
        cols=profile["cols"],
        num_res_blocks=profile["num_res_blocks"],
        num_channels=profile["num_channels"],
        batch_size=profile["batch_size"],
        buffer_size=profile["buffer_size"],
        checkpoint_dir=profile["checkpoint_dir"],
        playout_full_prob=profile["playout_full_prob"],
        playout_full_cap=profile["playout_full_cap"],
        playout_fast_cap=profile["playout_fast_cap"],
        infer_batch_size=profile["infer_batch_size"],
        infer_timeout=profile["infer_timeout"],
        use_forced_playouts=profile["use_forced_playouts"],
        use_policy_target_pruning=profile["use_policy_target_pruning"],
        forced_playout_k=profile["forced_playout_k"],
        selfplay_workers=profile["selfplay_workers"],
        reanalyze_workers=profile["reanalyze_workers"],
        reanalyze_ratio=profile["reanalyze_ratio"],
        reanalyze_min_samples=profile["reanalyze_min_samples"],
        reanalyze_candidate_pool=profile["reanalyze_candidate_pool"],
        reanalyze_sims=profile["reanalyze_sims"],
        reanalyze_debug=profile["reanalyze_debug"],
        reanalyze_debug_top_k=profile["reanalyze_debug_top_k"]
    )


if __name__ == "__main__":
    main()
