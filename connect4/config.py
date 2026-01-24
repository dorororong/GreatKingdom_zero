CONFIG_PROFILES = {
    "c4_standard": {
        "rows": 6,
        "cols": 7,
        "num_res_blocks": 3,
        "num_channels": 64,
        "num_simulations": 100,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "playout_full_prob": 0.5,
        "playout_full_cap": 120,
        "playout_fast_cap": 20,
        "infer_batch_size": 32,
        "infer_timeout": 0.002,
        "use_forced_playouts": False,
        "use_policy_target_pruning": True,
        "forced_playout_k": 2.0,
        "selfplay_workers": 10,
        "reanalyze_workers": 10,
        "reanalyze_ratio": 0.1,
        "reanalyze_min_samples": 0,
        "reanalyze_candidate_pool": 2000,
        "reanalyze_sims": None,
        "reanalyze_debug": False,
        "reanalyze_debug_top_k": 5,
        "num_iterations": 100,
        "games_per_iteration": 400,
        "batches_per_iteration": 600,
        "eval_games": 20,
        "save_interval": 5,
        "temperature_threshold": 7,
        "fresh_start": False,
        "batch_size": 128,
        "buffer_size": 100000,
        "checkpoint_dir": "connect4/checkpoints"
    }
}


def list_profiles():
    return sorted(CONFIG_PROFILES.keys())


def get_profile(profile_name):
    if profile_name not in CONFIG_PROFILES:
        raise KeyError(f"Unknown profile: {profile_name}")
    return dict(CONFIG_PROFILES[profile_name])
