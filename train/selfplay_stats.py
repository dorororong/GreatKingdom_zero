import argparse
import os
import sys
import multiprocessing as mp
import json
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch

from env.env import GreatKingdomEnv
from env.env_fast import GreatKingdomEnvFast
from network import AlphaZeroNetwork, infer_head_type_from_state_dict
from mcts_alphazero import AlphaZeroMCTS
from train.selfplay_workers import BatchInferenceServer, AsyncNetworkClient


def _infer_network_config(checkpoint):
    num_res_blocks = checkpoint.get("num_res_blocks", 3)
    num_channels = checkpoint.get("num_channels", 64)
    use_liberty_features = checkpoint.get("use_liberty_features")
    liberty_bins = checkpoint.get("liberty_bins")
    use_last_moves = checkpoint.get("use_last_moves")

    if "num_res_blocks" not in checkpoint or "num_channels" not in checkpoint:
        policy_weight = checkpoint["network"].get("policy_conv.weight")
        if policy_weight is not None:
            num_channels = policy_weight.shape[1]
        res_block_keys = [
            k for k in checkpoint["network"].keys()
            if k.startswith("res_blocks.") and k.endswith(".conv1.weight")
        ]
        if res_block_keys:
            num_res_blocks = len(res_block_keys)

    if use_liberty_features is None or liberty_bins is None or use_last_moves is None:
        conv_weight = checkpoint["network"].get("conv_input.weight")
        if conv_weight is None:
            raise RuntimeError("Checkpoint missing conv_input.weight for feature inference.")
        in_channels = conv_weight.shape[1]
        if in_channels == 4:
            use_liberty_features = False
            liberty_bins = 2
            use_last_moves = False
        elif in_channels == 6:
            use_liberty_features = True
            liberty_bins = 1
            use_last_moves = False
        elif in_channels == 8:
            use_liberty_features = True
            liberty_bins = 2
            use_last_moves = False
        elif in_channels == 10:
            use_liberty_features = True
            liberty_bins = 3
            use_last_moves = False
        elif in_channels == 12:
            use_liberty_features = True
            liberty_bins = 3
            use_last_moves = True
        else:
            raise RuntimeError(f"Unsupported input channels in checkpoint: {in_channels}")

    return {
        "num_res_blocks": num_res_blocks,
        "num_channels": num_channels,
        "use_liberty_features": use_liberty_features,
        "liberty_bins": liberty_bins,
        "use_last_moves": use_last_moves,
        "network_head": infer_head_type_from_state_dict(checkpoint.get("network", {})),
    }


def _init_worker_async(
    worker_id,
    input_queue,
    output_queue,
    board_size,
    num_res_blocks,
    num_channels,
    num_simulations,
    c_puct,
    dirichlet_alpha,
    dirichlet_epsilon,
    center_wall,
    komi,
    use_fast_env,
    use_liberty_features,
    liberty_bins,
    use_last_moves,
    mcts_eval_batch_size,
    temperature,
):
    global _worker_env, _worker_mcts, _worker_use_last_moves, _worker_temperature

    env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
    _worker_env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)
    _worker_use_last_moves = use_last_moves
    _worker_temperature = temperature

    async_net = AsyncNetworkClient(input_queue, output_queue, worker_id)
    _worker_mcts = AlphaZeroMCTS(
        async_net,
        _worker_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=mcts_eval_batch_size,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves,
    )


def _classify_result(info, env):
    result = info.get("result", "")
    if result == "Win by Capture":
        return "capture"
    if result == "Loss by Suicide":
        return "suicide"
    if result == "Territory Count":
        return "territory"
    if result.startswith("Draw"):
        return "draw"
    return "other"


def _winner_from_info(info, env):
    result = info.get("result", "")
    if result == "Territory Count":
        winner_label = info.get("winner", "Draw")
        if winner_label == "Black":
            return 1
        if winner_label == "White":
            return 2
        return 0
    if result == "Win by Capture":
        return env.current_player
    if result == "Loss by Suicide":
        return 2 if env.current_player == 1 else 1
    return 0


def _worker_loop(
    game_queue,
    result_queue,
    worker_id,
    input_queue,
    output_queue,
    board_size,
    num_res_blocks,
    num_channels,
    num_simulations,
    c_puct,
    dirichlet_alpha,
    dirichlet_epsilon,
    center_wall,
    use_fast_env,
    use_liberty_features,
    liberty_bins,
    use_last_moves,
    mcts_eval_batch_size,
    temperature,
):
    _init_worker_async(
        worker_id,
        input_queue,
        output_queue,
        board_size,
        num_res_blocks,
        num_channels,
        num_simulations,
        c_puct,
        dirichlet_alpha,
        dirichlet_epsilon,
        center_wall,
        komi,
        use_fast_env,
        use_liberty_features,
        liberty_bins,
        use_last_moves,
        mcts_eval_batch_size,
        temperature,
    )
    global _worker_env, _worker_mcts, _worker_use_last_moves, _worker_temperature

    while True:
        item = game_queue.get()
        if item is None:
            break

        _worker_env.reset()
        done = False
        moves = 0
        last_moves = (None, None) if _worker_use_last_moves else None
        info = {}

        while not done and moves < 512:
            if _worker_use_last_moves:
                state = (
                    _worker_env.board.copy(),
                    _worker_env.current_player,
                    _worker_env.consecutive_passes,
                    last_moves,
                )
            else:
                state = (
                    _worker_env.board.copy(),
                    _worker_env.current_player,
                    _worker_env.consecutive_passes,
                )

            action, _ = _worker_mcts.run(state, temperature=_worker_temperature, add_root_noise=False)
            _, _, done, info = _worker_env.step(action)

            if _worker_use_last_moves and action != _worker_env.pass_action:
                last_moves = (action, last_moves[0])
            moves += 1

        win_type = _classify_result(info, _worker_env)
        winner = _winner_from_info(info, _worker_env)
        territory_diff = None
        if win_type == "territory":
            black = info.get("black_territory", 0)
            white = info.get("white_territory", 0)
            territory_diff = abs(int(black) - int(white))

        result_queue.put({
            "win_type": win_type,
            "winner": winner,
            "moves": moves,
            "territory_diff": territory_diff,
        })


def _summarize_results(results):
    stats = {
        "capture": {1: {"count": 0, "moves": 0}, 2: {"count": 0, "moves": 0}},
        "territory": {
            1: {"count": 0, "moves": 0, "diff": 0},
            2: {"count": 0, "moves": 0, "diff": 0},
        },
        "suicide": {1: 0, 2: 0},
        "draws": 0,
        "other": 0,
    }

    for r in results:
        win_type = r["win_type"]
        winner = r["winner"]
        moves = r["moves"]
        diff = r["territory_diff"]

        if win_type == "capture":
            if winner in (1, 2):
                stats["capture"][winner]["count"] += 1
                stats["capture"][winner]["moves"] += moves
        elif win_type == "territory":
            if winner in (1, 2):
                stats["territory"][winner]["count"] += 1
                stats["territory"][winner]["moves"] += moves
                stats["territory"][winner]["diff"] += diff or 0
        elif win_type == "suicide":
            if winner in (1, 2):
                stats["suicide"][winner] += 1
        elif win_type == "draw":
            stats["draws"] += 1
        else:
            stats["other"] += 1

    return stats


def _print_summary(label, stats):
    def _avg(total, count):
        return (total / count) if count else 0.0

    print(f"\nModel: {label}")
    for color, name in ((1, "Black"), (2, "White")):
        c = stats["capture"][color]["count"]
        m = stats["capture"][color]["moves"]
        print(f"  Capture {name}: {c} | Avg Moves: {_avg(m, c):.2f}")
    for color, name in ((1, "Black"), (2, "White")):
        c = stats["territory"][color]["count"]
        m = stats["territory"][color]["moves"]
        d = stats["territory"][color]["diff"]
        print(
            f"  Territory {name}: {c} | Avg Diff: {_avg(d, c):.2f} | Avg Moves: {_avg(m, c):.2f}"
        )
    if stats["suicide"][1] or stats["suicide"][2]:
        print(f"  Suicide Wins - Black: {stats['suicide'][1]}, White: {stats['suicide'][2]}")
    if stats["draws"] or stats["other"]:
        print(f"  Draws: {stats['draws']} | Other: {stats['other']}")


def run_for_checkpoint(
    checkpoint_path,
    num_games,
    num_workers,
    board_size,
    center_wall,
    komi,
    use_fast_env,
    num_simulations,
    c_puct,
    dirichlet_alpha,
    dirichlet_epsilon,
    infer_batch_size,
    infer_timeout,
    mcts_eval_batch_size,
    temperature,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net_cfg = _infer_network_config(checkpoint)
    if komi is None:
        komi = checkpoint.get("komi", 0)

    network_state_dict = {k: v.cpu() for k, v in checkpoint["network"].items()}
    server = BatchInferenceServer(
        network_state_dict=network_state_dict,
        board_size=board_size,
        num_res_blocks=net_cfg["num_res_blocks"],
        num_channels=net_cfg["num_channels"],
        batch_size=infer_batch_size,
        timeout=infer_timeout,
        use_liberty_features=net_cfg["use_liberty_features"],
        liberty_bins=net_cfg["liberty_bins"],
        use_last_moves=net_cfg["use_last_moves"],
        network_head=net_cfg.get("network_head", "fc"),
    )
    output_queues = [mp.Queue(maxsize=1000) for _ in range(num_workers)]
    server.output_queues = output_queues

    server_process = mp.Process(target=server.run)
    server_process.start()

    game_queue = mp.Queue()
    result_queue = mp.Queue()

    game_workers = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=_worker_loop,
            args=(
                game_queue,
                result_queue,
                worker_id,
                server.input_queue,
                output_queues[worker_id],
                board_size,
                net_cfg["num_res_blocks"],
                net_cfg["num_channels"],
                num_simulations,
                c_puct,
                dirichlet_alpha,
                dirichlet_epsilon,
                center_wall,
                komi,
                use_fast_env,
                net_cfg["use_liberty_features"],
                net_cfg["liberty_bins"],
                net_cfg["use_last_moves"],
                mcts_eval_batch_size,
                temperature,
            ),
        )
        p.start()
        game_workers.append(p)

    for _ in range(num_games):
        game_queue.put(1)

    results = []
    for _ in range(num_games):
        results.append(result_queue.get())

    for _ in game_workers:
        game_queue.put(None)
    for p in game_workers:
        p.join()

    server.input_queue.put(None)
    server_process.join()

    return results


def main():
    mp.freeze_support()

    parser = argparse.ArgumentParser(description="Self-play stats for multiple checkpoints.")
    parser.add_argument("--model-dir", type=str, default="checkpoints/board_7/center_wall_on")
    parser.add_argument("--iterations", type=str, default="150,155,160,165,170")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--board-size", type=int, default=7)
    parser.add_argument("--center-wall", action="store_true", default=True)
    parser.add_argument("--komi", type=float, default=None)
    parser.add_argument("--use-fast-env", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=11)
    parser.add_argument("--num-simulations", type=int, default=300)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    parser.add_argument("--infer-timeout", type=float, default=0.001)
    parser.add_argument("--mcts-eval-batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--out-dir", type=str, default="logs")
    args = parser.parse_args()

    iterations = [int(x.strip()) for x in args.iterations.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out_dir, f"selfplay_stats_{stamp}.csv")
    json_path = os.path.join(args.out_dir, f"selfplay_stats_{stamp}.json")
    csv_lines = [
        "iteration,win_type,winner,count,avg_moves,avg_diff"
    ]
    json_summary = {
        "model_dir": args.model_dir,
        "iterations": iterations,
        "num_games": args.games,
        "num_simulations": args.num_simulations,
        "c_puct": args.c_puct,
        "dirichlet_alpha": args.dirichlet_alpha,
        "dirichlet_epsilon": args.dirichlet_epsilon,
        "infer_batch_size": args.infer_batch_size,
        "infer_timeout": args.infer_timeout,
        "mcts_eval_batch_size": args.mcts_eval_batch_size,
        "temperature": args.temperature,
        "results": {},
    }
    for it in iterations:
        ckpt_name = f"alphazero_iter{it}.pt"
        ckpt_path = os.path.join(args.model_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Missing checkpoint: {ckpt_path}")
            continue

        results = run_for_checkpoint(
            ckpt_path,
            num_games=args.games,
            num_workers=args.num_workers,
            board_size=args.board_size,
            center_wall=args.center_wall,
            komi=args.komi,
            use_fast_env=args.use_fast_env,
            num_simulations=args.num_simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
            infer_batch_size=args.infer_batch_size,
            infer_timeout=args.infer_timeout,
            mcts_eval_batch_size=args.mcts_eval_batch_size,
            temperature=args.temperature,
        )
        stats = _summarize_results(results)
        _print_summary(f"iter {it}", stats)
        json_summary["results"][str(it)] = stats
        for win_type in ("capture", "territory"):
            for color, name in ((1, "Black"), (2, "White")):
                entry = stats[win_type][color]
                count = entry["count"]
                avg_moves = (entry["moves"] / count) if count else 0.0
                avg_diff = (entry.get("diff", 0) / count) if count else 0.0
                csv_lines.append(
                    f"{it},{win_type},{name},{count},{avg_moves:.2f},{avg_diff:.2f}"
                )
        csv_lines.append(f"{it},suicide,Black,{stats['suicide'][1]},0.00,0.00")
        csv_lines.append(f"{it},suicide,White,{stats['suicide'][2]},0.00,0.00")
        csv_lines.append(f"{it},draw,Draw,{stats['draws']},0.00,0.00")
        csv_lines.append(f"{it},other,Other,{stats['other']},0.00,0.00")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
