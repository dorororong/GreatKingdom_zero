import argparse
import json
import os
import sys
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from connect4.env import Connect4Env
from connect4.network import AlphaZeroNetwork
from connect4.mcts_alphazero import AlphaZeroMCTS
from connect4.mcts import MCTS


def _play_one_game_az_vs_mcts(env, az_mcts, az_player, mcts_sims):
    """AlphaZero vs Pure MCTS 대결"""
    env.reset()
    done = False
    moves = 0
    last_action = None
    states = [{
        "board": env.board.copy().tolist(),
        "current_player": int(env.current_player),
        "consecutive_passes": int(env.consecutive_passes)
    }]

    pure_mcts = MCTS(env, simulations_per_move=mcts_sims)

    while not done and moves < (env.rows * env.cols):
        state = (env.board.copy(), env.current_player, env.consecutive_passes)
        if env.current_player == az_player:
            az_mcts.num_simulations = az_mcts.num_simulations
            action, _ = az_mcts.run(state, temperature=0.0, add_root_noise=False)
        else:
            action = pure_mcts.run(state, opponent_last_action=last_action)

        _, reward, done, info = env.step(action)
        moves += 1
        last_action = action
        states.append({
            "board": env.board.copy().tolist(),
            "current_player": int(env.current_player),
            "consecutive_passes": int(env.consecutive_passes)
        })

    winner = info.get("winner", 0) if done else 0
    return winner, moves, states


def _play_one_game_az_vs_az(env, az_mcts_1, az_mcts_2, az1_player):
    """AlphaZero vs AlphaZero 대결"""
    env.reset()
    done = False
    moves = 0
    states = [{
        "board": env.board.copy().tolist(),
        "current_player": int(env.current_player),
        "consecutive_passes": int(env.consecutive_passes)
    }]

    while not done and moves < (env.rows * env.cols):
        state = (env.board.copy(), env.current_player, env.consecutive_passes)
        if env.current_player == az1_player:
            action, _ = az_mcts_1.run(state, temperature=0.0, add_root_noise=False)
        else:
            action, _ = az_mcts_2.run(state, temperature=0.0, add_root_noise=False)

        _, reward, done, info = env.step(action)
        moves += 1
        states.append({
            "board": env.board.copy().tolist(),
            "current_player": int(env.current_player),
            "consecutive_passes": int(env.consecutive_passes)
        })

    winner = info.get("winner", 0) if done else 0
    return winner, moves, states


def run_benchmark(
    games=10,
    rows=6,
    cols=7,
    az_sims=200,
    mcts_sims=2000,
    az_ckpt="connect4/checkpoints/alphazero_latest.pt",
    az_res_blocks=3,
    az_channels=64,
    device=None,
    save_path=None
):
    """AlphaZero vs Pure MCTS 벤치마크"""
    if not os.path.exists(az_ckpt):
        raise FileNotFoundError(f"AlphaZero checkpoint not found: {az_ckpt}")

    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    env = Connect4Env(rows=rows, cols=cols)
    network = AlphaZeroNetwork(
        rows=rows,
        cols=cols,
        num_res_blocks=az_res_blocks,
        num_channels=az_channels
    ).to(device)
    checkpoint = torch.load(az_ckpt, map_location=device)
    network.load_state_dict(checkpoint["network"])
    network.eval()

    az_mcts = AlphaZeroMCTS(
        network,
        env,
        num_simulations=az_sims,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25
    )

    records = []
    az_wins = 0
    mcts_wins = 0
    draws = 0

    for i in range(games):
        az_player = 1 if i % 2 == 0 else 2
        winner, moves, states = _play_one_game_az_vs_mcts(env, az_mcts, az_player, mcts_sims)
        if winner == 0:
            draws += 1
            result = "Draw"
        elif winner == az_player:
            az_wins += 1
            result = "AlphaZero"
        else:
            mcts_wins += 1
            result = "MCTS"

        record = {
            "game": i + 1,
            "az_player": az_player,
            "winner": winner,
            "moves": moves,
            "result": result,
            "states": states
        }
        records.append(record)
        print(f"Game {i + 1}/{games} | AZ as {'Red' if az_player == 1 else 'Yellow'} | Result: {result} | Moves: {moves}")

    total = max(1, games)
    stats = {
        "games": games,
        "az_wins": az_wins,
        "mcts_wins": mcts_wins,
        "draws": draws,
        "az_win_rate": az_wins / total,
        "mcts_win_rate": mcts_wins / total,
        "draw_rate": draws / total,
        "az_sims": az_sims,
        "mcts_sims": mcts_sims,
        "az_ckpt": az_ckpt
    }

    print("\n=== Summary ===")
    print(f"AlphaZero wins: {az_wins} | MCTS wins: {mcts_wins} | Draws: {draws}")
    print(f"AZ win rate: {stats['az_win_rate']*100:.1f}% | MCTS win rate: {stats['mcts_win_rate']*100:.1f}% | Draw rate: {stats['draw_rate']*100:.1f}%")

    if save_path:
        payload = {"stats": stats, "records": records}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        print(f"Saved results to {save_path}")

    return stats, records


def run_az_vs_az_benchmark(
    games=10,
    rows=6,
    cols=7,
    az1_sims=200,
    az2_sims=200,
    az1_ckpt="connect4/checkpoints/alphazero_latest.pt",
    az2_ckpt="connect4/checkpoints/alphazero_latest.pt",
    az1_res_blocks=3,
    az1_channels=64,
    az2_res_blocks=3,
    az2_channels=64,
    device=None,
    save_path=None
):
    """AlphaZero vs AlphaZero 벤치마크"""
    if not os.path.exists(az1_ckpt):
        raise FileNotFoundError(f"AlphaZero 1 checkpoint not found: {az1_ckpt}")
    if not os.path.exists(az2_ckpt):
        raise FileNotFoundError(f"AlphaZero 2 checkpoint not found: {az2_ckpt}")

    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    env = Connect4Env(rows=rows, cols=cols)

    # Load AlphaZero 1
    network1 = AlphaZeroNetwork(
        rows=rows,
        cols=cols,
        num_res_blocks=az1_res_blocks,
        num_channels=az1_channels
    ).to(device)
    checkpoint1 = torch.load(az1_ckpt, map_location=device)
    network1.load_state_dict(checkpoint1["network"])
    network1.eval()

    az_mcts_1 = AlphaZeroMCTS(
        network1,
        env,
        num_simulations=az1_sims,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25
    )

    # Load AlphaZero 2
    network2 = AlphaZeroNetwork(
        rows=rows,
        cols=cols,
        num_res_blocks=az2_res_blocks,
        num_channels=az2_channels
    ).to(device)
    checkpoint2 = torch.load(az2_ckpt, map_location=device)
    network2.load_state_dict(checkpoint2["network"])
    network2.eval()

    az_mcts_2 = AlphaZeroMCTS(
        network2,
        env,
        num_simulations=az2_sims,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25
    )

    records = []
    az1_wins = 0
    az2_wins = 0
    draws = 0

    az1_name = os.path.basename(az1_ckpt)
    az2_name = os.path.basename(az2_ckpt)

    for i in range(games):
        az1_player = 1 if i % 2 == 0 else 2
        winner, moves, states = _play_one_game_az_vs_az(env, az_mcts_1, az_mcts_2, az1_player)
        if winner == 0:
            draws += 1
            result = "Draw"
        elif winner == az1_player:
            az1_wins += 1
            result = f"AZ1 ({az1_name})"
        else:
            az2_wins += 1
            result = f"AZ2 ({az2_name})"

        record = {
            "game": i + 1,
            "az1_player": az1_player,
            "winner": winner,
            "moves": moves,
            "result": result,
            "states": states
        }
        records.append(record)
        print(f"Game {i + 1}/{games} | AZ1 as {'Red' if az1_player == 1 else 'Yellow'} | Result: {result} | Moves: {moves}")

    total = max(1, games)
    stats = {
        "games": games,
        "az1_wins": az1_wins,
        "az2_wins": az2_wins,
        "draws": draws,
        "az1_win_rate": az1_wins / total,
        "az2_win_rate": az2_wins / total,
        "draw_rate": draws / total,
        "az1_sims": az1_sims,
        "az2_sims": az2_sims,
        "az1_ckpt": az1_ckpt,
        "az2_ckpt": az2_ckpt
    }

    print("\n=== Summary ===")
    print(f"AZ1 ({az1_name}) wins: {az1_wins} | AZ2 ({az2_name}) wins: {az2_wins} | Draws: {draws}")
    print(f"AZ1 win rate: {stats['az1_win_rate']*100:.1f}% | AZ2 win rate: {stats['az2_win_rate']*100:.1f}% | Draw rate: {stats['draw_rate']*100:.1f}%")

    if save_path:
        payload = {"stats": stats, "records": records}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        print(f"Saved results to {save_path}")

    return stats, records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaZero vs MCTS or AlphaZero vs AlphaZero evaluation (Connect4)")
    parser.add_argument("--mode", type=str, default="az_vs_mcts", choices=["az_vs_mcts", "az_vs_az"],
                        help="Test mode: az_vs_mcts or az_vs_az")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    
    # AlphaZero 1 (or single AZ for az_vs_mcts mode)
    parser.add_argument("--az_sims", type=int, default=500, help="AZ1 simulations (or AZ sims for az_vs_mcts)")
    parser.add_argument("--az_ckpt", type=str, default="connect4/checkpoints/alphazero_iter200.pt",
                        help="AZ1 checkpoint path (or AZ for az_vs_mcts)")
    parser.add_argument("--az_res_blocks", type=int, default=3, help="AZ1 residual blocks")
    parser.add_argument("--az_channels", type=int, default=64, help="AZ1 channels")
    
    # AlphaZero 2 (for az_vs_az mode)
    parser.add_argument("--az2_sims", type=int, default=500, help="AZ2 simulations")
    parser.add_argument("--az2_ckpt", type=str, default="connect4/checkpoints/alphazero_iter70_reanalyze_ver.pt",
                        help="AZ2 checkpoint path (required for az_vs_az mode)")
    parser.add_argument("--az2_res_blocks", type=int, default=3, help="AZ2 residual blocks")
    parser.add_argument("--az2_channels", type=int, default=64, help="AZ2 channels")
    
    # Pure MCTS (for az_vs_mcts mode)
    parser.add_argument("--mcts_sims", type=int, default=5000, help="Pure MCTS simulations")
    
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    if args.mode == "az_vs_mcts":
        run_benchmark(
            games=args.games,
            rows=args.rows,
            cols=args.cols,
            az_sims=args.az_sims,
            mcts_sims=args.mcts_sims,
            az_ckpt=args.az_ckpt,
            az_res_blocks=args.az_res_blocks,
            az_channels=args.az_channels,
            device=args.device,
            save_path=args.save
        )
    elif args.mode == "az_vs_az":
        if args.az2_ckpt is None:
            parser.error("--az2_ckpt is required for az_vs_az mode")
        run_az_vs_az_benchmark(
            games=args.games,
            rows=args.rows,
            cols=args.cols,
            az1_sims=args.az_sims,
            az2_sims=args.az2_sims,
            az1_ckpt=args.az_ckpt,
            az2_ckpt=args.az2_ckpt,
            az1_res_blocks=args.az_res_blocks,
            az1_channels=args.az_channels,
            az2_res_blocks=args.az2_res_blocks,
            az2_channels=args.az2_channels,
            device=args.device,
            save_path=args.save
        )
