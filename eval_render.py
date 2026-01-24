import os
import argparse
import time
import numpy as np
import multiprocessing as mp

from env.env import GreatKingdomEnv
from game_result import winner_label_from_step
from network import AlphaZeroNetwork
from mcts_alphazero import AlphaZeroMCTS
from mcts import MCTS


def load_checkpoint_torch_state(path="checkpoints/alphazero_latest.pt", device="cpu"):
    import torch
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint


def infer_network_config_from_checkpoint(checkpoint):
    state_dict = checkpoint["network"]
    conv_input_weight = state_dict["conv_input.weight"]
    num_channels = int(conv_input_weight.shape[0])

    res_block_indices = set()
    for key in state_dict.keys():
        if key.startswith("res_blocks."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                res_block_indices.add(int(parts[1]))
    num_res_blocks = max(res_block_indices) + 1 if res_block_indices else 0

    policy_fc_weight = state_dict["policy_fc.weight"]
    in_features = int(policy_fc_weight.shape[1])
    if in_features % 2 != 0:
        raise ValueError(f"Unexpected policy_fc input features: {in_features}")
    board_area = in_features // 2
    board_size = int(round(board_area ** 0.5))
    if board_size * board_size != board_area:
        raise ValueError(f"Cannot infer board_size from policy_fc input: {in_features}")
    action_space = int(policy_fc_weight.shape[0])
    if action_space != board_size * board_size + 1:
        raise ValueError(
            f"Checkpoint action space mismatch: {action_space} vs expected {board_size * board_size + 1}"
        )

    return board_size, num_res_blocks, num_channels


def load_network_from_checkpoint(path, device="cpu", board_size=None, num_res_blocks=None, num_channels=None):
    checkpoint = load_checkpoint_torch_state(path, device=device)
    inferred_board, inferred_blocks, inferred_channels = infer_network_config_from_checkpoint(checkpoint)

    if board_size is None:
        board_size = inferred_board
    if num_res_blocks is None:
        num_res_blocks = inferred_blocks
    if num_channels is None:
        num_channels = inferred_channels

    if board_size != inferred_board:
        raise ValueError(f"board_size mismatch: arg={board_size}, checkpoint={inferred_board}")
    if num_res_blocks != inferred_blocks:
        raise ValueError(f"num_res_blocks mismatch: arg={num_res_blocks}, checkpoint={inferred_blocks}")
    if num_channels != inferred_channels:
        raise ValueError(f"num_channels mismatch: arg={num_channels}, checkpoint={inferred_channels}")

    network = AlphaZeroNetwork(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels
    )
    network.load_state_dict(checkpoint["network"])
    network.eval()
    return network, checkpoint


def random_player(env):
    legal_moves = env.get_legal_moves()
    legal_indices = np.where(legal_moves)[0]
    non_pass = legal_indices[legal_indices != env.pass_action]
    if len(non_pass) > 0 and np.random.random() > 0.1:
        return np.random.choice(non_pass)
    return np.random.choice(legal_indices)


def mcts_player(env, mcts, temperature=0):
    state = (env.board.copy(), env.current_player, env.consecutive_passes)
    action, _ = mcts.run(state, temperature=temperature)
    return action


_WORKER_ENV = None
_WORKER_AZ_MCTS = None


def _init_eval_worker(network_state_dict, board_size, num_res_blocks, num_channels, num_simulations, c_puct, center_wall):
    global _WORKER_ENV, _WORKER_AZ_MCTS
    _WORKER_ENV = GreatKingdomEnv(board_size=board_size, center_wall=center_wall)
    network = AlphaZeroNetwork(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels
    )
    network.load_state_dict(network_state_dict)
    network.eval()
    _WORKER_AZ_MCTS = AlphaZeroMCTS(
        network,
        _WORKER_ENV,
        num_simulations=num_simulations,
        c_puct=c_puct
    )


def _play_one_game_worker(args):
    game_index, mcts_sims, az_is_black, max_moves = args
    env = _WORKER_ENV
    az_mcts = _WORKER_AZ_MCTS
    env.reset()

    pure_mcts = MCTS(env, simulations_per_move=mcts_sims)
    if az_is_black:
        black_fn = lambda e: mcts_player(e, az_mcts, temperature=0)
        white_fn = lambda e: pure_mcts.run((e.board.copy(), e.current_player, e.consecutive_passes))
    else:
        black_fn = lambda e: pure_mcts.run((e.board.copy(), e.current_player, e.consecutive_passes))
        white_fn = lambda e: mcts_player(e, az_mcts, temperature=0)

    winner, moves, duration = play_game(env, black_fn, white_fn, render=False, max_moves=max_moves)
    return game_index, winner, moves, duration, az_is_black


def play_game(env, black_fn, white_fn, render=True, max_moves=200):
    """한 게임 진행 후 (승자, 수, 소요시간) 반환"""
    env.reset()
    done = False
    moves = 0
    info = {}
    reward = 0

    start_time = time.time()

    if render:
        print("\n=== New Game ===")
        env.render()

    while not done and moves < max_moves:
        current_player = env.current_player
        if current_player == 1:
            action = black_fn(env)
        else:
            action = white_fn(env)

        _, reward, done, info = env.step(action)
        moves += 1

        if render:
            env.render()

    duration = time.time() - start_time
    winner = winner_label_from_step(info, reward, env.current_player)

    return winner, moves, duration


def run_series(env, black_fn, white_fn, num_games=3, label="Series", render=True):
    """일반 시리즈 진행 및 통계 출력"""
    results = {"Black": 0, "White": 0, "Draw": 0}
    total_moves = 0
    total_time = 0.0

    print(f"\n=== {label} ({num_games} games) ===")
    for i in range(num_games):
        winner, moves, duration = play_game(env, black_fn, white_fn, render=render)
        results[winner] += 1
        total_moves += moves
        total_time += duration
        print(f"Game {i+1}: Winner={winner}, Moves={moves}, Time={duration:.2f}s")

    avg_moves = total_moves / num_games
    avg_time = total_time / num_games
    print(f"Results: {results}")
    print(f"Avg Moves: {avg_moves:.1f}, Avg Time: {avg_time:.2f}s")
    return {
        "results": results,
        "avg_moves": avg_moves,
        "avg_time": avg_time
    }


def run_alphazero_vs_mcts(env, az_mcts, mcts_sims, num_games=20, render=False, num_workers=0, max_moves=200):
    """AlphaZero vs Pure MCTS ?????????? ??? ???"""
    results = {"Black": 0, "White": 0, "Draw": 0}
    total_moves = 0
    total_time = 0.0
    az_wins = 0

    print(f"=== AlphaZero({az_mcts.num_simulations} sims) vs Pure MCTS({mcts_sims} sims) ({num_games} games) ===")
    if num_workers and num_workers > 1:
        if render:
            print("Render disabled in parallel mode.")
        network_state = {k: v.cpu() for k, v in az_mcts.network.state_dict().items()}
        num_res_blocks = len(az_mcts.network.res_blocks)
        num_channels = az_mcts.network.conv_input.out_channels
        tasks = [(i, mcts_sims, i % 2 == 0, max_moves) for i in range(num_games)]

        with mp.Pool(
            processes=num_workers,
            initializer=_init_eval_worker,
            initargs=(
                network_state,
                env.board_size,
                num_res_blocks,
                num_channels,
                az_mcts.num_simulations,
                az_mcts.c_puct,
                env.center_wall
            )
        ) as pool:
            game_results = pool.map(_play_one_game_worker, tasks)

        game_results.sort(key=lambda item: item[0])
        for i, winner, moves, duration, az_is_black in game_results:
            results[winner] += 1
            total_moves += moves
            total_time += duration
            if (az_is_black and winner == "Black") or ((not az_is_black) and winner == "White"):
                az_wins += 1
            print(f"Game {i+1}: Winner={winner}, Moves={moves}, Time={duration:.2f}s, AZ={'Black' if az_is_black else 'White'}")
    else:
        for i in range(num_games):
            env.reset()
            pure_mcts = MCTS(env, simulations_per_move=mcts_sims)
            az_is_black = (i % 2 == 0)

            if az_is_black:
                black_fn = lambda e: mcts_player(e, az_mcts, temperature=0)
                white_fn = lambda e: pure_mcts.run((e.board.copy(), e.current_player, e.consecutive_passes))
            else:
                black_fn = lambda e: pure_mcts.run((e.board.copy(), e.current_player, e.consecutive_passes))
                white_fn = lambda e: mcts_player(e, az_mcts, temperature=0)

            winner, moves, duration = play_game(env, black_fn, white_fn, render=render, max_moves=max_moves)
            results[winner] += 1
            total_moves += moves
            total_time += duration
            if (az_is_black and winner == "Black") or ((not az_is_black) and winner == "White"):
                az_wins += 1
            print(f"Game {i+1}: Winner={winner}, Moves={moves}, Time={duration:.2f}s, AZ={'Black' if az_is_black else 'White'}")

    avg_moves = total_moves / num_games
    avg_time = total_time / num_games
    print(f"\nResults: {results}")
    print(f"AlphaZero Wins: {az_wins}/{num_games}")
    print(f"Avg Moves: {avg_moves:.1f}, Avg Time: {avg_time:.2f}s")
    return {
        "results": results,
        "az_wins": az_wins,
        "avg_moves": avg_moves,
        "avg_time": avg_time
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render each move.")
    parser.add_argument("--checkpoint", default="checkpoints/board_7/alphazero_latest.pt")
    parser.add_argument("--board-size", type=int, default=None)
    parser.add_argument("--num-res-blocks", type=int, default=None)
    parser.add_argument("--num-channels", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--center_wall", type=str, default="True",
                        help="Center neutral wall placement (True/False).")
    return parser.parse_args()


def main():
    args = parse_args()
    network, _ = load_network_from_checkpoint(
        args.checkpoint,
        board_size=args.board_size,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels
    )
    board_size = network.board_size
    env = GreatKingdomEnv(board_size=board_size, center_wall=args.center_wall.lower() == "true")

    az_mcts = AlphaZeroMCTS(network, env, num_simulations=150, c_puct=1.5)

    for sims in (300,500):
        run_alphazero_vs_mcts(
            env,
            az_mcts,
            sims,
            num_games=20,
            render=args.render,
            num_workers=args.num_workers
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
