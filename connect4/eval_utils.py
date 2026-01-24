import numpy as np

from connect4.env import Connect4Env
from connect4.network import AlphaZeroNetwork
from connect4.mcts_alphazero import AlphaZeroMCTS
from connect4.mcts import MCTS
from connect4.selfplay_workers import AsyncNetworkClient


def _render_recorded_game(record, rows, cols):
    env = Connect4Env(rows=rows, cols=cols)
    print(f"\n=== Debug Render | Winner: {record['winner']} | Moves: {record['moves']} ===")
    for state in record["states"]:
        env.board = state["board"].copy()
        env.current_player = state["current_player"]
        env.consecutive_passes = state["consecutive_passes"]
        env.render()


def _init_eval_worker_async(worker_id, input_queue, output_queue, rows, cols, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon):
    global _eval_env, _eval_mcts
    _eval_env = Connect4Env(rows=rows, cols=cols)
    async_net = AsyncNetworkClient(input_queue, output_queue, worker_id)
    _eval_mcts = AlphaZeroMCTS(
        async_net, _eval_env,
        c_puct=c_puct,
        num_simulations=80,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=16
    )


def _eval_one_game_mcts(ai_player, mcts_sims):
    global _eval_env, _eval_mcts
    _eval_env.reset()
    done = False
    moves = 0
    last_action = None
    states = [{
        "board": _eval_env.board.copy(),
        "current_player": _eval_env.current_player,
        "consecutive_passes": _eval_env.consecutive_passes
    }]

    pure_mcts = MCTS(_eval_env, simulations_per_move=mcts_sims)

    while not done and moves < (_eval_env.rows * _eval_env.cols):
        state = (
            _eval_env.board.copy(),
            _eval_env.current_player,
            _eval_env.consecutive_passes
        )

        if _eval_env.current_player == ai_player:
            action, _ = _eval_mcts.run(state, temperature=0)
        else:
            action = pure_mcts.run(state, opponent_last_action=last_action)

        _, reward, done, info = _eval_env.step(action)
        moves += 1
        last_action = action
        states.append({
            "board": _eval_env.board.copy(),
            "current_player": _eval_env.current_player,
            "consecutive_passes": _eval_env.consecutive_passes
        })

    winner = info.get("winner", 0) if done else 0
    return winner, ai_player, moves, states


def _eval_worker_loop_mcts(game_queue, result_queue, worker_id, input_queue, output_queue, rows, cols, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, mcts_sims):
    _init_eval_worker_async(
        worker_id,
        input_queue,
        output_queue,
        rows,
        cols,
        num_res_blocks,
        num_channels,
        num_simulations,
        c_puct,
        dirichlet_alpha,
        dirichlet_epsilon
    )
    while True:
        item = game_queue.get()
        if item is None:
            break
        winner, ai_player, moves, states = _eval_one_game_mcts(item, mcts_sims)
        result_queue.put((winner, ai_player, moves, states))
