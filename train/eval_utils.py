import time
import numpy as np

from env.env import GreatKingdomEnv
from env.env_fast import GreatKingdomEnvFast
from game_result import winner_id_from_step, winner_label_from_step
from network import AlphaZeroNetwork, infer_head_type_from_state_dict, load_state_dict_safe
from mcts_alphazero import AlphaZeroMCTS
from mcts import MCTS
from train.selfplay_workers import AsyncNetworkClient

_eval_profile_timing = False


def _render_recorded_game(record, board_size, center_wall):
    env = GreatKingdomEnv(board_size=board_size, center_wall=center_wall)
    print(f"\n=== Debug Render | Winner: {record['winner']} | Moves: {record['moves']} ===")
    for state in record["states"]:
        env.board = state["board"].copy()
        env.current_player = state["current_player"]
        env.consecutive_passes = state["consecutive_passes"]
        env.render()


def _init_eval_worker(network_state_dict, board_size, num_res_blocks, num_channels, num_simulations, c_puct, center_wall, komi, use_fast_env=False,
                      use_liberty_features=True, liberty_bins=2, use_last_moves=False, mcts_eval_batch_size=1, mcts_profile=False, mcts_profile_every=0,
                      profile_timing=False):
    """평가 워커 초기화 (CPU)."""
    global _eval_env, _eval_mcts, _eval_use_liberty_features, _eval_liberty_bins, _eval_use_last_moves, _eval_profile_timing
    _eval_use_liberty_features = use_liberty_features
    _eval_liberty_bins = liberty_bins
    _eval_use_last_moves = use_last_moves
    _eval_profile_timing = profile_timing
    env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
    _eval_env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)
    head_type = infer_head_type_from_state_dict(network_state_dict)
    _eval_network = AlphaZeroNetwork(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves,
        head_type=head_type
    )
    load_state_dict_safe(_eval_network, network_state_dict)
    _eval_network.eval()
    _eval_mcts = AlphaZeroMCTS(
        _eval_network, _eval_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        eval_batch_size=mcts_eval_batch_size,
        profile=mcts_profile,
        profile_every=mcts_profile_every,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )


def _eval_one_game(ai_player):
    """???????? ??? (AI vs Random)."""
    global _eval_env, _eval_mcts
    _eval_env.reset()
    done = False
    moves = 0
    max_moves = max(100, _eval_env.board_size * _eval_env.board_size + 20)
    last_moves = (None, None) if _eval_use_last_moves else None
    t_start = time.perf_counter() if _eval_profile_timing else None
    mcts_time = 0.0
    env_time = 0.0

    while not done and moves < max_moves:
        current_player = _eval_env.current_player
        if _eval_use_last_moves:
            state = (
                _eval_env.board.copy(),
                _eval_env.current_player,
                _eval_env.consecutive_passes,
                last_moves
            )
        else:
            state = (
                _eval_env.board.copy(),
                _eval_env.current_player,
                _eval_env.consecutive_passes
            )

        if current_player == ai_player:
            if _eval_profile_timing:
                t0 = time.perf_counter()
                action, _ = _eval_mcts.run(state, temperature=0)
                mcts_time += time.perf_counter() - t0
            else:
                action, _ = _eval_mcts.run(state, temperature=0)
        else:
            legal_moves = _eval_env.get_legal_moves()
            legal_indices = np.where(legal_moves)[0]
            non_pass = legal_indices[legal_indices != _eval_env.pass_action]
            if len(non_pass) > 0 and np.random.random() > 0.1:
                action = np.random.choice(non_pass)
            else:
                action = np.random.choice(legal_indices)

        if _eval_profile_timing:
            t0 = time.perf_counter()
            _, reward, done, info = _eval_env.step(action)
            env_time += time.perf_counter() - t0
        else:
            _, reward, done, info = _eval_env.step(action)
        if _eval_use_last_moves and action != _eval_env.pass_action:
            last_moves = (action, last_moves[0])
        moves += 1

    winner = winner_label_from_step(info, reward, _eval_env.current_player)

    timing = None
    if _eval_profile_timing:
        timing = {
            "total_time": time.perf_counter() - t_start,
            "mcts_time": mcts_time,
            "opponent_mcts_time": 0.0,
            "env_time": env_time,
            "moves": moves
        }

    return winner, ai_player, timing


def _init_eval_worker_async(worker_id, input_queue, output_queue, board_size, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, center_wall, komi, use_fast_env=False,
                            use_liberty_features=True, liberty_bins=2, use_last_moves=False, mcts_eval_batch_size=1, mcts_profile=False, mcts_profile_every=0, profile_timing=False):
    global _eval_env, _eval_mcts, _eval_board_size, _eval_use_liberty_features, _eval_liberty_bins, _eval_use_last_moves, _eval_profile_timing
    _eval_board_size = board_size
    _eval_use_liberty_features = use_liberty_features
    _eval_liberty_bins = liberty_bins
    _eval_use_last_moves = use_last_moves
    _eval_profile_timing = profile_timing
    env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
    _eval_env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)
    async_net = AsyncNetworkClient(input_queue, output_queue, worker_id)
    _eval_mcts = AlphaZeroMCTS(
        async_net, _eval_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=mcts_eval_batch_size,
        profile=mcts_profile,
        profile_every=mcts_profile_every,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )


def _eval_one_game_random(ai_player):
    global _eval_env, _eval_mcts
    _eval_env.reset()
    done = False
    moves = 0
    max_moves = max(100, _eval_env.board_size * _eval_env.board_size + 20)
    last_moves = (None, None) if _eval_use_last_moves else None
    info = {}
    reward = 0
    states = [{
        "board": _eval_env.board.copy(),
        "current_player": _eval_env.current_player,
        "consecutive_passes": _eval_env.consecutive_passes
    }]
    t_start = time.perf_counter() if _eval_profile_timing else None
    mcts_time = 0.0
    env_time = 0.0

    while not done and moves < max_moves:
        current_player = _eval_env.current_player
        if _eval_use_last_moves:
            state = (
                _eval_env.board.copy(),
                _eval_env.current_player,
                _eval_env.consecutive_passes,
                last_moves
            )
        else:
            state = (
                _eval_env.board.copy(),
                _eval_env.current_player,
                _eval_env.consecutive_passes
            )

        if current_player == ai_player:
            if _eval_profile_timing:
                t0 = time.perf_counter()
                action, _ = _eval_mcts.run(state, temperature=0)
                mcts_time += time.perf_counter() - t0
            else:
                action, _ = _eval_mcts.run(state, temperature=0)
        else:
            legal_moves = _eval_env.get_legal_moves()
            legal_indices = np.where(legal_moves)[0]
            non_pass = legal_indices[legal_indices != _eval_env.pass_action]
            if len(non_pass) > 0 and np.random.random() > 0.1:
                action = np.random.choice(non_pass)
            else:
                action = np.random.choice(legal_indices)

        if _eval_profile_timing:
            t0 = time.perf_counter()
            _, reward, done, info = _eval_env.step(action)
            env_time += time.perf_counter() - t0
        else:
            _, reward, done, info = _eval_env.step(action)
        if _eval_use_last_moves and action != _eval_env.pass_action:
            last_moves = (action, last_moves[0])
        moves += 1
        states.append({
            "board": _eval_env.board.copy(),
            "current_player": _eval_env.current_player,
            "consecutive_passes": _eval_env.consecutive_passes
        })

    winner = winner_id_from_step(info, reward, _eval_env.current_player)
    timing = None
    if _eval_profile_timing:
        timing = {
            "total_time": time.perf_counter() - t_start,
            "mcts_time": mcts_time,
            "opponent_mcts_time": 0.0,
            "env_time": env_time,
            "moves": moves
        }
    return winner, ai_player, moves, states, info, _eval_env.current_player, timing


def _eval_one_game_mcts(ai_player, mcts_sims):
    global _eval_env, _eval_mcts
    _eval_env.reset()
    done = False
    moves = 0
    max_moves = max(100, _eval_env.board_size * _eval_env.board_size + 20)
    last_action = None
    last_moves = (None, None) if _eval_use_last_moves else None
    info = {}
    reward = 0
    states = [{
        "board": _eval_env.board.copy(),
        "current_player": _eval_env.current_player,
        "consecutive_passes": _eval_env.consecutive_passes
    }]
    t_start = time.perf_counter() if _eval_profile_timing else None
    mcts_time = 0.0
    opp_mcts_time = 0.0
    env_time = 0.0

    pure_mcts = MCTS(_eval_env, simulations_per_move=mcts_sims)

    while not done and moves < max_moves:
        if _eval_use_last_moves:
            state = (
                _eval_env.board.copy(),
                _eval_env.current_player,
                _eval_env.consecutive_passes,
                last_moves
            )
        else:
            state = (
                _eval_env.board.copy(),
                _eval_env.current_player,
                _eval_env.consecutive_passes
            )

        if _eval_env.current_player == ai_player:
            if _eval_profile_timing:
                t0 = time.perf_counter()
                action, _ = _eval_mcts.run(state, temperature=0)
                mcts_time += time.perf_counter() - t0
            else:
                action, _ = _eval_mcts.run(state, temperature=0)
        else:
            if _eval_profile_timing:
                t0 = time.perf_counter()
                action = pure_mcts.run(state[:3], opponent_last_action=last_action)
                opp_mcts_time += time.perf_counter() - t0
            else:
                action = pure_mcts.run(state[:3], opponent_last_action=last_action)

        if _eval_profile_timing:
            t0 = time.perf_counter()
            _, reward, done, info = _eval_env.step(action)
            env_time += time.perf_counter() - t0
        else:
            _, reward, done, info = _eval_env.step(action)
        if _eval_use_last_moves and action != _eval_env.pass_action:
            last_moves = (action, last_moves[0])
        moves += 1
        last_action = action
        states.append({
            "board": _eval_env.board.copy(),
            "current_player": _eval_env.current_player,
            "consecutive_passes": _eval_env.consecutive_passes
        })

    winner = winner_id_from_step(info, reward, _eval_env.current_player)
    timing = None
    if _eval_profile_timing:
        timing = {
            "total_time": time.perf_counter() - t_start,
            "mcts_time": mcts_time,
            "opponent_mcts_time": opp_mcts_time,
            "env_time": env_time,
            "moves": moves
        }
    return winner, ai_player, moves, states, info, _eval_env.current_player, timing


def _eval_worker_loop_random(game_queue, result_queue, worker_id, input_queue, output_queue, board_size, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, center_wall, komi, use_fast_env=False,
                             use_liberty_features=True, liberty_bins=2, use_last_moves=False, mcts_eval_batch_size=1, mcts_profile=False, mcts_profile_every=0, profile_timing=False):
    _init_eval_worker_async(
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
        mcts_profile,
        mcts_profile_every,
        profile_timing
    )
    while True:
        item = game_queue.get()
        if item is None:
            break
        winner, ai_player, moves, states, info, final_player, timing = _eval_one_game_random(item)
        result_queue.put((winner, ai_player, moves, states, info, final_player, timing))


def _init_eval_worker_best_async(worker_id, input_queue, output_queue, board_size, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, opponent_state_dict, center_wall, komi, use_fast_env=False,
                                 use_liberty_features=True, liberty_bins=2, use_last_moves=False, mcts_eval_batch_size=1, mcts_profile=False, mcts_profile_every=0, profile_timing=False):
    global _eval_env, _eval_mcts, _eval_opponent_mcts, _eval_board_size, _eval_use_liberty_features, _eval_liberty_bins, _eval_use_last_moves, _eval_profile_timing
    _eval_board_size = board_size
    _eval_use_liberty_features = use_liberty_features
    _eval_liberty_bins = liberty_bins
    _eval_use_last_moves = use_last_moves
    _eval_profile_timing = profile_timing
    env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
    _eval_env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)

    async_net = AsyncNetworkClient(input_queue, output_queue, worker_id)
    _eval_mcts = AlphaZeroMCTS(
        async_net, _eval_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=mcts_eval_batch_size,
        profile=mcts_profile,
        profile_every=mcts_profile_every,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )

    opponent_head = infer_head_type_from_state_dict(opponent_state_dict)
    opponent_network = AlphaZeroNetwork(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves,
        head_type=opponent_head
    )
    load_state_dict_safe(opponent_network, opponent_state_dict)
    opponent_network.eval()

    _eval_opponent_mcts = AlphaZeroMCTS(
        opponent_network, _eval_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=mcts_eval_batch_size,
        profile=mcts_profile,
        profile_every=mcts_profile_every,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )


def _init_eval_worker_best_dual_async(worker_id, input_queue, output_queue, opponent_input_queue, opponent_output_queue, board_size, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, center_wall, komi, use_fast_env=False,
                                      use_liberty_features=True, liberty_bins=2, use_last_moves=False, mcts_eval_batch_size=1, mcts_profile=False, mcts_profile_every=0, profile_timing=False):
    global _eval_env, _eval_mcts, _eval_opponent_mcts, _eval_board_size, _eval_use_liberty_features, _eval_liberty_bins, _eval_use_last_moves, _eval_profile_timing
    _eval_board_size = board_size
    _eval_use_liberty_features = use_liberty_features
    _eval_liberty_bins = liberty_bins
    _eval_use_last_moves = use_last_moves
    _eval_profile_timing = profile_timing
    env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
    _eval_env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)

    async_net = AsyncNetworkClient(input_queue, output_queue, worker_id)
    _eval_mcts = AlphaZeroMCTS(
        async_net, _eval_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=mcts_eval_batch_size,
        profile=mcts_profile,
        profile_every=mcts_profile_every,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )

    opponent_async = AsyncNetworkClient(opponent_input_queue, opponent_output_queue, worker_id)
    _eval_opponent_mcts = AlphaZeroMCTS(
        opponent_async, _eval_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=mcts_eval_batch_size,
        profile=mcts_profile,
        profile_every=mcts_profile_every,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )


def _eval_worker_loop_best(game_queue, result_queue, worker_id, input_queue, output_queue, board_size, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, opponent_state_dict, center_wall, komi, use_fast_env=False,
                           use_liberty_features=True, liberty_bins=2, use_last_moves=False, mcts_eval_batch_size=1, mcts_profile=False, mcts_profile_every=0, profile_timing=False):
    _init_eval_worker_best_async(
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
        opponent_state_dict,
        center_wall,
        komi,
        use_fast_env,
        use_liberty_features,
        liberty_bins,
        use_last_moves,
        mcts_eval_batch_size,
        mcts_profile,
        mcts_profile_every,
        profile_timing
    )
    while True:
        item = game_queue.get()
        if item is None:
            break
        winner, ai_player, moves, states, info, final_player, timing = _eval_one_game_best(item)
        result_queue.put((winner, ai_player, moves, states, info, final_player, timing))


def _eval_worker_loop_best_dual_async(game_queue, result_queue, worker_id, input_queue, output_queue, opponent_input_queue, opponent_output_queue, board_size, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, center_wall, komi, use_fast_env=False,
                                      use_liberty_features=True, liberty_bins=2, use_last_moves=False, mcts_eval_batch_size=1, mcts_profile=False, mcts_profile_every=0, profile_timing=False):
    _init_eval_worker_best_dual_async(
        worker_id,
        input_queue,
        output_queue,
        opponent_input_queue,
        opponent_output_queue,
        board_size,
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
        mcts_profile,
        mcts_profile_every,
        profile_timing
    )
    while True:
        item = game_queue.get()
        if item is None:
            break
        winner, ai_player, moves, states, info, final_player, timing = _eval_one_game_best(item)
        result_queue.put((winner, ai_player, moves, states, info, final_player, timing))


def _eval_one_game_best(ai_player):
    global _eval_env, _eval_mcts, _eval_opponent_mcts
    _eval_env.reset()
    done = False
    moves = 0
    max_moves = max(100, _eval_env.board_size * _eval_env.board_size + 20)
    last_moves = (None, None) if _eval_use_last_moves else None
    info = {}
    reward = 0
    states = [{
        "board": _eval_env.board.copy(),
        "current_player": _eval_env.current_player,
        "consecutive_passes": _eval_env.consecutive_passes
    }]
    t_start = time.perf_counter() if _eval_profile_timing else None
    mcts_time = 0.0
    opp_mcts_time = 0.0
    env_time = 0.0

    while not done and moves < max_moves:
        if _eval_use_last_moves:
            state = (
                _eval_env.board.copy(),
                _eval_env.current_player,
                _eval_env.consecutive_passes,
                last_moves
            )
        else:
            state = (
                _eval_env.board.copy(),
                _eval_env.current_player,
                _eval_env.consecutive_passes
            )

        if _eval_env.current_player == ai_player:
            if _eval_profile_timing:
                t0 = time.perf_counter()
                action, _ = _eval_mcts.run(state, temperature=0)
                mcts_time += time.perf_counter() - t0
            else:
                action, _ = _eval_mcts.run(state, temperature=0)
        else:
            if _eval_profile_timing:
                t0 = time.perf_counter()
                action, _ = _eval_opponent_mcts.run(state, temperature=0)
                opp_mcts_time += time.perf_counter() - t0
            else:
                action, _ = _eval_opponent_mcts.run(state, temperature=0)

        if _eval_profile_timing:
            t0 = time.perf_counter()
            _, reward, done, info = _eval_env.step(action)
            env_time += time.perf_counter() - t0
        else:
            _, reward, done, info = _eval_env.step(action)
        if _eval_use_last_moves and action != _eval_env.pass_action:
            last_moves = (action, last_moves[0])
        moves += 1
        states.append({
            "board": _eval_env.board.copy(),
            "current_player": _eval_env.current_player,
            "consecutive_passes": _eval_env.consecutive_passes
        })

    winner = winner_id_from_step(info, reward, _eval_env.current_player)
    timing = None
    if _eval_profile_timing:
        timing = {
            "total_time": time.perf_counter() - t_start,
            "mcts_time": mcts_time,
            "opponent_mcts_time": opp_mcts_time,
            "env_time": env_time,
            "moves": moves
        }
    return winner, ai_player, moves, states, info, _eval_env.current_player, timing


def _eval_worker_loop_mcts(game_queue, result_queue, worker_id, input_queue, output_queue, board_size, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, center_wall, komi, mcts_sims, use_fast_env=False,
                           use_liberty_features=True, liberty_bins=2, use_last_moves=False, mcts_eval_batch_size=1, mcts_profile=False, mcts_profile_every=0, profile_timing=False):
    _init_eval_worker_async(
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
        mcts_profile,
        mcts_profile_every,
        profile_timing
    )
    while True:
        item = game_queue.get()
        if item is None:
            break
        winner, ai_player, moves, states, info, final_player, timing = _eval_one_game_mcts(item, mcts_sims)
        result_queue.put((winner, ai_player, moves, states, info, final_player, timing))
