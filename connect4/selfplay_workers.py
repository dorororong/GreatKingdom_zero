import time
import numpy as np
import torch
import multiprocessing as mp

from connect4.env import Connect4Env
from connect4.network import AlphaZeroNetwork, encode_board_from_state
from connect4.mcts_alphazero import AlphaZeroMCTS


class BatchInferenceServer:
    def __init__(self, network_state_dict, rows, cols, num_res_blocks, num_channels, device=None, batch_size=64, timeout=0.01, stats_queue=None):
        self.network_state_dict = network_state_dict
        self.rows = rows
        self.cols = cols
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.timeout = timeout
        self.input_queue = mp.Queue(maxsize=10000)
        self.output_queues = []
        self.running = mp.Value('b', True)
        self.stats_queue = stats_queue

    def run(self):
        network = AlphaZeroNetwork(
            rows=self.rows,
            cols=self.cols,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels
        ).to(self.device)
        network.load_state_dict(self.network_state_dict)
        network.eval()

        total_requests = 0
        total_batches = 0
        total_infer_time = 0.0
        total_wait_time = 0.0

        while self.running.value:
            batch_items = []
            start_wait = time.time()
            while len(batch_items) < self.batch_size:
                try:
                    item = self.input_queue.get(timeout=0.001)
                    if item is None:
                        self.running.value = False
                        break
                    batch_items.append(item)
                except Exception:
                    if len(batch_items) > 0:
                        break
                    if time.time() - start_wait > self.timeout:
                        break

            if not batch_items:
                continue
            total_wait_time += (time.time() - start_wait)

            worker_ids, req_ids, states = zip(*batch_items)
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            with torch.no_grad():
                t0 = time.perf_counter()
                policies, values = network(states_tensor)
                total_infer_time += (time.perf_counter() - t0)

            policies = torch.exp(policies).cpu().numpy()
            values = values.squeeze(1).cpu().numpy()

            total_requests += len(batch_items)
            total_batches += 1

            for idx, worker_id in enumerate(worker_ids):
                self.output_queues[worker_id].put((req_ids[idx], policies[idx], values[idx]))

        if self.stats_queue is not None:
            self.stats_queue.put({
                "total_requests": total_requests,
                "total_batches": total_batches,
                "total_infer_time": total_infer_time,
                "total_wait_time": total_wait_time,
                "batch_size": self.batch_size
            })


class AsyncNetworkClient:
    def __init__(self, input_queue, output_queue, worker_id):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.worker_id = worker_id
        self._req_id = 0

    def predict(self, encoded_state):
        policies, values = self.predict_batch(np.expand_dims(encoded_state, axis=0))
        return policies[0], float(values[0])

    def predict_batch(self, encoded_states):
        req_ids = []
        for state in encoded_states:
            req_id = self._req_id
            self._req_id += 1
            req_ids.append(req_id)
            self.input_queue.put((self.worker_id, req_id, state))

        results = {}
        while len(results) < len(req_ids):
            req_id, policy, value = self.output_queue.get()
            results[req_id] = (policy, value)

        policies = [results[rid][0] for rid in req_ids]
        values = [results[rid][1] for rid in req_ids]
        return np.array(policies), np.array(values)


def _scheduled_temperature(move_count, rows, cols):
    max_moves = rows * cols
    t1 = int(max_moves * 0.2)
    t2 = int(max_moves * 0.4)
    if move_count < t1:
        return 1.0
    if move_count < t2:
        return 0.5
    return 0.1


def _init_worker(network_state_dict, rows, cols, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, playout_full_prob, playout_full_cap, playout_fast_cap, use_forced_playouts, use_policy_target_pruning, forced_playout_k):
    global _worker_env, _worker_mcts, _worker_rows, _worker_cols
    global _worker_playout_full_prob, _worker_playout_full_cap, _worker_playout_fast_cap

    _worker_rows = rows
    _worker_cols = cols
    _worker_playout_full_prob = playout_full_prob
    _worker_playout_full_cap = playout_full_cap
    _worker_playout_fast_cap = playout_fast_cap
    _worker_env = Connect4Env(rows=rows, cols=cols)
    network = AlphaZeroNetwork(
        rows=rows,
        cols=cols,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels
    )
    network.load_state_dict(network_state_dict)
    network.eval()

    _worker_mcts = AlphaZeroMCTS(
        network,
        _worker_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=16,
        use_forced_playouts=use_forced_playouts,
        use_policy_target_pruning=use_policy_target_pruning,
        forced_playout_k=forced_playout_k
    )


def _init_worker_async(worker_id, input_queue, output_queue, rows, cols, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, playout_full_prob, playout_full_cap, playout_fast_cap, use_forced_playouts, use_policy_target_pruning, forced_playout_k):
    global _worker_env, _worker_mcts, _worker_rows, _worker_cols
    global _worker_playout_full_prob, _worker_playout_full_cap, _worker_playout_fast_cap
    _worker_rows = rows
    _worker_cols = cols
    _worker_playout_full_prob = playout_full_prob
    _worker_playout_full_cap = playout_full_cap
    _worker_playout_fast_cap = playout_fast_cap
    _worker_env = Connect4Env(rows=rows, cols=cols)
    async_net = AsyncNetworkClient(input_queue, output_queue, worker_id)
    _worker_mcts = AlphaZeroMCTS(
        async_net,
        _worker_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=16,
        use_forced_playouts=use_forced_playouts,
        use_policy_target_pruning=use_policy_target_pruning,
        forced_playout_k=forced_playout_k
    )


def _play_one_game(_temperature_threshold):
    global _worker_env, _worker_mcts, _worker_rows, _worker_cols
    global _worker_playout_full_prob, _worker_playout_full_cap, _worker_playout_fast_cap

    _worker_env.reset()
    game_data = []
    move_count = 0

    while True:
        state = (
            _worker_env.board.copy(),
            _worker_env.current_player,
            _worker_env.consecutive_passes
        )
        encoded_state = encode_board_from_state(
            state[0], state[1], _worker_rows, _worker_cols
        )

        temperature = _scheduled_temperature(move_count, _worker_rows, _worker_cols)

        full_search = np.random.random() < _worker_playout_full_prob
        if full_search:
            _worker_mcts.num_simulations = _worker_playout_full_cap
            action, action_probs = _worker_mcts.run(
                state, temperature=temperature, add_root_noise=True
            )
            game_data.append((encoded_state, action_probs))
        else:
            _worker_mcts.num_simulations = _worker_playout_fast_cap
            action, _ = _worker_mcts.run(
                state, temperature=temperature, add_root_noise=False
            )

        _, reward, done, info = _worker_env.step(action)
        move_count += 1

        if done:
            winner = info.get("winner", 0)
            break

        if move_count > (_worker_rows * _worker_cols):
            winner = 0
            break

    final_data = []
    for enc_state, policy in game_data:
        current_player = 1 if enc_state[2, 0, 0] > 0.5 else 2
        if winner == 0:
            value = 0.0
        elif winner == current_player:
            value = 1.0
        else:
            value = -1.0
        final_data.append((enc_state, policy, value))

    return final_data, winner, move_count


def _worker_loop(game_queue, result_queue, worker_id, input_queue, output_queue, rows, cols, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, playout_full_prob, playout_full_cap, playout_fast_cap, use_forced_playouts, use_policy_target_pruning, forced_playout_k):
    _init_worker_async(
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
        dirichlet_epsilon,
        playout_full_prob,
        playout_full_cap,
        playout_fast_cap,
        use_forced_playouts,
        use_policy_target_pruning,
        forced_playout_k
    )
    while True:
        item = game_queue.get()
        if item is None:
            break
        game_data, winner, moves = _play_one_game(item)
        result_queue.put((game_data, winner, moves))


def _decode_board_from_encoded(encoded_state):
    current_player = 1 if encoded_state[2, 0, 0] > 0.5 else 2
    opponent = 2 if current_player == 1 else 1
    board = encoded_state[0] * current_player + encoded_state[1] * opponent
    return board.astype(np.int8), current_player


def _init_reanalyze_worker_async(worker_id, input_queue, output_queue, rows, cols, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, use_forced_playouts, use_policy_target_pruning, forced_playout_k):
    global _worker_env, _worker_mcts
    _worker_env = Connect4Env(rows=rows, cols=cols)
    async_net = AsyncNetworkClient(input_queue, output_queue, worker_id)
    _worker_mcts = AlphaZeroMCTS(
        async_net,
        _worker_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=16,
        use_forced_playouts=use_forced_playouts,
        use_policy_target_pruning=use_policy_target_pruning,
        forced_playout_k=forced_playout_k
    )


def _reanalyze_worker_loop(task_queue, result_queue, worker_id, input_queue, output_queue, rows, cols, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, use_forced_playouts, use_policy_target_pruning, forced_playout_k):
    _init_reanalyze_worker_async(
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
        dirichlet_epsilon,
        use_forced_playouts,
        use_policy_target_pruning,
        forced_playout_k
    )

    while True:
        item = task_queue.get()
        if item is None:
            break
        encoded_state, value_target = item
        board, current_player = _decode_board_from_encoded(encoded_state)
        state = (board, current_player, 0)
        action, action_probs = _worker_mcts.run(state, temperature=1.0, add_root_noise=False)
        result_queue.put((encoded_state, action_probs, value_target))


def _benchmark_async_selfplay(trainer, num_games=10, num_workers=None):
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 2)

    network_state_dict = {k: v.cpu() for k, v in trainer.network.state_dict().items()}
    stats_queue = mp.Queue()
    server = BatchInferenceServer(
        network_state_dict=network_state_dict,
        rows=trainer.rows,
        cols=trainer.cols,
        num_res_blocks=trainer.num_res_blocks,
        num_channels=trainer.num_channels,
        batch_size=trainer.infer_batch_size,
        timeout=trainer.infer_timeout,
        stats_queue=stats_queue
    )
    output_queues = [mp.Queue(maxsize=1000) for _ in range(num_workers)]
    server.output_queues = output_queues

    server_process = mp.Process(target=server.run)
    server_process.start()

    game_queue = mp.Queue()
    result_queue = mp.Queue()

    workers = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=_worker_loop,
            args=(
                game_queue,
                result_queue,
                worker_id,
                server.input_queue,
                output_queues[worker_id],
                trainer.rows,
                trainer.cols,
                trainer.num_res_blocks,
                trainer.num_channels,
                trainer.num_simulations,
                trainer.c_puct,
                trainer.dirichlet_alpha,
                trainer.dirichlet_epsilon,
                trainer.playout_full_prob,
                trainer.playout_full_cap,
                trainer.playout_fast_cap,
                trainer.use_forced_playouts,
                trainer.use_policy_target_pruning,
                trainer.forced_playout_k
            )
        )
        p.start()
        workers.append(p)

    start_time = time.time()
    for _ in range(num_games):
        game_queue.put(0)

    total_moves = 0
    for _ in range(num_games):
        _, _, moves = result_queue.get()
        total_moves += moves

    for _ in workers:
        game_queue.put(None)
    for p in workers:
        p.join()

    server.input_queue.put(None)
    server_process.join()

    stats = None
    if not stats_queue.empty():
        stats = stats_queue.get()

    elapsed = time.time() - start_time
    avg_moves = total_moves / max(1, num_games)
    print(f"    Benchmark: games={num_games}, workers={num_workers}, time={elapsed:.1f}s, avg_moves={avg_moves:.1f}")
    if stats:
        reqs = stats.get("total_requests", 0)
        infer_time = stats.get("total_infer_time", 0.0)
        wait_time = stats.get("total_wait_time", 0.0)
        batches = stats.get("total_batches", 0)
        avg_batch = reqs / max(1, batches)
        avg_wait_ms = 1000 * wait_time / max(1, batches)
        wait_ratio = wait_time / max(1e-9, wait_time + infer_time)
        print(
            f"    Benchmark: infer_reqs={reqs}, infer_time={infer_time:.2f}s, "
            f"wait_time={wait_time:.2f}s, avg_batch={avg_batch:.1f}, "
            f"avg_wait_ms={avg_wait_ms:.2f}, wait_ratio={wait_ratio:.2f}"
        )

    return stats
