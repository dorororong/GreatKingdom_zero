import time
import numpy as np
import torch
import multiprocessing as mp
from collections import OrderedDict

from env.env import GreatKingdomEnv
from env.env_fast import GreatKingdomEnvFast
from game_result import winner_id_from_step
from network import AlphaZeroNetwork, encode_board_from_state, infer_head_type_from_state_dict, load_state_dict_safe
from mcts_alphazero import AlphaZeroMCTS
from symmetry import canonicalize_encoded_state, invert_action_probs


class BatchInferenceServer:
    def __init__(self, network_state_dict, board_size, num_res_blocks, num_channels,
                 device=None, batch_size=64, timeout=0.01, stats_queue=None,
                 use_liberty_features=True, liberty_bins=2, use_last_moves=False,
                 network_head="fc", cache_debug_samples=0, cache_max_entries=50000,
                 cache_enabled=True):
        self.network_state_dict = network_state_dict
        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.timeout = timeout
        self.input_queue = mp.Queue(maxsize=10000)
        self.output_queues = []
        self.running = mp.Value('b', True)
        self.stats_queue = stats_queue
        self.use_liberty_features = use_liberty_features
        self.liberty_bins = liberty_bins
        self.use_last_moves = use_last_moves
        self.network_head = network_head
        self.cache_max_entries = int(cache_max_entries)
        self.cache_enabled = bool(cache_enabled)
        self._eval_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._deduped = 0
        self._unique_evals = 0
        self.cache_debug_samples = int(cache_debug_samples)
        self._debug_samples = []

    def run(self):
        network = AlphaZeroNetwork(
            board_size=self.board_size,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels,
            use_liberty_features=self.use_liberty_features,
            liberty_bins=self.liberty_bins,
            use_last_moves=self.use_last_moves,
            head_type=self.network_head
        ).to(self.device)
        load_state_dict_safe(network, self.network_state_dict)
        network.eval()

        print(f"?? Inference Server Started on {self.device} (Batch: {self.batch_size})")

        total_requests = 0
        total_batches = 0
        total_infer_time = 0.0
        total_wait_time = 0.0

        while self.running.value:
            batch_items = []
            start_wait = time.time()
            while len(batch_items) < self.batch_size:
                try:
                    item = self.input_queue.get(timeout=self.timeout)
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

            if not self.cache_enabled:
                states_tensor = torch.FloatTensor(
                    np.array([state for _, _, state in batch_items])
                ).to(self.device)
                with torch.no_grad():
                    t0 = time.perf_counter()
                    if self.device.startswith("cuda"):
                        with torch.amp.autocast("cuda"):
                            policies, values, _, _, _ = network(states_tensor)
                    else:
                        policies, values, _, _, _ = network(states_tensor)
                    total_infer_time += (time.perf_counter() - t0)

                policies = torch.exp(policies).float().cpu().numpy()
                values = values.squeeze(1).float().cpu().numpy()

                total_batches += 1
                self._unique_evals += len(batch_items)
                for (worker_id, req_id, _), policy, value in zip(batch_items, policies, values):
                    self.output_queues[worker_id].put((req_id, policy, float(value)))
                total_requests += len(batch_items)
                continue

            pending = {}
            for worker_id, req_id, state in batch_items:
                canon_state, transform_id, key = canonicalize_encoded_state(state)
                cached = self._eval_cache.get(key)
                if cached is not None:
                    self._cache_hits += 1
                    policy_canon, value = cached
                    policy = invert_action_probs(policy_canon, self.board_size, transform_id)
                    self.output_queues[worker_id].put((req_id, policy, value))
                    continue
                self._cache_misses += 1
                if key not in pending:
                    pending[key] = {
                        "canon_state": canon_state,
                        "requests": [],
                        "debug_state": state,
                        "debug_transform_id": transform_id
                    }
                pending[key]["requests"].append((worker_id, req_id, transform_id))

            if pending:
                keys = list(pending.keys())
                unique_count = len(keys)
                total_pending = sum(len(v["requests"]) for v in pending.values())
                self._deduped += max(0, total_pending - unique_count)
                self._unique_evals += unique_count
                states_tensor = torch.FloatTensor(
                    np.array([pending[k]["canon_state"] for k in keys])
                ).to(self.device)
                with torch.no_grad():
                    t0 = time.perf_counter()
                    if self.device.startswith("cuda"):
                        with torch.amp.autocast("cuda"):
                            policies, values, _, _, _ = network(states_tensor)
                    else:
                        policies, values, _, _, _ = network(states_tensor)
                    total_infer_time += (time.perf_counter() - t0)

                policies = torch.exp(policies).float().cpu().numpy()
                values = values.squeeze(1).float().cpu().numpy()

                total_batches += 1
                for idx, key in enumerate(keys):
                    policy_canon = policies[idx]
                    value = float(values[idx])
                    if self.cache_debug_samples > 0 and len(self._debug_samples) < self.cache_debug_samples:
                        self._debug_samples.append((
                            policy_canon,
                            value,
                            pending[key]["canon_state"],
                            pending[key]["debug_state"],
                            pending[key]["debug_transform_id"]
                        ))
                    self._eval_cache[key] = (policy_canon, value)
                    if self.cache_max_entries and len(self._eval_cache) > self.cache_max_entries:
                        self._eval_cache.popitem(last=False)
                    for worker_id, req_id, transform_id in pending[key]["requests"]:
                        policy = invert_action_probs(policy_canon, self.board_size, transform_id)
                        self.output_queues[worker_id].put((req_id, policy, value))

            total_requests += len(batch_items)

        cache_size = len(self._eval_cache)
        debug_samples = 0
        debug_policy_l1 = None
        debug_value_max = None
        debug_policy_l1_orig = None
        debug_value_max_orig = None
        if self.cache_enabled and self.cache_debug_samples > 0 and self._debug_samples:
            sample_states = []
            orig_states = []
            cached_policies = []
            cached_values = []
            debug_transform_ids = []
            for policy_canon, value, canon_state, orig_state, transform_id in self._debug_samples:
                sample_states.append(canon_state)
                orig_states.append(orig_state)
                cached_policies.append(policy_canon)
                cached_values.append(float(value))
                debug_transform_ids.append(transform_id)

            states_tensor = torch.FloatTensor(np.array(sample_states)).to(self.device)
            orig_tensor = torch.FloatTensor(np.array(orig_states)).to(self.device)
            with torch.no_grad():
                if self.device.startswith("cuda"):
                    with torch.amp.autocast("cuda"):
                        policies, values, _, _, _ = network(states_tensor)
                        policies_orig, values_orig, _, _, _ = network(orig_tensor)
                else:
                    policies, values, _, _, _ = network(states_tensor)
                    policies_orig, values_orig, _, _, _ = network(orig_tensor)
            policies = torch.exp(policies).float().cpu().numpy()
            values = values.squeeze(1).float().cpu().numpy()
            policies_orig = torch.exp(policies_orig).float().cpu().numpy()
            values_orig = values_orig.squeeze(1).float().cpu().numpy()

            diffs = []
            vdiffs = []
            diffs_orig = []
            vdiffs_orig = []
            for idx in range(len(cached_policies)):
                diffs.append(np.mean(np.abs(policies[idx] - cached_policies[idx])))
                vdiffs.append(abs(float(values[idx]) - cached_values[idx]))
                restored = invert_action_probs(cached_policies[idx], self.board_size, debug_transform_ids[idx])
                diffs_orig.append(np.mean(np.abs(policies_orig[idx] - restored)))
                vdiffs_orig.append(abs(float(values_orig[idx]) - cached_values[idx]))
            debug_samples = len(diffs)
            debug_policy_l1 = float(np.mean(diffs)) if diffs else None
            debug_value_max = float(np.max(vdiffs)) if vdiffs else None
            debug_policy_l1_orig = float(np.mean(diffs_orig)) if diffs_orig else None
            debug_value_max_orig = float(np.max(vdiffs_orig)) if vdiffs_orig else None

        if self.stats_queue is not None:
            self.stats_queue.put({
                "total_requests": total_requests,
                "total_batches": total_batches,
                "total_infer_time": total_infer_time,
                "total_wait_time": total_wait_time,
                "batch_size": self.batch_size,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_size": cache_size,
                "deduped": self._deduped,
                "unique_evals": self._unique_evals,
                "forward_calls": self._unique_evals,
                "cache_hits_bypassed": self._cache_hits,
                "debug_samples": debug_samples,
                "debug_policy_l1": debug_policy_l1,
                "debug_value_max": debug_value_max,
                "debug_policy_l1_orig": debug_policy_l1_orig,
                "debug_value_max_orig": debug_value_max_orig
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


def _classify_result(info):
    result = info.get("result", "") if isinstance(info, dict) else ""
    if result == "Win by Capture":
        return "capture"
    if result == "Loss by Suicide":
        return "suicide"
    if result == "Territory Count":
        return "territory"
    if isinstance(result, str) and result.startswith("Draw"):
        return "draw"
    return "other"


def _scheduled_temperature(move_count, schedule=None, board_size=None):
    if schedule:
        thresholds = sorted(int(k) for k in schedule.keys())
        for t in thresholds:
            value = schedule.get(t, schedule.get(str(t)))
            if value is None:
                continue
            if move_count <= t:
                return float(value)
        last = thresholds[-1]
        return float(schedule.get(last, schedule.get(str(last))))

    if board_size is not None:
        max_moves = board_size * board_size
        t1 = int(max_moves * 0.2)
        t2 = int(max_moves * 0.4)
        if move_count < t1:
            return 1.0
        if move_count < t2:
            return 0.5
        return 0.1
    return 1.0


def _get_ownership_target(env):
    black = env._get_territory_mask(1).astype(np.float32)
    white = env._get_territory_mask(2).astype(np.float32)
    return np.stack([black, white], axis=0)


def _init_worker(network_state_dict, board_size, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, center_wall, komi, playout_full_prob, playout_full_cap, playout_fast_cap, mcts_eval_batch_size, mcts_profile, mcts_profile_every, use_forced_playouts, use_policy_target_pruning, forced_playout_k, use_fast_env=False, use_liberty_features=True, liberty_bins=2, use_last_moves=False, network_head=None, ownership_loss_weight=0.2, ownership_loss_weight_capture=0.1):
    """워커 프로세스 초기화 - 각 워커마다 네트워크/환경 생성"""
    global _worker_network, _worker_env, _worker_mcts, _worker_board_size
    global _worker_playout_full_prob, _worker_playout_full_cap, _worker_playout_fast_cap
    global _worker_ownership_loss_weight, _worker_ownership_loss_weight_capture
    global _worker_use_liberty_features, _worker_liberty_bins, _worker_use_last_moves
    global _worker_ownership_loss_weight, _worker_ownership_loss_weight_capture

    _worker_board_size = board_size
    _worker_playout_full_prob = playout_full_prob
    _worker_playout_full_cap = playout_full_cap
    _worker_playout_fast_cap = playout_fast_cap
    _worker_use_liberty_features = use_liberty_features
    _worker_liberty_bins = liberty_bins
    _worker_use_last_moves = use_last_moves
    _worker_ownership_loss_weight = float(ownership_loss_weight)
    _worker_ownership_loss_weight_capture = float(ownership_loss_weight_capture)
    env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
    _worker_env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)
    head_type = network_head or infer_head_type_from_state_dict(network_state_dict)
    _worker_network = AlphaZeroNetwork(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves,
        head_type=head_type
    )
    load_state_dict_safe(_worker_network, network_state_dict)
    _worker_network.eval()

    _worker_mcts = AlphaZeroMCTS(
        _worker_network, _worker_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=mcts_eval_batch_size,
        profile=mcts_profile,
        profile_every=mcts_profile_every,
        use_forced_playouts=use_forced_playouts,
        use_policy_target_pruning=use_policy_target_pruning,
        forced_playout_k=forced_playout_k,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )


def _init_worker_async(worker_id, input_queue, output_queue, board_size, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, center_wall, komi, playout_full_prob, playout_full_cap, playout_fast_cap, mcts_eval_batch_size, mcts_profile, mcts_profile_every, use_forced_playouts, use_policy_target_pruning, forced_playout_k, use_fast_env=False, use_liberty_features=True, liberty_bins=2, use_last_moves=False, ownership_loss_weight=0.2, ownership_loss_weight_capture=0.1):
    global _worker_env, _worker_mcts, _worker_board_size
    global _worker_playout_full_prob, _worker_playout_full_cap, _worker_playout_fast_cap
    global _worker_use_liberty_features, _worker_liberty_bins, _worker_use_last_moves
    global _worker_ownership_loss_weight, _worker_ownership_loss_weight_capture
    _worker_board_size = board_size
    _worker_playout_full_prob = playout_full_prob
    _worker_playout_full_cap = playout_full_cap
    _worker_playout_fast_cap = playout_fast_cap
    _worker_use_liberty_features = use_liberty_features
    _worker_liberty_bins = liberty_bins
    _worker_use_last_moves = use_last_moves
    _worker_ownership_loss_weight = float(ownership_loss_weight)
    _worker_ownership_loss_weight_capture = float(ownership_loss_weight_capture)
    env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
    _worker_env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)
    async_net = AsyncNetworkClient(input_queue, output_queue, worker_id)
    _worker_mcts = AlphaZeroMCTS(
        async_net, _worker_env,
        c_puct=c_puct,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        eval_batch_size=mcts_eval_batch_size,
        profile=mcts_profile,
        profile_every=mcts_profile_every,
        use_forced_playouts=use_forced_playouts,
        use_policy_target_pruning=use_policy_target_pruning,
        forced_playout_k=forced_playout_k,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )


def _play_one_game(temperature_schedule):
    """워커에서 한 게임 수행 (멀티프로세싱용)"""
    global _worker_network, _worker_env, _worker_mcts, _worker_board_size
    global _worker_playout_full_prob, _worker_playout_full_cap, _worker_playout_fast_cap

    _worker_env.reset()
    game_data = []
    move_count = 0
    actions = []
    max_moves = max(100, _worker_board_size * _worker_board_size + 20)
    last_moves = (None, None) if _worker_use_last_moves else None
    info = {}
    reward = 0

    while True:
        if _worker_use_last_moves:
            state = (
                _worker_env.board.copy(),
                _worker_env.current_player,
                _worker_env.consecutive_passes,
                last_moves
            )
        else:
            state = (
                _worker_env.board.copy(),
                _worker_env.current_player,
                _worker_env.consecutive_passes
            )
        encoded_state = encode_board_from_state(
            state[0], state[1], _worker_board_size, last_moves,
            use_liberty_features=_worker_use_liberty_features,
            liberty_bins=_worker_liberty_bins,
            use_last_moves=_worker_use_last_moves
        )

        temperature = _scheduled_temperature(move_count, temperature_schedule, _worker_board_size)

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

        actions.append(int(action))
        _, reward, done, info = _worker_env.step(action)
        if _worker_use_last_moves and action != _worker_env.pass_action:
            last_moves = (action, last_moves[0])
        move_count += 1

        if done:
            winner = winner_id_from_step(info, reward, _worker_env.current_player)
            break

        if move_count > max_moves:
            winner = 0
            break

    win_type = _classify_result(info)
    territory_diff = None
    if win_type == "territory":
        black = info.get("black_territory", 0) if isinstance(info, dict) else 0
        white = info.get("white_territory", 0) if isinstance(info, dict) else 0
        territory_diff = abs(int(black) - int(white))
    if winner == 1:
        win_target = 1.0
        win_weight = 1.0
    elif winner == 2:
        win_target = 0.0
        win_weight = 1.0
    else:
        win_target = 0.5
        win_weight = 0.0

    if win_type == "capture":
        win_type_target = 1.0
        win_type_weight = 1.0
    elif win_type == "territory":
        win_type_target = 0.0
        win_type_weight = 1.0
    else:
        win_type_target = 0.5
        win_type_weight = 0.0
    ownership_target = _get_ownership_target(_worker_env)
    ownership_weight = (
        _worker_ownership_loss_weight_capture
        if win_type == "capture"
        else _worker_ownership_loss_weight
    )

    final_data = []
    for enc_state, policy in game_data:
        current_player = 1 if enc_state[3, 0, 0] > 0.5 else 2

        if winner == 0:
            value = 0.0
        elif winner == current_player:
            value = 1.0
        else:
            value = -1.0

        final_data.append(
            (
                enc_state,
                policy,
                value,
                ownership_target,
                ownership_weight,
                win_target,
                win_weight,
                win_type_target,
                win_type_weight
            )
        )

    return final_data, winner, move_count, win_type, territory_diff, actions


def _worker_loop(game_queue, result_queue, worker_id, input_queue, output_queue, board_size, num_res_blocks, num_channels, num_simulations, c_puct, dirichlet_alpha, dirichlet_epsilon, center_wall, komi, playout_full_prob, playout_full_cap, playout_fast_cap, mcts_eval_batch_size, mcts_profile, mcts_profile_every, use_forced_playouts, use_policy_target_pruning, forced_playout_k, use_fast_env=False, use_liberty_features=True, liberty_bins=2, use_last_moves=False, ownership_loss_weight=0.2, ownership_loss_weight_capture=0.1):
    """Async inference worker loop."""
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
        playout_full_prob,
        playout_full_cap,
        playout_fast_cap,
        mcts_eval_batch_size,
        mcts_profile,
        mcts_profile_every,
        use_forced_playouts,
        use_policy_target_pruning,
        forced_playout_k,
        use_fast_env,
        use_liberty_features,
        liberty_bins,
        use_last_moves,
        ownership_loss_weight,
        ownership_loss_weight_capture
    )
    while True:
        item = game_queue.get()
        if item is None:
            break
        game_data, winner, moves, win_type, territory_diff, actions = _play_one_game(item)
        result_queue.put((game_data, winner, moves, win_type, territory_diff, actions))


def _benchmark_async_selfplay(trainer, num_games=10, num_workers=None):
    if num_workers is None:
        num_workers = 11

    network_state_dict = {k: v.cpu() for k, v in trainer.network.state_dict().items()}
    stats_queue = mp.Queue()
    server = BatchInferenceServer(
        network_state_dict=network_state_dict,
        board_size=trainer.board_size,
        num_res_blocks=trainer.num_res_blocks,
        num_channels=trainer.num_channels,
        batch_size=64,
        timeout=0.002,
        stats_queue=stats_queue,
        use_liberty_features=trainer.use_liberty_features,
        liberty_bins=trainer.liberty_bins,
        use_last_moves=trainer.use_last_moves,
        network_head=trainer.network_head,
        cache_debug_samples=trainer.cache_debug_samples,
        cache_max_entries=trainer.cache_max_entries,
        cache_enabled=getattr(trainer, "cache_enabled", True)
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
                trainer.board_size,
                trainer.num_res_blocks,
                trainer.num_channels,
                trainer.num_simulations,
                trainer.c_puct,
                trainer.dirichlet_alpha,
                trainer.dirichlet_epsilon,
                trainer.center_wall,
                trainer.komi,
                trainer.playout_full_prob,
                trainer.playout_full_cap,
                trainer.playout_fast_cap,
                trainer.mcts_eval_batch_size,
                trainer.mcts_profile,
                trainer.mcts_profile_every,
                trainer.use_forced_playouts,
                trainer.use_policy_target_pruning,
                trainer.forced_playout_k,
                trainer.use_fast_env,
                trainer.use_liberty_features,
                trainer.liberty_bins,
                trainer.use_last_moves,
                trainer.ownership_loss_weight,
                trainer.ownership_loss_weight_capture
            )
        )
        p.start()
        workers.append(p)

    start_time = time.time()
    for _ in range(num_games):
        game_queue.put(trainer.temperature_schedule)

    total_moves = 0
    for _ in range(num_games):
        _, _, moves, *_ = result_queue.get()
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
        batch_size = stats.get("batch_size", 0)
        avg_batch = reqs / max(1, batches)
        avg_wait_ms = 1000 * wait_time / max(1, batches)
        wait_ratio = wait_time / max(1e-9, wait_time + infer_time)
        cache_hits = stats.get("cache_hits", 0)
        cache_misses = stats.get("cache_misses", 0)
        cache_total = cache_hits + cache_misses
        cache_hit_rate = cache_hits / max(1, cache_total)
        cache_size = stats.get("cache_size", 0)
        deduped = stats.get("deduped", 0)
        unique_evals = stats.get("unique_evals", 0)
        forward_calls = stats.get("forward_calls", unique_evals)
        avg_eval_ms = 1000 * infer_time / max(1, unique_evals)
        print(
            f"    Benchmark: infer_reqs={reqs}, infer_time={infer_time:.2f}s, "
            f"wait_time={wait_time:.2f}s, avg_batch={avg_batch:.1f}, "
            f"avg_wait_ms={avg_wait_ms:.2f}, wait_ratio={wait_ratio:.2f}"
        )
        print(
            f"    Cache stats: hit_rate={cache_hit_rate:.2%}, hits={cache_hits}, misses={cache_misses}, "
            f"cache_size={cache_size}, deduped={deduped}, avg_eval_ms={avg_eval_ms:.2f}"
        )
        print(
            f"    Cache bypass: forward_calls={forward_calls}, "
            f"cache_hits_bypassed={cache_hits}, unique_evals={unique_evals}"
        )
        debug_samples = stats.get("debug_samples", 0)
        if debug_samples:
            debug_policy_l1 = stats.get("debug_policy_l1")
            debug_value_max = stats.get("debug_value_max")
            debug_policy_l1_orig = stats.get("debug_policy_l1_orig")
            debug_value_max_orig = stats.get("debug_value_max_orig")
            print(
                f"    Cache debug: samples={debug_samples}, "
                f"policy_l1={debug_policy_l1:.6f}, value_max_diff={debug_value_max:.6f}, "
                f"policy_l1_orig={debug_policy_l1_orig:.6f}, value_max_diff_orig={debug_value_max_orig:.6f}"
            )
    else:
        reqs = 0
        infer_time = 0.0
        wait_time = 0.0
        batches = 0
        batch_size = 0

    return {
        "games": num_games,
        "workers": num_workers,
        "elapsed_sec": elapsed,
        "avg_moves": avg_moves,
        "infer_reqs": reqs,
        "infer_time_sec": infer_time,
        "wait_time_sec": wait_time,
        "total_batches": batches,
        "avg_batch_size": (reqs / max(1, batches)),
        "avg_wait_ms": (1000 * wait_time / max(1, batches)),
        "wait_ratio": (wait_time / max(1e-9, wait_time + infer_time)),
        "batch_size": batch_size,
        "cache_hit_rate": (cache_hit_rate if stats else None),
        "cache_size": (cache_size if stats else None),
        "deduped": (deduped if stats else None),
        "avg_eval_ms": (avg_eval_ms if stats else None),
        "forward_calls": (forward_calls if stats else None),
        "cache_hits_bypassed": (cache_hits if stats else None),
        "debug_samples": (stats.get("debug_samples", 0) if stats else None),
        "debug_policy_l1": (stats.get("debug_policy_l1") if stats else None),
        "debug_value_max": (stats.get("debug_value_max") if stats else None),
        "debug_policy_l1_orig": (stats.get("debug_policy_l1_orig") if stats else None),
        "debug_value_max_orig": (stats.get("debug_value_max_orig") if stats else None)
    }
