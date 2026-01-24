import numpy as np
import torch
import sys
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import time
import os
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from connect4.env import Connect4Env
from connect4.network import AlphaZeroNetwork, encode_board_from_state
from connect4.mcts_alphazero import AlphaZeroMCTS
from connect4.eval_utils import _eval_worker_loop_mcts, _render_recorded_game
from connect4.selfplay_workers import (
    BatchInferenceServer,
    _benchmark_async_selfplay,
    _init_worker,
    _worker_loop,
    _reanalyze_worker_loop
)
from connect4.symmetry import flip_state, flip_policy


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (
            np.array(states),
            np.array(policies),
            np.array(values, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

    def sample_candidates(self, n):
        if n <= 0 or len(self.buffer) == 0:
            return []
        items = list(self.buffer)
        if n >= len(items):
            return items
        indices = random.sample(range(len(items)), n)
        return [items[i] for i in indices]


class AlphaZeroTrainer:
    def __init__(
        self,
        rows=6,
        cols=7,
        num_res_blocks=3,
        num_channels=64,
        num_simulations=80,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        lr=0.001,
        batch_size=64,
        buffer_size=10000,
        device=None,
        playout_full_prob=0.25,
        playout_full_cap=400,
        playout_fast_cap=80,
        infer_batch_size=64,
        infer_timeout=0.005,
        use_forced_playouts=False,
        use_policy_target_pruning=False,
        forced_playout_k=2.0
    ):
        self.rows = rows
        self.cols = cols
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.playout_full_prob = playout_full_prob
        self.playout_full_cap = playout_full_cap
        self.playout_fast_cap = playout_fast_cap
        self.infer_batch_size = infer_batch_size
        self.infer_timeout = infer_timeout
        self.use_forced_playouts = use_forced_playouts
        self.use_policy_target_pruning = use_policy_target_pruning
        self.forced_playout_k = forced_playout_k

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        self.env = Connect4Env(rows=rows, cols=cols)
        self.network = AlphaZeroNetwork(
            rows=rows,
            cols=cols,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels
        ).to(self.device)
        self.mcts = AlphaZeroMCTS(
            self.network, self.env,
            c_puct=c_puct,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            eval_batch_size=16,
            use_forced_playouts=use_forced_playouts,
            use_policy_target_pruning=use_policy_target_pruning,
            forced_playout_k=forced_playout_k
        )

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)
        self.train_step = 0
        self.total_games = 0
    
    def save(self, path, iteration=None, history=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
            "total_games": self.total_games,
            "iteration": iteration
        }
        if history is not None:
            save_dict["history"] = history
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.train_step = checkpoint.get("train_step", 0)
            self.total_games = checkpoint.get("total_games", 0)
            print(f"Model loaded from {path}")
            return checkpoint
        return None

    def debug_flip_once(self):
        self.env.reset()
        game_data = []
        move_count = 0

        print("\n[Debug] Flip augmentation check (first game only)")
        self.env.render()

        while True:
            state = (
                self.env.board.copy(),
                self.env.current_player,
                self.env.consecutive_passes
            )
            encoded_state = encode_board_from_state(state[0], state[1], self.rows, self.cols)
            temperature = 1.0 if move_count < int(self.rows * self.cols * 0.2) else 0.5
            if move_count >= int(self.rows * self.cols * 0.4):
                temperature = 0.1
            self.mcts.num_simulations = self.playout_fast_cap
            action, action_probs = self.mcts.run(state, temperature=temperature, add_root_noise=False)
            game_data.append((state[0], state[1], encoded_state, action_probs))

            _, reward, done, info = self.env.step(action)
            move_count += 1
            self.env.render()
            if done or move_count > (self.rows * self.cols):
                break

        if not game_data:
            print("[Debug] No data generated.")
            return

        sample_idx = None
        for i, (raw_board, _, enc_state, _) in enumerate(game_data):
            if np.any(raw_board != 0):
                sample_idx = i
                break
        if sample_idx is None:
            sample_idx = 0

        raw_board, current_player, enc_state, policy = game_data[sample_idx]
        print(f"[Debug] Sample index: {sample_idx} | Current player: {current_player}")
        print("[Debug] Original policy:", np.round(policy, 3).tolist())
        print("[Debug] Flipped policy:", np.round(flip_policy(policy), 3).tolist())
        print("[Debug] Original board (raw):")
        print(raw_board)
        print("[Debug] Flipped board (raw):")
        print(np.fliplr(raw_board))
        print("[Debug] Encoded board (player 1 channel):")
        print(enc_state[0])
        print("[Debug] Flipped encoded (player 1 channel):")
        print(flip_state(enc_state)[0])

    def self_play_game(self):
        self.env.reset()
        game_data = []
        move_count = 0

        while True:
            state = (
                self.env.board.copy(),
                self.env.current_player,
                self.env.consecutive_passes
            )
            encoded_state = encode_board_from_state(state[0], state[1], self.rows, self.cols)

            temperature = 1.0 if move_count < int(self.rows * self.cols * 0.2) else 0.5
            if move_count >= int(self.rows * self.cols * 0.4):
                temperature = 0.1

            full_search = np.random.random() < self.playout_full_prob
            if full_search:
                self.mcts.num_simulations = self.playout_full_cap
                action, action_probs = self.mcts.run(state, temperature=temperature, add_root_noise=True)
                game_data.append((encoded_state, action_probs))
            else:
                self.mcts.num_simulations = self.playout_fast_cap
                action, _ = self.mcts.run(state, temperature=temperature, add_root_noise=False)

            _, reward, done, info = self.env.step(action)
            move_count += 1

            if done:
                winner = info.get("winner", 0)
                break
            if move_count > (self.rows * self.cols):
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

    def collect_self_play_data(self, num_games=10, verbose=True, use_multiprocessing=True, num_workers=None):
        if use_multiprocessing:
            return self._collect_self_play_multiprocess_batched(num_games, verbose, num_workers=num_workers)

        total_moves = 0
        wins = {0: 0, 1: 0, 2: 0}
        total_samples_raw = 0
        for game_idx in range(num_games):
            game_data, winner, moves = self.self_play_game()
            total_samples_raw += len(game_data)
            for state, policy, value in game_data:
                self.replay_buffer.push(state, policy, value)
                self.replay_buffer.push(flip_state(state), flip_policy(policy), value)
            total_moves += moves
            wins[winner] += 1
            self.total_games += 1
            if verbose and (game_idx + 1) % 5 == 0:
                print(f"  Game {game_idx + 1}/{num_games} | Moves: {moves} | Winner: {['Draw', 'P1', 'P2'][winner]}")

        return {
            "games": num_games,
            "avg_moves": total_moves / num_games,
            "p1_wins": wins[1],
            "p2_wins": wins[2],
            "draws": wins[0],
            "buffer_size": len(self.replay_buffer),
            "samples_raw": total_samples_raw,
            "samples_added": total_samples_raw * 2
        }

    def _collect_self_play_multiprocess_batched(self, num_games, verbose=True, num_workers=None):
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 2)
        if verbose:
            print(f"  Using {num_workers} workers with async inference...")

        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        stats_queue = mp.Queue()
        server = BatchInferenceServer(
            network_state_dict=network_state_dict,
            rows=self.rows,
            cols=self.cols,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels,
            batch_size=self.infer_batch_size,
            timeout=self.infer_timeout,
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
                    self.rows,
                    self.cols,
                    self.num_res_blocks,
                    self.num_channels,
                    self.num_simulations,
                    self.c_puct,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                    self.playout_full_prob,
                    self.playout_full_cap,
                    self.playout_fast_cap,
                    self.use_forced_playouts,
                    self.use_policy_target_pruning,
                    self.forced_playout_k
                )
            )
            p.start()
            workers.append(p)

        start_time = time.time()
        for _ in range(num_games):
            game_queue.put(0)

        total_moves = 0
        wins = {0: 0, 1: 0, 2: 0}
        games_completed = 0
        total_samples_raw = 0

        for _ in range(num_games):
            game_data, winner, moves = result_queue.get()
            total_samples_raw += len(game_data)
            for state, policy, value in game_data:
                self.replay_buffer.push(state, policy, value)
                self.replay_buffer.push(flip_state(state), flip_policy(policy), value)
            total_moves += moves
            wins[winner] += 1
            games_completed += 1

        for _ in workers:
            game_queue.put(None)
        for p in workers:
            p.join()

        server.input_queue.put(None)
        server_process.join()

        elapsed = time.time() - start_time
        try:
            stats = stats_queue.get_nowait()
        except Exception:
            stats = None

        if verbose:
            avg_game_time = elapsed / max(1, games_completed)
            moves_per_sec = total_moves / max(1e-9, elapsed)
            print(f"  Async self-play time: {elapsed:.2f}s | Avg game: {avg_game_time:.2f}s | Moves/sec: {moves_per_sec:.1f}")
            if stats:
                avg_batch = stats["total_requests"] / max(1, stats["total_batches"])
                avg_infer_ms = 1000 * stats["total_infer_time"] / max(1, stats["total_batches"])
                avg_wait_ms = 1000 * stats.get("total_wait_time", 0.0) / max(1, stats["total_batches"])
                wait_ratio = stats.get("total_wait_time", 0.0) / max(1e-9, stats.get("total_wait_time", 0.0) + stats.get("total_infer_time", 0.0))
                print(
                    f"  Inference stats: batches={stats['total_batches']}, avg_batch={avg_batch:.1f}, "
                    f"avg_batch_infer_ms={avg_infer_ms:.2f}, avg_wait_ms={avg_wait_ms:.2f}, wait_ratio={wait_ratio:.2f}"
                )

        return {
            "games": num_games,
            "avg_moves": total_moves / max(1, games_completed),
            "p1_wins": wins[1],
            "p2_wins": wins[2],
            "draws": wins[0],
            "buffer_size": len(self.replay_buffer),
            "samples_raw": total_samples_raw,
            "samples_added": total_samples_raw * 2
        }

    def _value_error_scores(self, states, values, batch_size=512):
        if len(states) == 0:
            return np.array([])

        self.network.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(states), batch_size):
                batch_states = torch.FloatTensor(states[i:i + batch_size]).to(self.device)
                _, pred_values = self.network(batch_states)
                pred_values = pred_values.squeeze(1).cpu().numpy()
                target = values[i:i + batch_size]
                errors.extend(np.abs(pred_values - target))
        return np.array(errors)

    def reanalyze_buffer(self, num_samples, num_workers=2, candidate_pool=2000, reanalyze_sims=None, debug_path=None, debug_top_k=5):
        if num_samples <= 0 or num_workers <= 0 or len(self.replay_buffer) == 0:
            return {"reanalyzed": 0, "avg_error": 0.0, "max_error": 0.0, "min_error": 0.0}

        pool_size = min(len(self.replay_buffer), max(num_samples, candidate_pool))
        candidates = self.replay_buffer.sample_candidates(pool_size)
        if not candidates:
            return {"reanalyzed": 0, "avg_error": 0.0, "max_error": 0.0, "min_error": 0.0}

        states = np.array([s for s, _, _ in candidates], dtype=np.float32)
        values = np.array([v for _, _, v in candidates], dtype=np.float32)
        errors = self._value_error_scores(states, values)
        if len(errors) == 0:
            return {"reanalyzed": 0, "avg_error": 0.0, "max_error": 0.0, "min_error": 0.0}

        top_n = min(num_samples, len(candidates))
        top_indices = np.argsort(errors)[-top_n:]
        selected = [candidates[i] for i in top_indices]
        avg_error = float(np.mean(errors[top_indices])) if top_indices.size > 0 else 0.0
        max_error = float(np.max(errors[top_indices])) if top_indices.size > 0 else 0.0
        min_error = float(np.min(errors[top_indices])) if top_indices.size > 0 else 0.0
        top_errors = sorted([float(errors[i]) for i in top_indices], reverse=True)[:max(1, debug_top_k)]

        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        server = BatchInferenceServer(
            network_state_dict=network_state_dict,
            rows=self.rows,
            cols=self.cols,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels,
            batch_size=self.infer_batch_size,
            timeout=self.infer_timeout,
            stats_queue=None
        )
        output_queues = [mp.Queue(maxsize=1000) for _ in range(num_workers)]
        server.output_queues = output_queues
        server_process = mp.Process(target=server.run)
        server_process.start()

        task_queue = mp.Queue()
        result_queue = mp.Queue()

        if reanalyze_sims is None:
            reanalyze_sims = max(self.playout_full_cap, self.playout_fast_cap, self.num_simulations)

        workers = []
        for worker_id in range(num_workers):
            p = mp.Process(
                target=_reanalyze_worker_loop,
                args=(
                    task_queue,
                    result_queue,
                    worker_id,
                    server.input_queue,
                    output_queues[worker_id],
                    self.rows,
                    self.cols,
                    self.num_res_blocks,
                    self.num_channels,
                    reanalyze_sims,
                    self.c_puct,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                    self.use_forced_playouts,
                    self.use_policy_target_pruning,
                    self.forced_playout_k
                )
            )
            p.start()
            workers.append(p)

        for state, _, value in selected:
            task_queue.put((state, value))

        reanalyzed = 0
        for _ in range(len(selected)):
            state, policy, value = result_queue.get()
            self.replay_buffer.push(state, policy, value)
            self.replay_buffer.push(flip_state(state), flip_policy(policy), value)
            reanalyzed += 1

        for _ in workers:
            task_queue.put(None)
        for p in workers:
            p.join()

        server.input_queue.put(None)
        server_process.join()

        if debug_path:
            debug_payload = {
                "buffer_size": len(self.replay_buffer),
                "candidate_pool": int(pool_size),
                "selected": int(len(selected)),
                "reanalyzed": int(reanalyzed),
                "avg_error": avg_error,
                "max_error": max_error,
                "min_error": min_error,
                "top_errors": top_errors
            }
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                import json
                json.dump(debug_payload, f, ensure_ascii=True, indent=2)

        return {
            "reanalyzed": reanalyzed,
            "avg_error": avg_error,
            "max_error": max_error,
            "min_error": min_error,
            "top_errors": top_errors
        }

    def train_step_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, policies, values = self.replay_buffer.sample(self.batch_size)
        states_t = torch.FloatTensor(states).to(self.device)
        policies_t = torch.FloatTensor(policies).to(self.device)
        values_t = torch.FloatTensor(values).unsqueeze(1).to(self.device)

        self.network.train()
        pred_policies, pred_values = self.network(states_t)

        policy_loss = -torch.sum(policies_t * pred_policies) / self.batch_size
        value_loss = torch.mean((pred_values - values_t) ** 2)
        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.train_step += 1
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "total_loss": float(total_loss.item())
        }

    def train_epoch(self, num_batches=50):
        policy_losses = []
        value_losses = []
        total_losses = []

        for _ in range(num_batches):
            stats = self.train_step_batch()
            if stats is None:
                continue
            policy_losses.append(stats["policy_loss"])
            value_losses.append(stats["value_loss"])
            total_losses.append(stats["total_loss"])

        if not total_losses:
            return None

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "total_loss": float(np.mean(total_losses)),
            "num_batches": len(total_losses)
        }

    def evaluate_vs_mcts_async(self, num_games=20, mcts_sims=100, num_workers=None):
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 2)

        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        server = BatchInferenceServer(
            network_state_dict=network_state_dict,
            rows=self.rows,
            cols=self.cols,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels,
            batch_size=self.infer_batch_size,
            timeout=self.infer_timeout,
            stats_queue=None
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
                target=_eval_worker_loop_mcts,
                args=(
                    game_queue,
                    result_queue,
                    worker_id,
                    server.input_queue,
                    output_queues[worker_id],
                    self.rows,
                    self.cols,
                    self.num_res_blocks,
                    self.num_channels,
                    self.num_simulations,
                    self.c_puct,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                    mcts_sims
                )
            )
            p.start()
            workers.append(p)

        ai_players = [1 if i < num_games // 2 else 2 for i in range(num_games)]
        for ai_player in ai_players:
            game_queue.put(ai_player)

        wins = 0
        game_records = []
        for _ in range(num_games):
            winner, ai_player, moves, states = result_queue.get()
            if winner == ai_player:
                wins += 1
            game_records.append({
                "winner": winner,
                "ai_player": ai_player,
                "moves": moves,
                "states": states
            })

        for _ in workers:
            game_queue.put(None)
        for p in workers:
            p.join()

        server.input_queue.put(None)
        server_process.join()

        return wins / num_games, game_records


def train_alphazero(
    num_iterations=200,
    games_per_iteration=100,
    batches_per_iteration=200,
    eval_games=20,
    num_simulations=80,
    save_interval=5,
    temperature_threshold=20,
    c_puct=1.5,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    fresh_start=False,
    rows=6,
    cols=7,
    num_res_blocks=3,
    num_channels=64,
    batch_size=128,
    buffer_size=50000,
    checkpoint_dir="connect4/checkpoints",
    playout_full_prob=0.25,
    playout_full_cap=400,
    playout_fast_cap=80,
    infer_batch_size=64,
    infer_timeout=0.005,
    use_forced_playouts=False,
    use_policy_target_pruning=False,
    forced_playout_k=2.0,
    selfplay_workers=None,
    reanalyze_workers=0,
    reanalyze_ratio=0.0,
    reanalyze_min_samples=0,
    reanalyze_candidate_pool=2000,
    reanalyze_sims=None,
    reanalyze_debug=False,
    reanalyze_debug_top_k=5
):
    log_dir = f"connect4/runs/alphazero_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    os.makedirs("connect4/logs", exist_ok=True)
    log_file = f"connect4/logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    history = {
        "iterations": [],
        "policy_loss": [],
        "value_loss": [],
        "total_loss": [],
        "win_rate": [],
        "avg_moves": [],
        "buffer_size": [],
        "p1_wins": [],
        "p2_wins": [],
        "draws": []
    }

    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = AlphaZeroTrainer(
        rows=rows,
        cols=cols,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        batch_size=batch_size,
        buffer_size=buffer_size,
        lr=0.001,
        playout_full_prob=playout_full_prob,
        playout_full_cap=playout_full_cap,
        playout_fast_cap=playout_fast_cap,
        infer_batch_size=infer_batch_size,
        infer_timeout=infer_timeout,
        use_forced_playouts=use_forced_playouts,
        use_policy_target_pruning=use_policy_target_pruning,
        forced_playout_k=forced_playout_k
    )

    latest_ckpt = os.path.join(checkpoint_dir, "alphazero_latest.pt")
    start_iteration = 1
    if not fresh_start and os.path.exists(latest_ckpt):
        checkpoint = trainer.load(latest_ckpt)
        if checkpoint:
            history = checkpoint.get("history", history)
            start_iteration = checkpoint.get("iteration", 0) + 1
            print(f"Resuming from iteration {start_iteration}")

    trainer.debug_flip_once()

    if reanalyze_workers <= 0:
        reanalyze_workers = selfplay_workers if selfplay_workers else max(1, mp.cpu_count() - 2)

    for iteration in range(start_iteration, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{num_iterations}")
        print("=" * 60)

        print(f"\n[1] Self-Play ({games_per_iteration} games)...")
        start_time = time.time()
        sp_stats = trainer.collect_self_play_data(
            num_games=games_per_iteration,
            use_multiprocessing=True,
            num_workers=selfplay_workers
        )
        sp_time = time.time() - start_time

        print(f"    Games: {sp_stats['games']} | Avg Moves: {sp_stats['avg_moves']:.1f} | "
              f"P1/P2/D: {sp_stats['p1_wins']}/{sp_stats['p2_wins']}/{sp_stats['draws']} | "
              f"Samples: {sp_stats.get('samples_raw', 0)} | Buffer: {sp_stats['buffer_size']} | Time: {sp_time:.1f}s")

        writer.add_scalar('SelfPlay/avg_moves', sp_stats['avg_moves'], iteration)
        writer.add_scalar('SelfPlay/p1_wins', sp_stats['p1_wins'], iteration)
        writer.add_scalar('SelfPlay/p2_wins', sp_stats['p2_wins'], iteration)
        writer.add_scalar('SelfPlay/draws', sp_stats['draws'], iteration)
        writer.add_scalar('SelfPlay/buffer_size', sp_stats['buffer_size'], iteration)
        writer.add_scalar('SelfPlay/time_seconds', sp_time, iteration)

        if reanalyze_ratio > 0 and reanalyze_workers > 0:
            target_samples = int(sp_stats.get("samples_raw", 0) * reanalyze_ratio)
            target_samples = max(target_samples, reanalyze_min_samples)
            if target_samples > 0:
                print(f"\n[1.5] Reanalyze ({target_samples} samples)...")
            start_time = time.time()
            debug_path = None
            if reanalyze_debug:
                debug_path = os.path.join("connect4", "logs", f"reanalyze_iter_{iteration}.json")
            re_stats = trainer.reanalyze_buffer(
                num_samples=target_samples,
                num_workers=reanalyze_workers,
                candidate_pool=reanalyze_candidate_pool,
                reanalyze_sims=reanalyze_sims,
                debug_path=debug_path,
                debug_top_k=reanalyze_debug_top_k
            )
            re_time = time.time() - start_time
            print(
                f"    Reanalyzed: {re_stats['reanalyzed']} | "
                f"Avg value error: {re_stats['avg_error']:.4f} | "
                f"Max/Min: {re_stats['max_error']:.4f}/{re_stats['min_error']:.4f} | "
                f"Time: {re_time:.1f}s"
            )
            if reanalyze_debug and re_stats.get("top_errors"):
                top_err_str = ", ".join([f"{e:.4f}" for e in re_stats["top_errors"]])
                print(f"    Top value errors: {top_err_str}")
                writer.add_scalar('Reanalyze/count', re_stats['reanalyzed'], iteration)
                writer.add_scalar('Reanalyze/avg_error', re_stats['avg_error'], iteration)
                writer.add_scalar('Reanalyze/max_error', re_stats['max_error'], iteration)
                writer.add_scalar('Reanalyze/min_error', re_stats['min_error'], iteration)
                writer.add_scalar('Reanalyze/time_seconds', re_time, iteration)
            else:
                print("\n[1.5] Reanalyze skipped (no samples selected).")

        print(f"\n[2] Training ({batches_per_iteration} batches)...")
        start_time = time.time()
        train_stats = trainer.train_epoch(num_batches=batches_per_iteration)
        train_time = time.time() - start_time

        if train_stats:
            print(f"    Policy Loss: {train_stats['policy_loss']:.4f} | "
                  f"Value Loss: {train_stats['value_loss']:.4f} | "
                  f"Total: {train_stats['total_loss']:.4f}")
            print(f"    Batches: {train_stats['num_batches']} | Time: {train_time:.1f}s")
            writer.add_scalar('Loss/policy', train_stats['policy_loss'], iteration)
            writer.add_scalar('Loss/value', train_stats['value_loss'], iteration)
            writer.add_scalar('Loss/total', train_stats['total_loss'], iteration)
            writer.add_scalar('Training/time_seconds', train_time, iteration)

            history['iterations'].append(iteration)
            history['policy_loss'].append(train_stats['policy_loss'])
            history['value_loss'].append(train_stats['value_loss'])
            history['total_loss'].append(train_stats['total_loss'])
            history['avg_moves'].append(sp_stats['avg_moves'])
            history['buffer_size'].append(sp_stats['buffer_size'])
            history['p1_wins'].append(sp_stats['p1_wins'])
            history['p2_wins'].append(sp_stats['p2_wins'])
            history['draws'].append(sp_stats['draws'])

        if iteration % 10 == 0:
            print("\n[Benchmark] Async self-play bottleneck check...")
            _benchmark_async_selfplay(trainer, num_games=10)

        if iteration % save_interval == 0:
            print(f"\n[3] Evaluation vs MCTS ladder ({eval_games} games each)...")
            ladder_sims = (100, 200, 300)
            ladder_stage = int(history.get("ladder_stage", 0))
            last_win_rate = None
            eval_records = []

            if ladder_stage < len(ladder_sims):
                sims = ladder_sims[ladder_stage]
                start_time = time.time()
                win_rate, records = trainer.evaluate_vs_mcts_async(
                    num_games=eval_games,
                    mcts_sims=sims
                )
                eval_time = time.time() - start_time
                last_win_rate = win_rate
                print(f"    MCTS {sims}: Win Rate {win_rate*100:.1f}% | Time: {eval_time:.1f}s")
                writer.add_scalar(f'Evaluation/mcts_{sims}_win_rate', win_rate, iteration)
                writer.add_scalar(f'Evaluation/mcts_{sims}_time_seconds', eval_time, iteration)
                eval_records = records
                if win_rate >= 0.9:
                    history["ladder_stage"] = ladder_stage + 1

            if last_win_rate is not None:
                history['win_rate'].append(last_win_rate)
                writer.add_scalar('Evaluation/win_rate', last_win_rate, iteration)

            with open(log_file, 'a', encoding='utf-8') as f:
                win_rate_pct = last_win_rate * 100 if last_win_rate is not None else 0.0
                f.write(f"[Iter {iteration}] "
                        f"Loss: {train_stats['total_loss']:.4f} "
                        f"(P:{train_stats['policy_loss']:.4f}, V:{train_stats['value_loss']:.4f}) | "
                        f"WinRate: {win_rate_pct:.1f}% | "
                        f"AvgMoves: {sp_stats['avg_moves']:.1f} | "
                        f"Buffer: {sp_stats['buffer_size']}\n")

            debug_dir = os.path.join("connect4/debug_eval", f"iter_{iteration}")
            os.makedirs(debug_dir, exist_ok=True)
            try:
                import pickle
                with open(os.path.join(debug_dir, "mcts_eval.pkl"), "wb") as f:
                    pickle.dump(eval_records, f)
            except Exception as e:
                print(f"    Debug save failed: {e}")
            for record in eval_records[:2]:
                _render_recorded_game(record, rows, cols)

            trainer.save(os.path.join(checkpoint_dir, f"alphazero_iter{iteration}.pt"), iteration=iteration, history=history)
            trainer.save(os.path.join(checkpoint_dir, "alphazero_latest.pt"), iteration=iteration, history=history)

    writer.close()
    trainer.save(os.path.join(checkpoint_dir, "alphazero_final.pt"), iteration=num_iterations, history=history)
    trainer.save(os.path.join(checkpoint_dir, "alphazero_latest.pt"), iteration=num_iterations, history=history)
    return trainer
