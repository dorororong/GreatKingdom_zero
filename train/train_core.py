"""
AlphaZero Training - Self-Play 학습 시스템

구성요소:
1. Self-Play: 자가 대전으로 학습 데이터 생성 (멀티프로세싱 지원)
2. Replay Buffer: 경험 저장소
3. Training: 신경망 학습
4. Evaluation: 성능 평가
"""

import numpy as np
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import time
import os
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from env.env import GreatKingdomEnv
from env.env_fast import GreatKingdomEnvFast
from game_result import winner_id_from_step
from network import AlphaZeroNetwork, encode_board_from_state, infer_head_type_from_state_dict, load_state_dict_safe
from mcts_alphazero import AlphaZeroMCTS
from symmetry import CANON_TRANSFORMS, _apply_transform_array, transform_action_probs
from train.eval_utils import (
    _init_eval_worker,
    _eval_one_game,
    _eval_worker_loop_random,
    _eval_worker_loop_mcts,
    _eval_worker_loop_best,
    _render_recorded_game
)
from train.selfplay_workers import (
    BatchInferenceServer,
    _benchmark_async_selfplay,
    _get_ownership_target,
    _init_worker,
    _worker_loop,
    _scheduled_temperature
)


# ========== 멀티프로세싱용 전역 함수 ==========

# Helper utilities moved to train/selfplay_workers.py and train/eval_utils.py

def _classify_eval_result(info):
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


def _winner_from_eval_info(info, final_player, fallback=None):
    if not isinstance(info, dict):
        return fallback if fallback is not None else 0
    result = info.get("result", "")
    if result == "Territory Count":
        winner_label = info.get("winner")
        if winner_label == "Black":
            return 1
        if winner_label == "White":
            return 2
        if winner_label == "Draw":
            return 0
    if result == "Win by Capture":
        if final_player in (1, 2):
            return final_player
    if result == "Loss by Suicide":
        if final_player == 1:
            return 2
        if final_player == 2:
            return 1
    return fallback if fallback is not None else 0


def _summarize_eval_records(records):
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

    for record in records:
        info = record.get("info", {})
        win_type = _classify_eval_result(info)
        winner = _winner_from_eval_info(info, record.get("final_player"), record.get("winner"))
        moves = record.get("moves", 0)
        diff = None
        if win_type == "territory":
            black = info.get("black_territory", 0) if isinstance(info, dict) else 0
            white = info.get("white_territory", 0) if isinstance(info, dict) else 0
            diff = abs(int(black) - int(white))

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


def _format_eval_stats(stats):
    def _avg(total, count):
        return (total / count) if count else 0.0

    lines = []
    for color, name in ((1, "Black"), (2, "White")):
        count = stats["capture"][color]["count"]
        moves = stats["capture"][color]["moves"]
        lines.append(f"Capture {name}: {count} | Avg Moves: {_avg(moves, count):.2f}")
    for color, name in ((1, "Black"), (2, "White")):
        count = stats["territory"][color]["count"]
        moves = stats["territory"][color]["moves"]
        diff = stats["territory"][color]["diff"]
        lines.append(
            f"Territory {name}: {count} | Avg Diff: {_avg(diff, count):.2f} | Avg Moves: {_avg(moves, count):.2f}"
        )
    if stats["suicide"][1] or stats["suicide"][2]:
        lines.append(f"Suicide Wins - Black: {stats['suicide'][1]}, White: {stats['suicide'][2]}")
    if stats["draws"] or stats["other"]:
        lines.append(f"Draws: {stats['draws']} | Other: {stats['other']}")
    return lines


def _summarize_selfplay_results(results):
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
        win_type = r.get("win_type", "other")
        winner = r.get("winner", 0)
        moves = r.get("moves", 0)
        diff = r.get("territory_diff")

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


def _format_selfplay_stats(stats):
    def _avg(total, count):
        return (total / count) if count else 0.0

    lines = []
    for color, name in ((1, "Black"), (2, "White")):
        count = stats["capture"][color]["count"]
        moves = stats["capture"][color]["moves"]
        lines.append(f"Capture wins - {name}: {count} | Avg Moves: {_avg(moves, count):.2f}")
    for color, name in ((1, "Black"), (2, "White")):
        count = stats["territory"][color]["count"]
        moves = stats["territory"][color]["moves"]
        lines.append(f"Territory wins - {name}: {count} | Avg Moves: {_avg(moves, count):.2f}")
    if stats["suicide"][1] or stats["suicide"][2]:
        lines.append(f"Suicide wins - Black: {stats['suicide'][1]}, White: {stats['suicide'][2]}")
    if stats["draws"] or stats["other"]:
        lines.append(f"Draws: {stats['draws']} | Other: {stats['other']}")
    return lines


def _extract_stone_positions(board):
    arr = np.asarray(board)
    black = np.argwhere(arr == 1)
    white = np.argwhere(arr == 2)
    black_positions = [[int(r), int(c)] for r, c in black]
    white_positions = [[int(r), int(c)] for r, c in white]
    return black_positions, white_positions


def _build_selfplay_record(actions, winner, win_type, moves, territory_diff, board_size, komi, center_wall):
    winner_label = {0: "Draw", 1: "Black", 2: "White"}.get(winner, "Unknown")
    return {
        "board_size": int(board_size),
        "komi": float(komi),
        "center_wall": bool(center_wall),
        "winner": int(winner),
        "winner_label": winner_label,
        "win_type": win_type,
        "moves": int(moves),
        "territory_diff": None if territory_diff is None else int(territory_diff),
        "actions": [int(a) for a in actions]
    }


def _save_selfplay_records(records, output_path, meta=None):
    if not records:
        return
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    payload = {
        "meta": meta or {},
        "records": records
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

class ReplayBuffer:
    """학습 데이터 저장소"""
    
    def __init__(self, max_size=10000, default_ownership_weight=0.2, default_win_weight=0.0, default_win_type_weight=0.0, board_size=5):
        self.buffer = deque(maxlen=max_size)
        self.default_ownership_weight = float(default_ownership_weight)
        self.default_win_weight = float(default_win_weight)
        self.default_win_type_weight = float(default_win_type_weight)
        self.board_size = int(board_size)
    
    def push(
        self,
        state,
        policy,
        value,
        ownership,
        ownership_weight=None,
        win_target=None,
        win_weight=None,
        win_type_target=None,
        win_type_weight=None
    ):
        """데이터 추가"""
        if ownership_weight is None:
            ownership_weight = self.default_ownership_weight
        if win_target is None:
            win_target = 0.5
        if win_weight is None:
            win_weight = self.default_win_weight
        if win_type_target is None:
            win_type_target = 0.5
        if win_type_weight is None:
            win_type_weight = self.default_win_type_weight
        self.buffer.append(
            (
                state,
                policy,
                value,
                ownership,
                float(ownership_weight),
                float(win_target),
                float(win_weight),
                float(win_type_target),
                float(win_type_weight)
            )
        )
    
    def sample(self, batch_size):
        """랜덤 샘플링"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = []
        policies = []
        values = []
        ownerships = []
        ownership_weights = []
        win_targets = []
        win_weights = []
        win_type_targets = []
        win_type_weights = []
        for item in batch:
            if len(item) == 4:
                state, policy, value, ownership = item
                ownership_weight = self.default_ownership_weight
                win_target = 0.5
                win_weight = self.default_win_weight
                win_type_target = 0.5
                win_type_weight = self.default_win_type_weight
            elif len(item) == 5:
                state, policy, value, ownership, ownership_weight = item
                win_target = 0.5
                win_weight = self.default_win_weight
                win_type_target = 0.5
                win_type_weight = self.default_win_type_weight
            elif len(item) == 7:
                state, policy, value, ownership, ownership_weight, win_target, win_weight = item
                win_type_target = 0.5
                win_type_weight = self.default_win_type_weight
            else:
                state, policy, value, ownership, ownership_weight, win_target, win_weight, win_type_target, win_type_weight = item
            transform = random.choice(CANON_TRANSFORMS)
            t_state = _apply_transform_array(state, transform)
            t_state = np.ascontiguousarray(t_state)
            t_policy = transform_action_probs(policy, self.board_size, transform)
            if ownership is not None:
                t_owner = _apply_transform_array(ownership, transform)
                t_owner = np.ascontiguousarray(t_owner)
            else:
                t_owner = ownership
            states.append(t_state)
            policies.append(t_policy)
            values.append(value)
            ownerships.append(t_owner)
            ownership_weights.append(ownership_weight)
            win_targets.append(win_target)
            win_weights.append(win_weight)
            win_type_targets.append(win_type_target)
            win_type_weights.append(win_type_weight)
        return (
            np.array(states),
            np.array(policies),
            np.array(values, dtype=np.float32),
            np.array(ownerships, dtype=np.float32),
            np.array(ownership_weights, dtype=np.float32),
            np.array(win_targets, dtype=np.float32),
            np.array(win_weights, dtype=np.float32),
            np.array(win_type_targets, dtype=np.float32),
            np.array(win_type_weights, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class AlphaZeroTrainer:
    """AlphaZero 학습 관리자"""
    
    def __init__(
        self,
        board_size=5,
        num_res_blocks=3,
        num_channels=64,
        num_simulations=40,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        lr=0.001,
        batch_size=64,
        buffer_size=10000,
        device=None,
        center_wall=True,
        komi=0,
        use_fast_env=False,
        playout_full_prob=0.25,
        playout_full_cap=600,
        playout_fast_cap=100,
        infer_batch_size=256,
        infer_timeout=0.05,
        eval_infer_batch_size=None,
        eval_infer_timeout=None,
        mcts_eval_batch_size=1,
        mcts_profile=False,
        mcts_profile_every=0,
        temperature_schedule=None,
        use_forced_playouts=False,
        use_policy_target_pruning=False,
        forced_playout_k=2.0,
        use_liberty_features=True,
        liberty_bins=2,
        use_last_moves=False,
        network_head="fc",
        freeze_backbone=False,
        freeze_backbone_blocks=None,
        freeze_backbone_input=True,
        ownership_loss_weight=0.2,
        ownership_loss_weight_capture=0.1,
        win_loss_weight=0.1,
        win_type_loss_weight=0.1,
        train_buffer_min_factor=2.0,
        cache_debug_samples=0,
        cache_max_entries=50000
    ):
        self.board_size = board_size
        self.center_wall = center_wall
        self.komi = komi
        self.use_fast_env = use_fast_env
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
        self.eval_infer_batch_size = eval_infer_batch_size if eval_infer_batch_size is not None else infer_batch_size
        self.eval_infer_timeout = eval_infer_timeout if eval_infer_timeout is not None else infer_timeout
        self.mcts_eval_batch_size = mcts_eval_batch_size
        self.mcts_profile = mcts_profile
        self.mcts_profile_every = mcts_profile_every
        self.temperature_schedule = temperature_schedule
        self.use_forced_playouts = use_forced_playouts
        self.use_policy_target_pruning = use_policy_target_pruning
        self.forced_playout_k = forced_playout_k
        self.use_liberty_features = use_liberty_features
        self.liberty_bins = liberty_bins
        self.use_last_moves = use_last_moves
        self.network_head = network_head
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_blocks = freeze_backbone_blocks
        self.freeze_backbone_input = freeze_backbone_input
        self.ownership_loss_weight = float(ownership_loss_weight)
        self.ownership_loss_weight_capture = float(ownership_loss_weight_capture)
        self.win_loss_weight = float(win_loss_weight)
        self.win_type_loss_weight = float(win_type_loss_weight)
        self.cache_debug_samples = int(cache_debug_samples)
        self.cache_max_entries = int(cache_max_entries)
        self.train_buffer_min_factor = float(train_buffer_min_factor)
        self.lr = lr
        self.weight_decay = 1e-4
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        # 환경
        env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
        self.env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)
        
        # 네트워크
        self.network = AlphaZeroNetwork(
            board_size=board_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            use_liberty_features=use_liberty_features,
            liberty_bins=liberty_bins,
            use_last_moves=use_last_moves,
            head_type=network_head
        ).to(self.device)

        if self.freeze_backbone or self.freeze_backbone_blocks is not None or self.freeze_backbone_input:
            self._apply_freeze()

        # MCTS
        self.mcts = AlphaZeroMCTS(
            self.network, self.env,
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
        self.net_version = 0
        if hasattr(self.mcts, "set_net_version"):
            self.mcts.set_net_version(self.net_version)
        
        # Optimizer (exclude frozen params if any)
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            max_size=buffer_size,
            default_ownership_weight=self.ownership_loss_weight,
            board_size=self.board_size
        )
        
        # 학습 통계
        self.train_step = 0
        self.total_games = 0
    
    def self_play_game(self, temperature_threshold=20):
        """
        자가 대전 한 게임 수행
        
        Args:
            temperature_threshold: 기존 API 호환용 (현재는 보드 크기 기반 스케줄 사용)
        
        Returns:
            game_data: [(state, policy, None), ...] - value는 게임 종료 후 채움
        """
        self.env.reset()
        game_data = []
        move_count = 0
        actions = []
        max_moves = max(100, self.board_size * self.board_size + 20)
        last_moves = (None, None) if self.use_last_moves else None
        info = {}
        reward = 0
        
        while True:
            # 현재 상태 인코딩
            if self.use_last_moves:
                state = (
                    self.env.board.copy(),
                    self.env.current_player,
                    self.env.consecutive_passes,
                    last_moves
                )
            else:
                state = (
                    self.env.board.copy(),
                    self.env.current_player,
                    self.env.consecutive_passes
                )
            encoded_state = encode_board_from_state(
                state[0], state[1], self.board_size, last_moves,
                use_liberty_features=self.use_liberty_features,
                liberty_bins=self.liberty_bins,
                use_last_moves=self.use_last_moves
            )
            
            # 온도 설정 (보드 크기 기반 스케줄)
            temperature = _scheduled_temperature(move_count, self.temperature_schedule, self.board_size)

            full_search = np.random.random() < self.playout_full_prob
            if full_search:
                self.mcts.num_simulations = self.playout_full_cap
                action, action_probs = self.mcts.run(
                    state, temperature=temperature, add_root_noise=True
                )
                game_data.append((encoded_state, action_probs))
            else:
                self.mcts.num_simulations = self.playout_fast_cap
                action, _ = self.mcts.run(
                    state, temperature=temperature, add_root_noise=False
                )
            
            # 행동 수행
            actions.append(int(action))
            _, reward, done, info = self.env.step(action)
            if self.use_last_moves and action != self.env.pass_action:
                last_moves = (action, last_moves[0])
            move_count += 1
            
            if done:
                # 승패 결정
                winner = winner_id_from_step(info, reward, self.env.current_player)
                
                break
            
            # 무한 루프 방지
            if move_count > max_moves:
                winner = 0
                break
        
        win_type = _classify_eval_result(info)
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
        ownership_target = _get_ownership_target(self.env)
        ownership_weight = (
            self.ownership_loss_weight_capture
            if win_type == "capture"
            else self.ownership_loss_weight
        )

        # 각 상태에 대한 value 계산 (승자 관점)
        # game_data의 각 상태에서 플레이어가 누구였는지 추적
        final_data = []
        
        for i, (enc_state, policy) in enumerate(game_data):
            # 이 시점의 플레이어 (i번째 수를 두기 직전)
            # enc_state[3]이 1이면 흑 차례, 0이면 백 차례
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

    def _apply_freeze(self):
        for param in self.network.parameters():
            param.requires_grad = True

        if self.freeze_backbone_input:
            for name, param in self.network.named_parameters():
                if name.startswith("conv_input.") or name.startswith("bn_input."):
                    param.requires_grad = False

        freeze_blocks = self.freeze_backbone_blocks
        if freeze_blocks is None and self.freeze_backbone:
            freeze_blocks = len(self.network.res_blocks)
        if freeze_blocks is None:
            return

        for idx, block in enumerate(self.network.res_blocks):
            if idx < freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

    def _set_backbone_eval(self):
        if self.freeze_backbone_input:
            self.network.bn_input.eval()
        freeze_blocks = self.freeze_backbone_blocks
        if freeze_blocks is None and self.freeze_backbone:
            freeze_blocks = len(self.network.res_blocks)
        if freeze_blocks is None:
            return
        for idx, block in enumerate(self.network.res_blocks):
            if idx < freeze_blocks:
                block.eval()

    def _rebuild_optimizer(self):
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)

    def set_freeze_backbone_blocks(self, freeze_blocks, freeze_input=None):
        if freeze_input is not None:
            self.freeze_backbone_input = freeze_input
        self.freeze_backbone_blocks = freeze_blocks
        self.freeze_backbone = False
        self._apply_freeze()
        self._rebuild_optimizer()


    def self_play_game_debug(self, temperature_threshold=20, top_k=5):
        'Debug: render first game and print key priors only.'
        self.env.reset()
        game_data = []
        move_count = 0
        max_moves = max(100, self.board_size * self.board_size + 20)
        last_moves = (None, None) if self.use_last_moves else None

        print("\n=== Debug Self-Play Game ===")
        self.env.render()

        while True:
            if self.use_last_moves:
                state = (
                    self.env.board.copy(),
                    self.env.current_player,
                    self.env.consecutive_passes,
                    last_moves
                )
            else:
                state = (
                    self.env.board.copy(),
                    self.env.current_player,
                    self.env.consecutive_passes
                )
            encoded_state = encode_board_from_state(
                state[0], state[1], self.board_size, last_moves,
                use_liberty_features=self.use_liberty_features,
                liberty_bins=self.liberty_bins,
                use_last_moves=self.use_last_moves
            )

            temperature = _scheduled_temperature(move_count, self.board_size)
            full_search = np.random.random() < self.playout_full_prob
            if full_search:
                self.mcts.num_simulations = self.playout_full_cap
                action, action_probs, mcts_info = self.mcts.run_with_info(
                    state, temperature=temperature, add_root_noise=True, top_k=top_k
                )
                game_data.append((encoded_state, action_probs))
            else:
                self.mcts.num_simulations = self.playout_fast_cap
                action, action_probs, mcts_info = self.mcts.run_with_info(
                    state, temperature=temperature, add_root_noise=False, top_k=top_k
                )

            print(f"\nMove {move_count+1} | Player: {'Black' if self.env.current_player==1 else 'White'}")
            print(f"  Selected action: {action}")
            def _fmt_action(idx):
                if idx == self.env.pass_action:
                    return "PASS"
                r, c = divmod(idx, self.board_size)
                return f"({r},{c})"

            print("  Top priors (before noise):")
            for idx, prob in mcts_info["top_priors_before"]:
                print(f"    {_fmt_action(idx)}: {prob:.4f}")
            print("  Top priors (after noise):")
            for idx, prob in mcts_info["top_priors_after"]:
                print(f"    {_fmt_action(idx)}: {prob:.4f}")

            _, reward, done, info = self.env.step(action)
            if self.use_last_moves and action != self.env.pass_action:
                last_moves = (action, last_moves[0])
            move_count += 1
            self.env.render()

            if done:
                winner = winner_id_from_step(info, reward, self.env.current_player)
                break

            if move_count > max_moves:
                winner = 0
                print("Game end | move limit")
                break

        win_type = _classify_eval_result(info)
        ownership_target = _get_ownership_target(self.env)
        ownership_weight = (
            self.ownership_loss_weight_capture
            if win_type == "capture"
            else self.ownership_loss_weight
        )
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

        return final_data, winner, move_count

    
    def collect_self_play_data(self, num_games=10, verbose=True, use_multiprocessing=True, temperature_threshold=20, temperature_schedule=None, use_async_inference=False, record_interval=0, record_limit=None):
        """
        여러 게임의 자가 대전 데이터 수집
        
        Args:
            num_games: 수집할 게임 수
            verbose: 진행 상황 출력
            use_multiprocessing: 멀티프로세싱 사용 여부
        """
        total_moves = 0
        wins = {0: 0, 1: 0, 2: 0}  # 무승부, 흑승, 백승
        detail_results = []
        records = []
        
        schedule = temperature_schedule if temperature_schedule is not None else self.temperature_schedule
        if use_multiprocessing and num_games >= 4:
            # 멀티프로세싱으로 병렬 실행
            if use_async_inference:
                return self._collect_self_play_multiprocess_batched(num_games, verbose, schedule, record_interval, record_limit)
            return self._collect_self_play_multiprocess(num_games, verbose, schedule, record_interval, record_limit)
        
        # 싱글 프로세스 실행
        for game_idx in range(num_games):
            game_data, winner, moves, win_type, territory_diff, actions = self.self_play_game(
                temperature_threshold=temperature_threshold
            )
            
            # Replay Buffer에 추가
            for state, policy, value, ownership, ownership_weight, win_target, win_weight, win_type_target, win_type_weight in game_data:
                self.replay_buffer.push(
                    state,
                    policy,
                    value,
                    ownership,
                    ownership_weight,
                    win_target,
                    win_weight,
                    win_type_target,
                    win_type_weight
                )
            
            total_moves += moves
            wins[winner] += 1
            self.total_games += 1
            detail_results.append({
                "win_type": win_type,
                "winner": winner,
                "moves": moves,
                "territory_diff": territory_diff
            })
            if record_interval and ((game_idx + 1) % record_interval == 0):
                if record_limit is None or len(records) < record_limit:
                    records.append(
                        _build_selfplay_record(
                            actions,
                            winner,
                            win_type,
                            moves,
                            territory_diff,
                            self.board_size,
                            self.komi,
                            self.center_wall
                        )
                    )
            
            if verbose and ((game_idx + 1) % 100 == 0 or (game_idx + 1) == num_games):
                print(f"  Game {game_idx + 1}/{num_games} | "
                      f"Moves: {moves} | Winner: {['Draw', 'Black', 'White'][winner]}")
        
        detail_summary = _summarize_selfplay_results(detail_results)
        return {
            "games": num_games,
            "avg_moves": total_moves / num_games,
            "black_wins": wins[1],
            "white_wins": wins[2],
            "draws": wins[0],
            "buffer_size": len(self.replay_buffer),
            "detail": detail_summary,
            "records": records
        }
    
    def _collect_self_play_multiprocess(self, num_games, verbose=True, temperature_schedule=None, record_interval=0, record_limit=None):
        """멀티프로세싱으로 자가 대전 데이터 수집"""
        # 워커 수 설정 (CPU 코어 수 - 2)
        num_workers = max(1, mp.cpu_count() - 2)
        
        if verbose:
            print(f"  Using {num_workers} workers for parallel self-play...")
        
        # 네트워크 가중치를 CPU로 이동하여 공유
        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        
        total_moves = 0
        wins = {0: 0, 1: 0, 2: 0}
        games_completed = 0
        detail_results = []
        records = []
        
        # ProcessPoolExecutor로 병렬 실행
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(
                network_state_dict,
                self.board_size,
                self.num_res_blocks,
                self.num_channels,
                self.num_simulations,
                self.c_puct,
                self.dirichlet_alpha,
                self.dirichlet_epsilon,
                self.center_wall,
                self.komi,
                self.playout_full_prob,
                self.playout_full_cap,
                self.playout_fast_cap,
                self.mcts_eval_batch_size,
                self.mcts_profile,
                self.mcts_profile_every,
                self.use_forced_playouts,
                self.use_policy_target_pruning,
                self.forced_playout_k,
                self.use_fast_env,
                self.use_liberty_features,
                self.liberty_bins,
                self.use_last_moves,
                self.network_head,
                self.ownership_loss_weight,
                self.ownership_loss_weight_capture
            )
        ) as executor:
            # 모든 게임을 제출
            futures = [executor.submit(_play_one_game, temperature_schedule) for _ in range(num_games)]
            
            # 결과 수집
            for future in as_completed(futures):
                try:
                    game_data, winner, moves, win_type, territory_diff, actions = future.result()
                    
                    # Replay Buffer에 추가
                    for state, policy, value, ownership, ownership_weight, win_target, win_weight, win_type_target, win_type_weight in game_data:
                        self.replay_buffer.push(
                            state,
                            policy,
                            value,
                            ownership,
                            ownership_weight,
                            win_target,
                            win_weight,
                            win_type_target,
                            win_type_weight
                        )
                    
                    total_moves += moves
                    wins[winner] += 1
                    self.total_games += 1
                    games_completed += 1
                    detail_results.append({
                        "win_type": win_type,
                        "winner": winner,
                        "moves": moves,
                        "territory_diff": territory_diff
                    })
                    if record_interval and (games_completed % record_interval == 0):
                        if record_limit is None or len(records) < record_limit:
                            records.append(
                                _build_selfplay_record(
                                    actions,
                                    winner,
                                    win_type,
                                    moves,
                                    territory_diff,
                                    self.board_size,
                                    self.komi,
                                    self.center_wall
                                )
                            )
                    
                    if verbose and (games_completed % 100 == 0 or games_completed == num_games):
                        print(f"  Completed {games_completed}/{num_games} games...")
                        
                except Exception as e:
                    print(f"  Warning: Game failed with error: {e}")
        
        detail_summary = _summarize_selfplay_results(detail_results)
        return {
            "games": num_games,
            "avg_moves": total_moves / max(1, games_completed),
            "black_wins": wins[1],
            "white_wins": wins[2],
            "draws": wins[0],
            "buffer_size": len(self.replay_buffer),
            "detail": detail_summary,
            "records": records
        }

    def _collect_self_play_multiprocess_batched(self, num_games, verbose=True, temperature_schedule=None, record_interval=0, record_limit=None):
        """배치 추론 서버로 자가 대전 데이터 수집."""
        num_workers = max(1, mp.cpu_count() - 2)
        if verbose:
            print(f"  Using {num_workers} workers with async inference...")

        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        stats_queue = mp.Queue()
        server = BatchInferenceServer(
            network_state_dict=network_state_dict,
            board_size=self.board_size,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels,
            batch_size=self.infer_batch_size,
            timeout=self.infer_timeout,
            stats_queue=stats_queue,
            use_liberty_features=self.use_liberty_features,
            liberty_bins=self.liberty_bins,
            use_last_moves=self.use_last_moves,
            network_head=self.network_head,
            cache_debug_samples=self.cache_debug_samples,
            cache_max_entries=self.cache_max_entries
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
                    self.board_size,
                    self.num_res_blocks,
                    self.num_channels,
                    self.num_simulations,
                    self.c_puct,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                    self.center_wall,
                    self.komi,
                    self.playout_full_prob,
                    self.playout_full_cap,
                    self.playout_fast_cap,
                    self.mcts_eval_batch_size,
                    self.mcts_profile,
                    self.mcts_profile_every,
                    self.use_forced_playouts,
                    self.use_policy_target_pruning,
                    self.forced_playout_k,
                    self.use_fast_env,
                    self.use_liberty_features,
                    self.liberty_bins,
                    self.use_last_moves,
                    self.ownership_loss_weight,
                    self.ownership_loss_weight_capture
                )
            )
            p.start()
            workers.append(p)

        start_time = time.time()
        for _ in range(num_games):
            game_queue.put(temperature_schedule)

        total_moves = 0
        wins = {0: 0, 1: 0, 2: 0}
        games_completed = 0
        detail_results = []
        records = []

        for _ in range(num_games):
            game_data, winner, moves, win_type, territory_diff, actions = result_queue.get()
            for state, policy, value, ownership, ownership_weight, win_target, win_weight, win_type_target, win_type_weight in game_data:
                self.replay_buffer.push(
                    state,
                    policy,
                    value,
                    ownership,
                    ownership_weight,
                    win_target,
                    win_weight,
                    win_type_target,
                    win_type_weight
                )

            total_moves += moves
            wins[winner] += 1
            games_completed += 1
            detail_results.append({
                "win_type": win_type,
                "winner": winner,
                "moves": moves,
                "territory_diff": territory_diff
            })
            if record_interval and (games_completed % record_interval == 0):
                if record_limit is None or len(records) < record_limit:
                    records.append(
                        _build_selfplay_record(
                            actions,
                            winner,
                            win_type,
                            moves,
                            territory_diff,
                            self.board_size,
                            self.komi,
                            self.center_wall
                        )
                    )

            if verbose and (games_completed % 100 == 0 or games_completed == num_games):
                print(f"  Completed {games_completed}/{num_games} games...")

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
                cache_hits = stats.get("cache_hits", 0)
                cache_misses = stats.get("cache_misses", 0)
                cache_total = cache_hits + cache_misses
                cache_hit_rate = cache_hits / max(1, cache_total)
                cache_size = stats.get("cache_size", 0)
                deduped = stats.get("deduped", 0)
                unique_evals = stats.get("unique_evals", 0)
                forward_calls = stats.get("forward_calls", unique_evals)
                avg_eval_ms = 1000 * stats["total_infer_time"] / max(1, unique_evals)
                print(
                    f"  Inference stats: batches={stats['total_batches']}, avg_batch={avg_batch:.1f}, "
                    f"avg_batch_infer_ms={avg_infer_ms:.2f}, avg_wait_ms={avg_wait_ms:.2f}, wait_ratio={wait_ratio:.2f}"
                )
                print(
                    f"  Cache stats: hit_rate={cache_hit_rate:.2%}, hits={cache_hits}, misses={cache_misses}, "
                    f"cache_size={cache_size}, deduped={deduped}, avg_eval_ms={avg_eval_ms:.2f}"
                )
                print(
                    f"  Cache bypass: forward_calls={forward_calls}, "
                    f"cache_hits_bypassed={cache_hits}, unique_evals={unique_evals}"
                )
                debug_samples = stats.get("debug_samples", 0)
                if debug_samples:
                    debug_policy_l1 = stats.get("debug_policy_l1")
                    debug_value_max = stats.get("debug_value_max")
                    print(
                        f"  Cache debug: samples={debug_samples}, "
                        f"policy_l1={debug_policy_l1:.6f}, value_max_diff={debug_value_max:.6f}"
                    )

        detail_summary = _summarize_selfplay_results(detail_results)
        return {
            "games": num_games,
            "avg_moves": total_moves / max(1, games_completed),
            "black_wins": wins[1],
            "white_wins": wins[2],
            "draws": wins[0],
            "buffer_size": len(self.replay_buffer),
            "detail": detail_summary,
            "records": records
        }
    
    def train_step_batch(self):
        """한 배치 학습"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 샘플링
        states, policies, values, ownerships, ownership_weights, win_targets, win_weights, win_type_targets, win_type_weights = self.replay_buffer.sample(self.batch_size)
        
        # Tensor 변환
        states_t = torch.FloatTensor(states).to(self.device)
        policies_t = torch.FloatTensor(policies).to(self.device)
        values_t = torch.FloatTensor(values).unsqueeze(1).to(self.device)
        ownerships_t = torch.FloatTensor(ownerships).to(self.device)
        ownership_weights_t = torch.FloatTensor(ownership_weights).to(self.device)
        win_targets_t = torch.FloatTensor(win_targets).unsqueeze(1).to(self.device)
        win_weights_t = torch.FloatTensor(win_weights).to(self.device)
        win_type_targets_t = torch.FloatTensor(win_type_targets).unsqueeze(1).to(self.device)
        win_type_weights_t = torch.FloatTensor(win_type_weights).to(self.device)
        
        # Forward
        self.network.train()
        if self.freeze_backbone:
            self._set_backbone_eval()
        pred_policies, pred_values, pred_ownership, pred_win, pred_win_type = self.network(states_t)
        
        # Loss 계산
        # Policy loss: Cross entropy (target은 확률 분포)
        policy_loss = -torch.sum(policies_t * pred_policies) / self.batch_size
        
        # Value loss: MSE
        value_loss = torch.mean((pred_values - values_t) ** 2)
        
        # Ownership loss (auxiliary)
        ownership_loss_map = F.binary_cross_entropy_with_logits(
            pred_ownership, ownerships_t, reduction="none"
        )
        ownership_loss_per_sample = ownership_loss_map.mean(dim=(1, 2, 3))
        ownership_loss = (ownership_loss_per_sample * ownership_weights_t).mean()

        # Win prediction loss (Black win probability)
        win_loss_map = F.binary_cross_entropy_with_logits(
            pred_win, win_targets_t, reduction="none"
        ).squeeze(1)
        win_loss = (win_loss_map * win_weights_t).mean() * self.win_loss_weight

        # Win type loss (capture vs territory)
        win_type_loss_map = F.binary_cross_entropy_with_logits(
            pred_win_type, win_type_targets_t, reduction="none"
        ).squeeze(1)
        win_type_loss = (win_type_loss_map * win_type_weights_t).mean() * self.win_type_loss_weight

        # Total loss
        total_loss = policy_loss + value_loss + ownership_loss + win_loss + win_type_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self.train_step += 1
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "ownership_loss": ownership_loss.item(),
            "win_loss": win_loss.item(),
            "win_type_loss": win_type_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def train_epoch(self, num_batches=100):
        """여러 배치 학습"""
        losses = {"policy": [], "value": [], "ownership": [], "win": [], "win_type": [], "total": []}
        
        for _ in range(num_batches):
            result = self.train_step_batch()
            if result:
                losses["policy"].append(result["policy_loss"])
                losses["value"].append(result["value_loss"])
                losses["ownership"].append(result["ownership_loss"])
                losses["win"].append(result["win_loss"])
                losses["win_type"].append(result["win_type_loss"])
                losses["total"].append(result["total_loss"])
        
        if losses["total"]:
            return {
                "policy_loss": np.mean(losses["policy"]),
                "value_loss": np.mean(losses["value"]),
                "ownership_loss": np.mean(losses["ownership"]),
                "win_loss": np.mean(losses["win"]),
                "win_type_loss": np.mean(losses["win_type"]),
                "total_loss": np.mean(losses["total"]),
                "num_batches": len(losses["total"])
            }
        return None
    
    def evaluate_vs_random(self, num_games=20):
        """랜덤 플레이어 대비 평가"""
        self.network.eval()
        wins = 0
        
        for game_idx in range(num_games):
            self.env.reset()
            
            # 절반은 AI가 흑, 절반은 AI가 백
            ai_player = 1 if game_idx < num_games // 2 else 2
            
            done = False
            moves = 0
            max_moves = max(100, self.board_size * self.board_size + 20)
            
            while not done and moves < max_moves:
                current_player = self.env.current_player
                state = (
                    self.env.board.copy(),
                    self.env.current_player,
                    self.env.consecutive_passes
                )
                
                if current_player == ai_player:
                    # AI (MCTS)
                    action, _ = self.mcts.run(state, temperature=0)
                else:
                    # Random
                    legal_moves = self.env.get_legal_moves()
                    legal_indices = np.where(legal_moves)[0]
                    # 패스 제외 (가능하면)
                    non_pass = legal_indices[legal_indices != self.env.pass_action]
                    if len(non_pass) > 0 and np.random.random() > 0.1:
                        action = np.random.choice(non_pass)
                    else:
                        action = np.random.choice(legal_indices)
                
                _, reward, done, info = self.env.step(action)
                moves += 1
            
            # 승패 판정
            winner = winner_id_from_step(info, reward, self.env.current_player)
            if winner == ai_player:
                wins += 1
        
        return wins / num_games

    def evaluate_vs_random_multiprocess(self, num_games=20, num_workers=None):
        """멀티프로세싱으로 랜덤 플레이어 대비 평가."""
        self.network.eval()
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 2)

        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        ai_players = [1 if i < num_games // 2 else 2 for i in range(num_games)]
        wins = 0

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_eval_worker,
            initargs=(
                network_state_dict,
                self.board_size,
                self.num_res_blocks,
                self.num_channels,
                self.num_simulations,
                self.c_puct,
                self.center_wall,
                self.komi,
                self.use_fast_env,
                self.use_liberty_features,
                self.liberty_bins,
                self.use_last_moves,
                self.mcts_eval_batch_size,
                self.mcts_profile,
                self.mcts_profile_every
            )
        ) as executor:
            futures = [executor.submit(_eval_one_game, p) for p in ai_players]
            for future in as_completed(futures):
                winner, ai_player = future.result()
                if winner == "Black" and ai_player == 1:
                    wins += 1
                elif winner == "White" and ai_player == 2:
                    wins += 1

        return wins / num_games

    def evaluate_vs_random_async(self, num_games=20, num_workers=None):
        """Async inference 서버로 랜덤 대비 평가."""
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 2)

        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        server = BatchInferenceServer(
            network_state_dict=network_state_dict,
            board_size=self.board_size,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels,
            batch_size=self.eval_infer_batch_size,
            timeout=self.eval_infer_timeout,
            use_liberty_features=self.use_liberty_features,
            liberty_bins=self.liberty_bins,
            use_last_moves=self.use_last_moves,
            network_head=self.network_head
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
                target=_eval_worker_loop_random,
                args=(
                    game_queue,
                    result_queue,
                    worker_id,
                    server.input_queue,
                    output_queues[worker_id],
                    self.board_size,
                    self.num_res_blocks,
                    self.num_channels,
                    self.num_simulations,
                    self.c_puct,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                    self.center_wall,
                    self.komi,
                    self.use_fast_env,
                    self.use_liberty_features,
                    self.liberty_bins,
                    self.use_last_moves,
                    self.mcts_eval_batch_size,
                    self.mcts_profile,
                    self.mcts_profile_every
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
            winner, ai_player, moves, states, info, final_player = result_queue.get()
            if winner == ai_player:
                wins += 1
            game_records.append({
                "winner": winner,
                "ai_player": ai_player,
                "moves": moves,
                "states": states,
                "info": info,
                "final_player": final_player
            })

        for _ in workers:
            game_queue.put(None)
        for p in workers:
            p.join()

        server.input_queue.put(None)
        server_process.join()

        return wins / num_games, game_records

    def evaluate_vs_mcts_async(self, num_games=20, mcts_sims=100, num_workers=None):
        """Async inference 서버로 순수 MCTS 대비 평가."""
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 2)

        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        server = BatchInferenceServer(
            network_state_dict=network_state_dict,
            board_size=self.board_size,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels,
            batch_size=self.eval_infer_batch_size,
            timeout=self.eval_infer_timeout,
            use_liberty_features=self.use_liberty_features,
            liberty_bins=self.liberty_bins,
            use_last_moves=self.use_last_moves,
            network_head=self.network_head
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
                    self.board_size,
                    self.num_res_blocks,
                    self.num_channels,
                    self.num_simulations,
                    self.c_puct,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                    self.center_wall,
                    self.komi,
                    mcts_sims,
                    self.use_fast_env,
                    self.use_liberty_features,
                    self.liberty_bins,
                    self.use_last_moves,
                    self.mcts_eval_batch_size,
                    self.mcts_profile,
                    self.mcts_profile_every
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
            winner, ai_player, moves, states, info, final_player = result_queue.get()
            if winner == ai_player:
                wins += 1
            game_records.append({
                "winner": winner,
                "ai_player": ai_player,
                "moves": moves,
                "states": states,
                "info": info,
                "final_player": final_player
            })

        for _ in workers:
            game_queue.put(None)
        for p in workers:
            p.join()

        server.input_queue.put(None)
        server_process.join()

        return wins / num_games, game_records

    def evaluate_vs_checkpoint(self, num_games=20, checkpoint_path="checkpoints/alphazero_best.pt"):
        """현재 모델 vs 체크포인트 모델 (단일 프로세스)."""
        if not os.path.exists(checkpoint_path):
            return None

        opponent_env = GreatKingdomEnv(board_size=self.board_size, center_wall=self.center_wall)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        opponent_head = infer_head_type_from_state_dict(checkpoint["network"])
        opponent_network = AlphaZeroNetwork(
            board_size=self.board_size,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels,
            use_liberty_features=self.use_liberty_features,
            liberty_bins=self.liberty_bins,
            use_last_moves=self.use_last_moves,
            head_type=opponent_head
        ).to(self.device)
        load_state_dict_safe(opponent_network, checkpoint["network"])
        opponent_network.eval()

        opponent_mcts = AlphaZeroMCTS(
            opponent_network,
            opponent_env,
            c_puct=self.c_puct,
            num_simulations=self.num_simulations,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            use_liberty_features=self.use_liberty_features,
            liberty_bins=self.liberty_bins,
            use_last_moves=self.use_last_moves
        )

        wins = 0
        for game_idx in range(num_games):
            self.env.reset()
            done = False
            moves = 0
            ai_player = 1 if game_idx < num_games // 2 else 2
            max_moves = max(100, self.board_size * self.board_size + 20)

            while not done and moves < max_moves:
                state = (
                    self.env.board.copy(),
                    self.env.current_player,
                    self.env.consecutive_passes
                )

                if self.env.current_player == ai_player:
                    action, _ = self.mcts.run(state, temperature=0)
                else:
                    action, _ = opponent_mcts.run(state, temperature=0)

                _, reward, done, info = self.env.step(action)
                moves += 1

            winner = winner_id_from_step(info, reward, self.env.current_player)
            if winner == ai_player:
                wins += 1

        return wins / num_games

    def evaluate_vs_checkpoint_async(self, num_games=20, checkpoint_path="checkpoints/alphazero_best.pt", num_workers=None):
        """Async inference 서버로 현재 모델 vs 체크포인트 모델 평가."""
        if not os.path.exists(checkpoint_path):
            return None, []
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 2)

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        opponent_state_dict = {k: v.cpu() for k, v in checkpoint["network"].items()}

        network_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        server = BatchInferenceServer(
            network_state_dict=network_state_dict,
            board_size=self.board_size,
            num_res_blocks=self.num_res_blocks,
            num_channels=self.num_channels,
            batch_size=self.eval_infer_batch_size,
            timeout=self.eval_infer_timeout,
            use_liberty_features=self.use_liberty_features,
            liberty_bins=self.liberty_bins,
            use_last_moves=self.use_last_moves,
            network_head=self.network_head
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
                target=_eval_worker_loop_best,
                args=(
                    game_queue,
                    result_queue,
                    worker_id,
                    server.input_queue,
                    output_queues[worker_id],
                    self.board_size,
                    self.num_res_blocks,
                    self.num_channels,
                    self.num_simulations,
                    self.c_puct,
                    self.dirichlet_alpha,
                    self.dirichlet_epsilon,
                    opponent_state_dict,
                    self.center_wall,
                    self.komi,
                    self.use_fast_env,
                    self.use_liberty_features,
                    self.liberty_bins,
                    self.use_last_moves,
                    self.mcts_eval_batch_size,
                    self.mcts_profile,
                    self.mcts_profile_every
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
            winner, ai_player, moves, states, info, final_player = result_queue.get()
            if winner == ai_player:
                wins += 1
            game_records.append({
                "winner": winner,
                "ai_player": ai_player,
                "moves": moves,
                "states": states,
                "info": info,
                "final_player": final_player
            })

        for _ in workers:
            game_queue.put(None)
        for p in workers:
            p.join()

        server.input_queue.put(None)
        server_process.join()

        return wins / num_games, game_records
    
    def save(self, path="checkpoints/alphazero.pt", iteration=None, history=None):
        """모델 및 학습 히스토리 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_step": self.train_step,
            "total_games": self.total_games,
            "iteration": iteration,
            "center_wall": self.center_wall,
            "komi": self.komi,
            "network_head": self.network_head
        }
        # 학습 히스토리 포함
        if history is not None:
            save_dict["history"] = history
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path="checkpoints/alphazero.pt", load_optimizer=True, load_meta=True):
        """모델 로드"""
        if os.path.exists(path):
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location=self.device)
            state_dict = checkpoint.get("network", {})
            try:
                self.network.load_state_dict(state_dict)
                if load_optimizer and "optimizer" in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint["optimizer"])
                    except ValueError as e:
                        print(f"Warning: optimizer state mismatch, skipping optimizer load: {e}")
                if load_meta:
                    self.train_step = checkpoint.get("train_step", 0)
                    self.total_games = checkpoint.get("total_games", 0)
                print(f"Model loaded from {path}")
            except RuntimeError:
                model_state = self.network.state_dict()
                filtered = {
                    k: v for k, v in state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape
                }
                missing = len(model_state) - len(filtered)
                unexpected = len(state_dict) - len(filtered)
                model_state.update(filtered)
                self.network.load_state_dict(model_state)
                if load_meta:
                    self.train_step = checkpoint.get("train_step", 0)
                    self.total_games = checkpoint.get("total_games", 0)
                print(
                    f"Model partially loaded from {path} (matched={len(filtered)}, missing={missing}, unexpected={unexpected})"
                )
            return checkpoint
        return None


def train_alphazero(
    num_iterations=10,
    games_per_iteration=20,
    batches_per_iteration=50,
    eval_games=10,
    num_simulations=50,
    save_interval=5,
    temperature_threshold=20,
    c_puct=2.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    fresh_start=False,
    board_size=5,
    num_res_blocks=3,
    num_channels=64,
    batch_size=32,
    buffer_size=80000,
    checkpoint_dir=None,
    center_wall=True,
    komi=0,
    use_fast_env=False,
    playout_full_prob=0.25,
    playout_full_cap=600,
    playout_fast_cap=100,
    infer_batch_size=256,
    infer_timeout=0.05,
    eval_infer_batch_size=None,
    eval_infer_timeout=None,
    mcts_eval_batch_size=1,
    mcts_profile=False,
    mcts_profile_every=0,
    temperature_schedule=None,
    use_forced_playouts=False,
    use_policy_target_pruning=False,
    forced_playout_k=2.0,
    use_liberty_features=True,
    liberty_bins=2,
    use_last_moves=False,
    network_head="fc",
    freeze_backbone=False,
    freeze_backbone_blocks=None,
    freeze_backbone_input=True,
    init_checkpoint=None,
    benchmark_interval=10,
    freeze_schedule=None,
    selfplay_record_interval=0,
    selfplay_record_dir="logs/selfplay_records",
    ownership_loss_weight=0.2,
    ownership_loss_weight_capture=0.1,
    win_loss_weight=0.1,
    win_type_loss_weight=0.1,
    cache_debug_samples=0,
    cache_max_entries=50000,
    train_buffer_min_factor=2.0
):
    """
    AlphaZero 학습 메인 루프
    
    Args:
        num_iterations: 전체 반복 횟수
        games_per_iteration: 반복당 자가 대전 게임 수
        batches_per_iteration: 반복당 학습 배치 수
        eval_games: 평가 게임 수
        num_simulations: MCTS 시뮬레이션 횟수
        save_interval: 저장 간격
        temperature_threshold: 기존 API 호환용 (현재는 보드 크기 기반 스케줄 사용)
        c_puct: PUCT 탐색 계수
        dirichlet_alpha: 루트 Dirichlet alpha
        dirichlet_epsilon: 루트 Dirichlet 혼합 비율
        fresh_start: True면 체크포인트를 무시하고 새로 시작
        board_size: 보드 크기
        num_res_blocks: ResBlock 개수
        num_channels: 채널 수
        batch_size: 학습 배치 크기
        buffer_size: 리플레이 버퍼 크기
        checkpoint_dir: 체크포인트 저장 디렉터리
    """
    print("=" * 60)
    print("AlphaZero Training")
    print("=" * 60)
    
    # TensorBoard 설정
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"runs/alphazero_{run_id}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log dir: {log_dir}")
    print(f"Run 'tensorboard --logdir=runs' to view logs")
    
    # 텍스트 로그 파일 설정
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/train_{run_id}.log"
    print(f"Training log file: {log_file}")
    
    # 학습 히스토리 초기화
    history = {
        'iterations': [],
        'policy_loss': [],
        'value_loss': [],
        'ownership_loss': [],
        'win_loss': [],
        'win_type_loss': [],
        'total_loss': [],
        'win_rate': [],
        'best_match_rate': [],
        'avg_moves': [],
        'buffer_size': [],
        'black_wins': [],
        'white_wins': [],
        'draws': []
    }
    
    # Checkpoint dir
    if checkpoint_dir is None:
        suffix = "on" if center_wall else "off"
        checkpoint_dir = os.path.join("checkpoints", f"board_{board_size}", f"center_wall_{suffix}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Trainer 생성
    trainer = AlphaZeroTrainer(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        batch_size=batch_size,
        buffer_size=buffer_size,
        lr=0.001,
        center_wall=center_wall,
        komi=komi,
        use_fast_env=use_fast_env,
        playout_full_prob=playout_full_prob,
        playout_full_cap=playout_full_cap,
        playout_fast_cap=playout_fast_cap,
        infer_batch_size=infer_batch_size,
        infer_timeout=infer_timeout,
        eval_infer_batch_size=eval_infer_batch_size,
        eval_infer_timeout=eval_infer_timeout,
        mcts_eval_batch_size=mcts_eval_batch_size,
        mcts_profile=mcts_profile,
        mcts_profile_every=mcts_profile_every,
        temperature_schedule=temperature_schedule,
        use_forced_playouts=use_forced_playouts,
        use_policy_target_pruning=use_policy_target_pruning,
        forced_playout_k=forced_playout_k,
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves,
        network_head=network_head,
        freeze_backbone=freeze_backbone,
        freeze_backbone_blocks=freeze_backbone_blocks,
        freeze_backbone_input=freeze_backbone_input,
        ownership_loss_weight=ownership_loss_weight,
        ownership_loss_weight_capture=ownership_loss_weight_capture,
        win_loss_weight=win_loss_weight,
        win_type_loss_weight=win_type_loss_weight,
        cache_debug_samples=cache_debug_samples,
        cache_max_entries=cache_max_entries,
        train_buffer_min_factor=train_buffer_min_factor
    )
    
    # 기존 체크포인트 로드 시도 (latest -> final -> best)
    checkpoint = None
    initialized_from_checkpoint = False
    if not fresh_start:
        checkpoint = trainer.load(os.path.join(checkpoint_dir, "alphazero_latest.pt"))
        if checkpoint is None:
            checkpoint = trainer.load(os.path.join(checkpoint_dir, "alphazero_final.pt"))
        if checkpoint is None:
            checkpoint = trainer.load(os.path.join(checkpoint_dir, "alphazero_best.pt"))
        if checkpoint is None and init_checkpoint:
            if os.path.exists(init_checkpoint):
                trainer.load(init_checkpoint, load_optimizer=False, load_meta=False)
                initialized_from_checkpoint = True
                print(f"Initialized from checkpoint: {init_checkpoint}")
            else:
                print(f"Init checkpoint not found: {init_checkpoint}")

    start_iteration = 0
    if checkpoint and checkpoint.get("iteration") is not None and not initialized_from_checkpoint:
        start_iteration = int(checkpoint["iteration"])
        # 기존 히스토리 복원
        if checkpoint.get("history") is not None:
            history = checkpoint["history"]
            print(f"Loaded training history ({len(history['iterations'])} iterations)")
            history.setdefault('best_match_rate', [])
            history.setdefault('ownership_loss', [])
            history.setdefault('win_loss', [])
            history.setdefault('win_type_loss', [])
    
    best_win_rate = 0.0
    if history['win_rate'] and not fresh_start:
        best_win_rate = max(history['win_rate'])
    
    eval_debug_rendered = False

    def _resolve_freeze_schedule(schedule, step):
        if not schedule:
            return None
        try:
            keys = sorted(int(k) for k in schedule.keys())
        except Exception:
            return None
        chosen = None
        for k in keys:
            if step >= k:
                chosen = k
        if chosen is None:
            return None
        value = schedule.get(chosen, schedule.get(str(chosen)))
        if value is None:
            return None
        if isinstance(value, str):
            if value.lower() in ("none", "unfreeze"):
                return None
        try:
            value = int(value)
        except Exception:
            return None
        if value < 0:
            return None
        return value

    for iteration in range(start_iteration + 1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{num_iterations}")
        print("=" * 60)

        if freeze_schedule:
            target_blocks = _resolve_freeze_schedule(freeze_schedule, iteration)
            if target_blocks is not None and target_blocks != trainer.freeze_backbone_blocks:
                trainer.set_freeze_backbone_blocks(target_blocks, freeze_input=freeze_backbone_input)
                print(f"    Freeze schedule applied: blocks={target_blocks}, input={freeze_backbone_input}")
        
        # === 1. Self-Play ===
        print(f"\n[1] Self-Play ({games_per_iteration} games)...")
        start_time = time.time()
        sp_stats = trainer.collect_self_play_data(
            num_games=games_per_iteration,
            temperature_threshold=temperature_threshold,
            temperature_schedule=temperature_schedule,
            use_async_inference=True,
            record_interval=selfplay_record_interval
        )
        sp_time = time.time() - start_time
        
        print(f"    Games: {sp_stats['games']} | "
              f"Avg Moves: {sp_stats['avg_moves']:.1f} | "
              f"B/W/D: {sp_stats['black_wins']}/{sp_stats['white_wins']}/{sp_stats['draws']}")
        print(f"    Buffer Size: {sp_stats['buffer_size']} | Time: {sp_time:.1f}s")
        detail = sp_stats.get("detail")
        if detail:
            for line in _format_selfplay_stats(detail):
                print(f"    {line}")
        if selfplay_record_interval and sp_stats.get("records"):
            record_path = os.path.join(
                selfplay_record_dir,
                f"selfplay_{run_id}_iter_{iteration:04d}.json"
            )
            _save_selfplay_records(
                sp_stats["records"],
                record_path,
                meta={
                    "iteration": iteration,
                    "games": sp_stats["games"],
                    "record_interval": selfplay_record_interval,
                    "board_size": board_size,
                    "komi": komi,
                    "center_wall": center_wall,
                    "run_id": run_id
                }
            )
            print(f"    Saved self-play records: {record_path} ({len(sp_stats['records'])})")
        
        # TensorBoard: Self-Play 통계
        writer.add_scalar('SelfPlay/avg_moves', sp_stats['avg_moves'], iteration)
        writer.add_scalar('SelfPlay/black_wins', sp_stats['black_wins'], iteration)
        writer.add_scalar('SelfPlay/white_wins', sp_stats['white_wins'], iteration)
        writer.add_scalar('SelfPlay/draws', sp_stats['draws'], iteration)
        writer.add_scalar('SelfPlay/buffer_size', sp_stats['buffer_size'], iteration)
        writer.add_scalar('SelfPlay/time_seconds', sp_time, iteration)
        
        # === 2. Training ===
        print(f"\n[2] Training ({batches_per_iteration} batches)...")
        min_factor = trainer.train_buffer_min_factor
        min_required = int(max(0.0, min_factor) * trainer.batch_size * batches_per_iteration)
        if min_factor > 0 and len(trainer.replay_buffer) < min_required:
            print(
                f"    Skipped: buffer size {len(trainer.replay_buffer)} < "
                f"min required {min_required} (factor={min_factor:.2f})"
            )
            train_stats = None
            train_time = 0.0
        else:
            start_time = time.time()
            train_stats = trainer.train_epoch(num_batches=batches_per_iteration)
            train_time = time.time() - start_time
        
        if train_stats:
            print(f"    Policy Loss: {train_stats['policy_loss']:.4f} | "
                  f"Value Loss: {train_stats['value_loss']:.4f} | "
                  f"Ownership Loss: {train_stats['ownership_loss']:.4f} | "
                  f"Win Loss: {train_stats['win_loss']:.4f} | "
                  f"WinType Loss: {train_stats['win_type_loss']:.4f} | "
                  f"Total: {train_stats['total_loss']:.4f}")
            print(f"    Batches: {train_stats['num_batches']} | Time: {train_time:.1f}s")
            
            # TensorBoard: Training 통계
            writer.add_scalar('Loss/policy', train_stats['policy_loss'], iteration)
            writer.add_scalar('Loss/value', train_stats['value_loss'], iteration)
            writer.add_scalar('Loss/ownership', train_stats['ownership_loss'], iteration)
            writer.add_scalar('Loss/win', train_stats['win_loss'], iteration)
            writer.add_scalar('Loss/win_type', train_stats['win_type_loss'], iteration)
            writer.add_scalar('Loss/total', train_stats['total_loss'], iteration)
            writer.add_scalar('Training/time_seconds', train_time, iteration)
            
            # 히스토리 기록
            history['iterations'].append(iteration)
            history['policy_loss'].append(train_stats['policy_loss'])
            history['value_loss'].append(train_stats['value_loss'])
            history['ownership_loss'].append(train_stats['ownership_loss'])
            history['win_loss'].append(train_stats['win_loss'])
            history['win_type_loss'].append(train_stats['win_type_loss'])
            history['total_loss'].append(train_stats['total_loss'])
            history['avg_moves'].append(sp_stats['avg_moves'])
            history['buffer_size'].append(sp_stats['buffer_size'])
            history['black_wins'].append(sp_stats['black_wins'])
            history['white_wins'].append(sp_stats['white_wins'])
            history['draws'].append(sp_stats['draws'])

            # Network updated -> bump version to invalidate eval cache
            trainer.net_version += 1
            if hasattr(trainer.mcts, "set_net_version"):
                trainer.mcts.set_net_version(trainer.net_version)
        
        # === 3. Evaluation (save_interval마다 실행) ===
        if benchmark_interval and iteration % benchmark_interval == 0:
            print("\n[Benchmark] Async self-play bottleneck check...")
            bench = _benchmark_async_selfplay(trainer, num_games=10)
            writer.add_scalar('Benchmark/elapsed_sec', bench["elapsed_sec"], iteration)
            writer.add_scalar('Benchmark/avg_moves', bench["avg_moves"], iteration)
            writer.add_scalar('Benchmark/infer_reqs', bench["infer_reqs"], iteration)
            writer.add_scalar('Benchmark/infer_time_sec', bench["infer_time_sec"], iteration)
            writer.add_scalar('Benchmark/wait_time_sec', bench["wait_time_sec"], iteration)
            writer.add_scalar('Benchmark/avg_batch_size', bench["avg_batch_size"], iteration)
            writer.add_scalar('Benchmark/avg_wait_ms', bench["avg_wait_ms"], iteration)
            writer.add_scalar('Benchmark/wait_ratio', bench["wait_ratio"], iteration)
            if bench.get("cache_hit_rate") is not None:
                writer.add_scalar('Benchmark/cache_hit_rate', bench["cache_hit_rate"], iteration)
            if bench.get("cache_size") is not None:
                writer.add_scalar('Benchmark/cache_size', bench["cache_size"], iteration)
            if bench.get("deduped") is not None:
                writer.add_scalar('Benchmark/deduped', bench["deduped"], iteration)
            if bench.get("avg_eval_ms") is not None:
                writer.add_scalar('Benchmark/avg_eval_ms', bench["avg_eval_ms"], iteration)
            if bench.get("forward_calls") is not None:
                writer.add_scalar('Benchmark/forward_calls', bench["forward_calls"], iteration)
            if bench.get("cache_hits_bypassed") is not None:
                writer.add_scalar('Benchmark/cache_hits_bypassed', bench["cache_hits_bypassed"], iteration)
            if bench.get("debug_policy_l1") is not None:
                writer.add_scalar('Benchmark/cache_policy_l1', bench["debug_policy_l1"], iteration)
            if bench.get("debug_value_max") is not None:
                writer.add_scalar('Benchmark/cache_value_max', bench["debug_value_max"], iteration)

            os.makedirs("logs", exist_ok=True)
            bench_file = os.path.join("logs", "benchmark_async_selfplay.csv")
            write_header = not os.path.exists(bench_file)
            with open(bench_file, 'a', encoding='utf-8') as f:
                if write_header:
                    f.write(
                        "iteration,games,workers,elapsed_sec,avg_moves,infer_reqs,infer_time_sec,"
                        "wait_time_sec,avg_batch_size,avg_wait_ms,wait_ratio,batch_size\n"
                    )
                f.write(
                    f"{iteration},{bench['games']},{bench['workers']},{bench['elapsed_sec']:.4f},"
                    f"{bench['avg_moves']:.2f},{bench['infer_reqs']},{bench['infer_time_sec']:.4f},"
                    f"{bench['wait_time_sec']:.4f},{bench['avg_batch_size']:.2f},{bench['avg_wait_ms']:.2f},"
                    f"{bench['wait_ratio']:.4f},{bench['batch_size']}\n"
                )

        if iteration % save_interval == 0:
            print(f"\n[3] Evaluation vs MCTS ladder ({eval_games} games each)...")
            ladder_sims = (200, 500, 1000)
            ladder_passed = False
            ladder_stage = int(history.get("ladder_stage", 0))
            last_win_rate = None
            last_eval_time = 0.0
            eval_records = []
            best_records = []
            eval_stats = None
            best_stats = None
            if ladder_stage < len(ladder_sims):
                sims = ladder_sims[ladder_stage]
                start_time = time.time()
                win_rate, records = trainer.evaluate_vs_mcts_async(
                    num_games=eval_games,
                    mcts_sims=sims
                )
                eval_time = time.time() - start_time
                last_win_rate = win_rate
                last_eval_time = eval_time
                print(f"    MCTS {sims}: Win Rate {win_rate*100:.1f}% | Time: {eval_time:.1f}s")
                writer.add_scalar(f'Evaluation/mcts_{sims}_win_rate', win_rate, iteration)
                writer.add_scalar(f'Evaluation/mcts_{sims}_time_seconds', eval_time, iteration)
                eval_records = records
                if eval_records:
                    eval_stats = _summarize_eval_records(eval_records)
                    print(f"    Detailed results vs MCTS {sims}:")
                    for line in _format_eval_stats(eval_stats):
                        print(f"      {line}")
                    print("    Rendering first evaluation game (vs MCTS)...")
                    _render_recorded_game(eval_records[0], board_size, center_wall)
                if win_rate >= 0.9:
                    ladder_stage += 1
                    history["ladder_stage"] = ladder_stage
                    ladder_passed = ladder_stage >= len(ladder_sims)
                else:
                    ladder_passed = False
            else:
                ladder_passed = True

            # 히스토리에 win_rate 기록 (마지막 평가 기준)
            if last_win_rate is not None:
                history['win_rate'].append(last_win_rate)
                writer.add_scalar('Evaluation/win_rate', last_win_rate, iteration)
                writer.add_scalar('Evaluation/time_seconds', last_eval_time, iteration)

            # Best 모델 갱신 (ladder 통과 시에만)
            if ladder_passed:
                best_match_rate, best_records = trainer.evaluate_vs_checkpoint_async(
                    num_games=20,
                    checkpoint_path=os.path.join(checkpoint_dir, "alphazero_best.pt")
                )
                if best_match_rate is not None:
                    print(f"    vs Best Win Rate: {best_match_rate*100:.1f}%")
                    if best_records:
                        best_stats = _summarize_eval_records(best_records)
                        print("    Detailed results vs Best:")
                        for line in _format_eval_stats(best_stats):
                            print(f"      {line}")
                        print("    Rendering first evaluation game (vs Best)...")
                        _render_recorded_game(best_records[0], board_size, center_wall)
                    history['best_match_rate'].append(best_match_rate)
                    writer.add_scalar('Evaluation/best_match_rate', best_match_rate, iteration)
                    if best_match_rate >= 0.6:
                        trainer.save(os.path.join(checkpoint_dir, "alphazero_best.pt"), iteration=iteration, history=history)
                        print(f"    ★ New Best! ({best_match_rate*100:.1f}%)")
                else:
                    # 첫 베스트 모델 생성
                    trainer.save(os.path.join(checkpoint_dir, "alphazero_best.pt"), iteration=iteration, history=history)
                    history['best_match_rate'].append(1.0)
                    writer.add_scalar('Evaluation/best_match_rate', 1.0, iteration)
                    print("    ★ New Best! (initial)")

            # 텍스트 로그 파일에 기록
            with open(log_file, 'a', encoding='utf-8') as f:
                win_rate_pct = last_win_rate * 100 if last_win_rate is not None else 0.0
                f.write(f"[Iter {iteration}] "
                        f"Loss: {train_stats['total_loss']:.4f} "
                        f"(P:{train_stats['policy_loss']:.4f}, V:{train_stats['value_loss']:.4f}) | "
                        f"WinRate: {win_rate_pct:.1f}% | "
                        f"AvgMoves: {sp_stats['avg_moves']:.1f} | "
                        f"Buffer: {sp_stats['buffer_size']}\n")
                if eval_stats is not None:
                    for line in _format_eval_stats(eval_stats):
                        f.write(f"[Iter {iteration}] Eval MCTS detail: {line}\n")
                if best_stats is not None:
                    for line in _format_eval_stats(best_stats):
                        f.write(f"[Iter {iteration}] Eval Best detail: {line}\n")

            # Debug 저장 폴더
            debug_dir = os.path.join("debug_eval", f"board_{board_size}", f"iter_{iteration}")
            os.makedirs(debug_dir, exist_ok=True)

            # Debug 저장 (랜덤/베스트 20게임)
            try:
                import pickle
                with open(os.path.join(debug_dir, "mcts_eval.pkl"), "wb") as f:
                    pickle.dump(eval_records, f)
                if best_records:
                    with open(os.path.join(debug_dir, "best_eval.pkl"), "wb") as f:
                        pickle.dump(best_records, f)
            except Exception as e:
                print(f"    Debug save failed: {e}")

            # Debug render removed for simpler logs
            
            # 주기적 저장
            trainer.save(os.path.join(checkpoint_dir, f"alphazero_iter{iteration}.pt"), iteration=iteration, history=history)
            trainer.save(os.path.join(checkpoint_dir, "alphazero_latest.pt"), iteration=iteration, history=history)
    
    # 최종 저장
    trainer.save(os.path.join(checkpoint_dir, "alphazero_final.pt"), iteration=num_iterations, history=history)
    trainer.save(os.path.join(checkpoint_dir, "alphazero_latest.pt"), iteration=num_iterations, history=history)
    
    # TensorBoard writer 닫기
    writer.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total Games: {trainer.total_games}")
    print(f"Best Win Rate: {best_win_rate*100:.1f}%")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Training log file: {log_file}")
    print("=" * 60)
    
    return trainer


