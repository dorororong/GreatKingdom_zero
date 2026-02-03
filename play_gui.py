import pygame
import numpy as np
import sys
import os
import torch
from env.env import GreatKingdomEnv
from env.env_fast import GreatKingdomEnvFast
from mcts import MCTS
from mcts_fast import MCTS as MCTSFast
from network import AlphaZeroNetwork, infer_head_type_from_state_dict, load_state_dict_safe, encode_board_from_state, get_input_channels

try:
    from train.config import list_profiles, get_profile
    _CONFIG_AVAILABLE = True
except Exception:
    _CONFIG_AVAILABLE = False
from mcts_alphazero import AlphaZeroMCTS

# 색상 정의
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
WOOD = (222, 184, 135)
WALL_COLOR = (139, 69, 19)
TERRITORY_BLACK = (0, 0, 0, 100)  # 투명도 포함
TERRITORY_WHITE = (255, 255, 255, 100)

class GreatKingdomGUI:
    def __init__(self, board_size=5, cell_size=80, ai_type='mcts', mcts_simulations=1000,
                 alphazero_simulations=100, checkpoint_path="checkpoints/alphazero_latest.pt",
                 center_wall=True, komi=0, alphazero_black=True, use_fast_env=False,
                 spectate_mode="az_vs_mcts", checkpoint_path_b=None,
                 az_use_last_moves_override=None, az_use_liberty_features_override=None,
                 az_liberty_bins_override=None, az_use_last_moves_override_b=None,
                 az_use_liberty_features_override_b=None, az_liberty_bins_override_b=None,
                 debug_model=False):
        pygame.init()
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = int(self.cell_size * 0.5)
        self.heatmap_size = int(self.cell_size * self.board_size * 0.5)
        self.info_panel_height = int(self.cell_size * 1.0)
        
        board_pixel = self.cell_size * self.board_size
        self.board_origin_x = self.margin + self.heatmap_size + self.margin
        self.board_origin_y = self.margin
        self.width = self.board_origin_x + board_pixel + self.margin
        self.height = board_pixel + 2 * self.margin + self.info_panel_height
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Great Kingdom")
        
        self.font = pygame.font.SysFont("malgungothic", 24)
        self.large_font = pygame.font.SysFont("malgungothic", 48)
        self.tiny_font = pygame.font.SysFont("malgungothic", max(10, self.cell_size // 4))
        
        self.use_fast_env = use_fast_env
        env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
        self.env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)
        self.obs = self.env.reset()
        self.done = False
        self.info = {}
        self.message = ""
        
        # AI 설정
        
        self.show_ownership_overlay = True  # 영토 예측 오버레이 표시 여부
        self.ai_type = ai_type  # 'mcts', 'alphazero', 'human' (2인 대전), 'spectate' (AI vs AI 관전)
        self.mcts = None
        self.alphazero_mcts = None
        self.alphazero_network = None
        self.alphazero_mcts_b = None
        self.alphazero_network_b = None
        self.mcts_simulations = mcts_simulations
        self.alphazero_simulations = alphazero_simulations
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path_b = checkpoint_path_b
        self.ai_thinking = False  # AI가 생각 중인지 표시
        self.last_human_action = None  # 사람의 마지막 수 기록 (AI에게 전달)
        self.last_ai_action = None  # AI의 마지막 수 기록 (관전 모드용)
        self.last_moves = (None, None)
        self.az_use_last_moves = True
        self.az_use_liberty_features = True
        self.az_liberty_bins = 2
        self.az_label = "AlphaZero"
        self.azb_use_last_moves = True
        self.azb_use_liberty_features = True
        self.azb_liberty_bins = 2
        self.az_label_b = "AlphaZeroB"
        self.az_use_last_moves_override = az_use_last_moves_override
        self.az_use_liberty_features_override = az_use_liberty_features_override
        self.az_liberty_bins_override = az_liberty_bins_override
        self.az_use_last_moves_override_b = az_use_last_moves_override_b
        self.az_use_liberty_features_override_b = az_use_liberty_features_override_b
        self.az_liberty_bins_override_b = az_liberty_bins_override_b
        self.debug_model = bool(debug_model)
        self.human_player = None  # 1=흑, 2=백 (None이면 색상 선택 화면)
        self.game_started = False  # game started flag
        self._aux_cache_key = None
        self._aux_cache = None
        
        # 관전 모드 설정
        self.alphazero_black = alphazero_black  # True: AlphaZero=흑, MCTS=백 / False: 반대
        self.spectate_mode = spectate_mode
        self.spectate_delay = 500  # 관전 모드에서 수 사이의 딜레이 (ms)
        
        # AI 초기화
        self._init_ai()
        
        # 버튼 영역 정의
        self.pass_button_rect = pygame.Rect(
            self.width - 150, self.height - 80, 120, 50
        )
        self.reset_button_rect = pygame.Rect(
            self.width - 280, self.height - 80, 120, 50
        )
        self.overlay_button_rect = pygame.Rect(
            self.width - 430, self.height - 80, 130, 50
        )
        
        # 색상 선택 버튼 (시작 화면용)
        button_width = 150
        button_height = 80
        button_gap = 50
        center_x = self.width // 2
        center_y = self.height // 2
        
        self.black_button_rect = pygame.Rect(
            center_x - button_width - button_gap // 2,
            center_y - button_height // 2,
            button_width,
            button_height
        )
        self.white_button_rect = pygame.Rect(
            center_x + button_gap // 2,
            center_y - button_height // 2,
            button_width,
            button_height
        )
    
    def _init_ai(self):
        """AI 초기화"""
        mcts_cls = MCTSFast if self.use_fast_env else MCTS
        if self.ai_type == 'mcts':
            self.mcts = mcts_cls(self.env, simulations_per_move=self.mcts_simulations)
            print(f"AI: Pure MCTS ({self.mcts_simulations} simulations)")
        elif self.ai_type == 'alphazero':
            self._load_alphazero()
            print(f"AI: AlphaZero ({self.alphazero_simulations} simulations)")
        elif self.ai_type == 'spectate':
            # Spectate: AlphaZero vs MCTS or AlphaZero vs AlphaZero
            self._load_alphazero()
            if self.spectate_mode == 'az_vs_az' and self.alphazero_mcts_b is not None:
                print(f"Spectate Mode: {self.az_label}({self.alphazero_simulations} sims) vs {self.az_label_b}({self.alphazero_simulations} sims)")
                if self.alphazero_black:
                    print(f"  {self.az_label} plays Black, {self.az_label_b} plays White")
                else:
                    print(f"  {self.az_label_b} plays Black, {self.az_label} plays White")
            else:
                self.spectate_mode = 'az_vs_mcts'
                self.mcts = mcts_cls(self.env, simulations_per_move=self.mcts_simulations)
                print(f"Spectate Mode: AlphaZero({self.alphazero_simulations} sims) vs MCTS({self.mcts_simulations} sims)")
                if self.alphazero_black:
                    print("  AlphaZero plays Black, MCTS plays White")
                else:
                    print("  MCTS plays Black, AlphaZero plays White")
            self.game_started = True  # start immediately
        else:  # human
            print("Mode: Human vs Human")
    
    def _load_alphazero_model(self, checkpoint_path, use_last_moves_override=None,
                              use_liberty_features_override=None, liberty_bins_override=None):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        num_res_blocks = checkpoint.get('num_res_blocks', 3)
        num_channels = checkpoint.get('num_channels', 64)
        use_liberty_features = checkpoint.get('use_liberty_features')
        liberty_bins = checkpoint.get('liberty_bins')
        use_last_moves = checkpoint.get('use_last_moves')
        head_type = infer_head_type_from_state_dict(checkpoint.get('network', {}))

        if 'num_res_blocks' not in checkpoint or 'num_channels' not in checkpoint:
            policy_weight = checkpoint['network'].get('policy_conv.weight')
            if policy_weight is not None:
                num_channels = policy_weight.shape[1]
            res_block_keys = [
                k for k in checkpoint['network'].keys()
                if k.startswith('res_blocks.') and k.endswith('.conv1.weight')
            ]
            if res_block_keys:
                num_res_blocks = len(res_block_keys)

        if use_liberty_features is None or liberty_bins is None or use_last_moves is None:
            conv_weight = checkpoint['network'].get('conv_input.weight')
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

        if use_liberty_features_override is not None:
            use_liberty_features = bool(use_liberty_features_override)
        if liberty_bins_override is not None:
            liberty_bins = int(liberty_bins_override)
        if use_last_moves_override is not None:
            use_last_moves = bool(use_last_moves_override)

        conv_weight = checkpoint['network'].get('conv_input.weight')
        if conv_weight is not None:
            expected_channels = get_input_channels(
                use_liberty_features=use_liberty_features,
                liberty_bins=liberty_bins,
                use_last_moves=use_last_moves
            )
            actual_channels = int(conv_weight.shape[1])
            if expected_channels != actual_channels:
                print(
                    "Warning: feature override mismatch. "
                    f"expected_in_channels={expected_channels}, checkpoint_in_channels={actual_channels}"
                )

        print(
            f"Network config: num_res_blocks={num_res_blocks}, num_channels={num_channels}, "
            f"use_liberty_features={use_liberty_features}, liberty_bins={liberty_bins}, "
            f"use_last_moves={use_last_moves}, head_type={head_type}"
        )

        network = AlphaZeroNetwork(
            board_size=self.board_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            use_liberty_features=use_liberty_features,
            liberty_bins=liberty_bins,
            use_last_moves=use_last_moves,
            head_type=head_type
        )
        missing, unexpected = load_state_dict_safe(network, checkpoint['network'])
        network.eval()
        if self.debug_model:
            print(
                f"Loaded state dict: missing={missing}, unexpected={unexpected} "
                f"(checkpoint={checkpoint_path})"
            )

        mcts = AlphaZeroMCTS(
            network,
            self.env,
            num_simulations=self.alphazero_simulations,
            c_puct=1.5,
            use_liberty_features=use_liberty_features,
            liberty_bins=liberty_bins,
            use_last_moves=use_last_moves
        )

        label = os.path.splitext(os.path.basename(checkpoint_path))[0]
        return {
            "network": network,
            "mcts": mcts,
            "use_last_moves": use_last_moves,
            "use_liberty_features": use_liberty_features,
            "liberty_bins": liberty_bins,
            "label": label,
            "checkpoint": checkpoint
        }

    def _load_alphazero(self):
        """AlphaZero model load."""
        mcts_cls = MCTSFast if self.use_fast_env else MCTS
        model = self._load_alphazero_model(
            self.checkpoint_path,
            use_last_moves_override=self.az_use_last_moves_override,
            use_liberty_features_override=self.az_use_liberty_features_override,
            liberty_bins_override=self.az_liberty_bins_override
        )
        if model is None:
            print("Falling back to MCTS")
            self.ai_type = 'mcts'
            self.mcts = mcts_cls(self.env, simulations_per_move=self.mcts_simulations)
            return

        self.alphazero_network = model["network"]
        self.alphazero_mcts = model["mcts"]
        self.az_use_last_moves = model["use_last_moves"]
        self.az_use_liberty_features = model["use_liberty_features"]
        self.az_liberty_bins = model["liberty_bins"]
        self.az_label = model["label"]

        print(f"Loaded AlphaZero from: {self.checkpoint_path}")
        if 'iteration' in model["checkpoint"]:
            print(f"  Iteration: {model['checkpoint']['iteration']}")
        if 'total_games' in model["checkpoint"]:
            print(f"  Total games: {model['checkpoint']['total_games']}")

        if self.ai_type == 'spectate' and self.spectate_mode == "az_vs_az":
            model_b = self._load_alphazero_model(
                self.checkpoint_path_b,
                use_last_moves_override=self.az_use_last_moves_override_b,
                use_liberty_features_override=self.az_use_liberty_features_override_b,
                liberty_bins_override=self.az_liberty_bins_override_b
            )
            if model_b is None:
                print("Spectate az_vs_az requested but second checkpoint missing. Falling back to az_vs_mcts.")
                self.spectate_mode = "az_vs_mcts"
            else:
                self.alphazero_network_b = model_b["network"]
                self.alphazero_mcts_b = model_b["mcts"]
                self.azb_use_last_moves = model_b["use_last_moves"]
                self.azb_use_liberty_features = model_b["use_liberty_features"]
                self.azb_liberty_bins = model_b["liberty_bins"]
                self.az_label_b = model_b["label"]

    def _make_state(self, use_last_moves):
        base_state = (
            self.env.board.copy(),
            self.env.current_player,
            self.env.consecutive_passes
        )
        if use_last_moves:
            return (
                self.env.board.copy(),
                self.env.current_player,
                self.env.consecutive_passes,
                self.last_moves
            )
        return base_state

    def _get_active_az_model(self):
        if self.ai_type == 'spectate' and self.spectate_mode == 'az_vs_az' and self.alphazero_network_b is not None:
            if (self.env.current_player == 1) == self.alphazero_black:
                return (self.alphazero_network, self.az_use_last_moves, self.az_use_liberty_features, self.az_liberty_bins)
            return (self.alphazero_network_b, self.azb_use_last_moves, self.azb_use_liberty_features, self.azb_liberty_bins)
        if self.alphazero_network is None:
            return None
        return (self.alphazero_network, self.az_use_last_moves, self.az_use_liberty_features, self.az_liberty_bins)

    def _get_ownership_probs(self):
        active = self._get_active_az_model()
        if active is None:
            if self.debug_model:
                print("Ownership heads unavailable: AlphaZero model not loaded.")
            return None, None, None, None
        network, use_last_moves, use_liberty_features, liberty_bins = active
        state = self._make_state(use_last_moves)
        cache_key = (
            id(network),
            self.env.current_player,
            self.env.consecutive_passes,
            self.last_moves,
            self.env.board.tobytes()
        )
        if cache_key == self._aux_cache_key and self._aux_cache is not None:
            return self._aux_cache
        encoded = encode_board_from_state(
            state[0], state[1], self.board_size, self.last_moves,
            use_liberty_features=use_liberty_features,
            liberty_bins=liberty_bins,
            use_last_moves=use_last_moves
        )
        with torch.no_grad():
            device = next(network.parameters()).device
            x = torch.from_numpy(encoded).unsqueeze(0).to(device)
            _, _, ownership_logits, win_logit, win_type_logit = network(x)
            probs = torch.sigmoid(ownership_logits).squeeze(0).cpu().numpy()
            win_prob = torch.sigmoid(win_logit).squeeze(0).cpu().numpy().item()
            win_type_prob = torch.sigmoid(win_type_logit).squeeze(0).cpu().numpy().item()
        p_black = probs[0]
        p_white = probs[1]
        denom = p_black + p_white + 1e-6
        signed = (p_black - p_white) / denom
        confidence = np.maximum(p_black, p_white)
        self._aux_cache_key = cache_key
        self._aux_cache = (signed, confidence, win_prob, win_type_prob)
        return signed, confidence, win_prob, win_type_prob

    def _blend_color(self, signed, confidence):
        base_black = np.array([30, 30, 30], dtype=np.float32)
        base_white = np.array([235, 235, 235], dtype=np.float32)
        neutral = np.array([160, 160, 160], dtype=np.float32)
        ratio = (signed + 1.0) * 0.5
        color = base_white * (1.0 - ratio) + base_black * ratio
        color = neutral * (1.0 - confidence) + color * confidence
        return tuple(color.astype(int))

    def draw_ownership_heatmap(self):
        panel_rect = pygame.Rect(self.margin, self.board_origin_y, self.heatmap_size, self.heatmap_size)
        pygame.draw.rect(self.screen, (235, 235, 235), panel_rect)

        signed, confidence, _, _ = self._get_ownership_probs()
        if signed is None:
            signed = np.zeros((self.board_size, self.board_size), dtype=np.float32)
            confidence = np.zeros((self.board_size, self.board_size), dtype=np.float32)

        surface = pygame.Surface((self.heatmap_size, self.heatmap_size), pygame.SRCALPHA)
        cell = self.heatmap_size / max(1, self.board_size)
        for r in range(self.board_size):
            y0 = int(r * cell)
            y1 = int((r + 1) * cell)
            for c in range(self.board_size):
                x0 = int(c * cell)
                x1 = int((c + 1) * cell)
                stone = self.env.board[r, c]
                if stone == 1:
                    color_rgb = (0, 0, 0)
                elif stone == 2:
                    color_rgb = (255, 255, 255)
                elif stone == 3:
                    color_rgb = (80, 80, 80)
                else:
                    color_rgb = self._blend_color(float(signed[r, c]), float(confidence[r, c]))
                pygame.draw.rect(surface, (*color_rgb, 180), pygame.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
        self.screen.blit(surface, panel_rect.topleft)

        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        for i in range(self.board_size + 1):
            x = panel_rect.x + int(i * cell)
            y = panel_rect.y + int(i * cell)
            pygame.draw.line(self.screen, (60, 60, 60), (x, panel_rect.y), (x, panel_rect.y + self.heatmap_size), 1)
            pygame.draw.line(self.screen, (60, 60, 60), (panel_rect.x, y), (panel_rect.x + self.heatmap_size, y), 1)
        if cell >= 18:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if self.env.board[r, c] == 0:
                        value = float(signed[r, c])
                        text = f"{value:.1f}"
                        color = BLACK if value >= 0 else WHITE
                        text_surf = self.tiny_font.render(text, True, color)
                        tx = panel_rect.x + int(c * cell + cell / 2 - text_surf.get_width() / 2)
                        ty = panel_rect.y + int(r * cell + cell / 2 - text_surf.get_height() / 2)
                        self.screen.blit(text_surf, (tx, ty))
        label = self.font.render("Ownership", True, BLACK)
        self.screen.blit(label, (panel_rect.x, panel_rect.y - label.get_height() - 4))

    def draw_board(self):
        self.screen.fill(WOOD)
        self.draw_ownership_heatmap()
        
        # 그리드 그리기
        for i in range(self.board_size):
            # 가로줄
            start_pos = (self.board_origin_x + self.cell_size // 2, self.board_origin_y + self.cell_size // 2 + i * self.cell_size)
            end_pos = (self.board_origin_x + self.cell_size * self.board_size - self.cell_size // 2, self.board_origin_y + self.cell_size // 2 + i * self.cell_size)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 2)
            
            # 세로줄
            start_pos = (self.board_origin_x + self.cell_size // 2 + i * self.cell_size, self.board_origin_y + self.cell_size // 2)
            end_pos = (self.board_origin_x + self.cell_size // 2 + i * self.cell_size, self.board_origin_y + self.cell_size * self.board_size - self.cell_size // 2)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 2)

        # 영토 표시 (반투명)
        if not self.done:
            black_territory = self.env._get_territory_mask(1)
            white_territory = self.env._get_territory_mask(2)
            
            surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            
            for r in range(self.board_size):
                for c in range(self.board_size):
                    cx = self.board_origin_x + c * self.cell_size
                    cy = self.board_origin_y + r * self.cell_size
                    
                    if black_territory[r, c]:
                        pygame.draw.rect(surface, (0, 0, 0, 50), surface.get_rect())
                        self.screen.blit(surface, (cx, cy))
                        # 작은 점 표시
                        pygame.draw.circle(self.screen, BLACK, (cx + self.cell_size//2, cy + self.cell_size//2), 5)
                        
                    elif white_territory[r, c]:
                        pygame.draw.rect(surface, (255, 255, 255, 50), surface.get_rect())
                        self.screen.blit(surface, (cx, cy))
                        # 작은 점 표시
                        pygame.draw.circle(self.screen, WHITE, (cx + self.cell_size//2, cy + self.cell_size//2), 5)

        if self.show_ownership_overlay:
            signed, confidence, _, _ = self._get_ownership_probs()
            if signed is not None:
                overlay = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                for r in range(self.board_size):
                    for c in range(self.board_size):
                        color_rgb = self._blend_color(float(signed[r, c]), float(confidence[r, c]))
                        alpha = int(60 + 140 * float(confidence[r, c]))
                        overlay.fill((*color_rgb, alpha))
                        self.screen.blit(
                            overlay,
                            (self.board_origin_x + c * self.cell_size, self.board_origin_y + r * self.cell_size)
                        )

        # 돌 그리기
        for r in range(self.board_size):
            for c in range(self.board_size):
                cx = self.board_origin_x + c * self.cell_size + self.cell_size // 2
                cy = self.board_origin_y + r * self.cell_size + self.cell_size // 2
                
                stone = self.env.board[r, c]
                if stone == 1:  # Black
                    pygame.draw.circle(self.screen, BLACK, (cx, cy), self.cell_size // 2 - 5)
                elif stone == 2:  # White
                    pygame.draw.circle(self.screen, WHITE, (cx, cy), self.cell_size // 2 - 5)
                    pygame.draw.circle(self.screen, BLACK, (cx, cy), self.cell_size // 2 - 5, 1) # 테두리
                elif stone == 3:  # Wall
                    rect = pygame.Rect(0, 0, self.cell_size // 2, self.cell_size // 2)
                    rect.center = (cx, cy)
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)

    def draw_color_selection_screen(self):
        """색상 선택 화면 그리기"""
        self.screen.fill(WOOD)
        
        # 제목
        title = "Choose Your Color"
        title_surf = self.large_font.render(title, True, BLACK)
        title_rect = title_surf.get_rect(center=(self.width // 2, self.height // 3))
        self.screen.blit(title_surf, title_rect)
        
        # 흑돌 버튼
        pygame.draw.rect(self.screen, BLACK, self.black_button_rect)
        pygame.draw.rect(self.screen, WHITE, self.black_button_rect, 3)
        black_text = self.font.render("BLACK", True, WHITE)
        black_text_rect = black_text.get_rect(center=self.black_button_rect.center)
        self.screen.blit(black_text, black_text_rect)
        
        # 백돌 버튼
        pygame.draw.rect(self.screen, WHITE, self.white_button_rect)
        pygame.draw.rect(self.screen, BLACK, self.white_button_rect, 3)
        white_text = self.font.render("WHITE", True, BLACK)
        white_text_rect = white_text.get_rect(center=self.white_button_rect.center)
        self.screen.blit(white_text, white_text_rect)
        
        # 설명 텍스트
        info_text = "(Black plays first)"
        info_surf = self.font.render(info_text, True, GRAY)
        info_rect = info_surf.get_rect(center=(self.width // 2, self.height * 2 // 3))
        self.screen.blit(info_surf, info_rect)
    
    def draw_spectate_ui(self):
        # Spectate mode label
        if self.spectate_mode == 'az_vs_az' and self.alphazero_mcts_b is not None:
            black_name = self.az_label if self.alphazero_black else self.az_label_b
            white_name = self.az_label_b if self.alphazero_black else self.az_label
            spectate_text = f"[SPECTATE] {black_name} (Black) vs {white_name} (White)"
        else:
            if self.alphazero_black:
                spectate_text = "[SPECTATE] AlphaZero (Black) vs MCTS (White)"
            else:
                spectate_text = "[SPECTATE] MCTS (Black) vs AlphaZero (White)"
        text_surf = self.font.render(spectate_text, True, RED)
        text_rect = text_surf.get_rect(center=(self.width // 2, 15))
        pygame.draw.rect(self.screen, WHITE, text_rect.inflate(20, 10))
        self.screen.blit(text_surf, text_rect)
    def draw_ui(self):
        # 하단 패널 배경
        pygame.draw.rect(self.screen, GRAY, (0, self.height - self.info_panel_height, self.width, self.info_panel_height))
        
        # AI 생각 중 표시
        if self.ai_thinking:
            thinking_text = "AI is thinking..."
            text_surf = self.large_font.render(thinking_text, True, RED)
            text_rect = text_surf.get_rect(center=(self.width // 2, self.height // 2))
            pygame.draw.rect(self.screen, WHITE, text_rect.inflate(40, 20))
            self.screen.blit(text_surf, text_rect)
        
        # 현재 턴 표시
        if not self.done:
            turn_text = "Black's Turn" if self.env.current_player == 1 else "White's Turn"
            if self.ai_type == 'spectate':
                if self.spectate_mode == 'az_vs_az' and self.alphazero_mcts_b is not None:
                    if self.alphazero_black:
                        ai_name = self.az_label if self.env.current_player == 1 else self.az_label_b
                    else:
                        ai_name = self.az_label_b if self.env.current_player == 1 else self.az_label
                else:
                    if self.alphazero_black:
                        ai_name = 'AlphaZero' if self.env.current_player == 1 else 'MCTS'
                    else:
                        ai_name = 'MCTS' if self.env.current_player == 1 else 'AlphaZero'
                turn_text += f" ({ai_name})"
            elif self.ai_type != 'human' and self.human_player is not None:
                if self.env.current_player == self.human_player:
                    turn_text += " (You)"
                else:
                    ai_name = "AlphaZero" if self.ai_type == 'alphazero' else "MCTS"
                    turn_text += f" ({ai_name})"
            
            color = BLACK if self.env.current_player == 1 else WHITE
            text_surf = self.font.render(turn_text, True, color)
            if self.env.current_player == 2: # 흰색 턴일 때 글씨 잘 보이게 배경 추가
                 pygame.draw.rect(self.screen, BLACK, (15, self.height - 85, text_surf.get_width() + 10, text_surf.get_height() + 10))
            self.screen.blit(text_surf, (20, self.height - 80))
            
            # 영토 점수 표시
            scores = self.env.get_territory_scores()
            score_text = f"Territory - B: {scores['black']}  W: {scores['white']}"
            score_surf = self.font.render(score_text, True, BLACK)
            self.screen.blit(score_surf, (20, self.height - 40))

            _, _, win_prob, win_type_prob = self._get_ownership_probs()
            if win_prob is not None:
                win_text = f"Win P(B): {win_prob:.2f}  P(W): {1.0 - win_prob:.2f}"
                win_surf = self.font.render(win_text, True, BLACK)
                self.screen.blit(
                    win_surf,
                    (self.width // 2 - win_surf.get_width() // 2, self.height - 80)
                )

                win_type_text = f"WinType P(Cap): {win_type_prob:.2f}  Terr: {1.0 - win_type_prob:.2f}"
                win_type_surf = self.font.render(win_type_text, True, BLACK)
                self.screen.blit(
                    win_type_surf,
                    (self.width // 2 - win_type_surf.get_width() // 2, self.height - 50)
                )
            
        else:
            # 게임 종료 메시지
            if "winner" in self.info:
                result_text = f"Game Over! {self.info['winner']} Wins!"
                detail_text = f"B: {self.info['black_territory']} vs W: {self.info['white_territory']}"
            else:
                # 캡처나 자살수 등
                result_text = "Game Over!"
                detail_text = self.info.get('result', '')
                
            text_surf = self.font.render(result_text, True, RED)
            self.screen.blit(text_surf, (20, self.height - 80))
            detail_surf = self.font.render(detail_text, True, BLACK)
            self.screen.blit(detail_surf, (20, self.height - 40))

        # 메시지 표시 (에러 등)
        if self.message:
            msg_surf = self.font.render(self.message, True, RED)
            self.screen.blit(msg_surf, (self.width // 2 - msg_surf.get_width() // 2, 10))

        # 버튼 그리기
        # Pass 버튼
        pygame.draw.rect(self.screen, BLUE, self.pass_button_rect)
        pass_text = self.font.render("PASS", True, WHITE)
        text_rect = pass_text.get_rect(center=self.pass_button_rect.center)
        self.screen.blit(pass_text, text_rect)
        
        # Reset 버튼
        pygame.draw.rect(self.screen, GREEN, self.reset_button_rect)
        reset_text = self.font.render("RESET", True, WHITE)
        text_rect = reset_text.get_rect(center=self.reset_button_rect.center)
        self.screen.blit(reset_text, text_rect)

        overlay_label = "OWN ON" if self.show_ownership_overlay else "OWN OFF"
        pygame.draw.rect(self.screen, (90, 90, 90), self.overlay_button_rect)
        overlay_btn_text = self.font.render(overlay_label, True, WHITE)
        overlay_btn_rect = overlay_btn_text.get_rect(center=self.overlay_button_rect.center)
        self.screen.blit(overlay_btn_text, overlay_btn_rect)

        overlay_text = f"Ownership overlay: {'ON' if self.show_ownership_overlay else 'OFF'} (O)"
        overlay_surf = self.font.render(overlay_text, True, BLACK)
        self.screen.blit(overlay_surf, (self.width - overlay_surf.get_width() - 20, self.height - 40))

    def handle_click(self, pos):
        self.message = ""
        
        # 색상 선택 화면
        if not self.game_started:
            if self.black_button_rect.collidepoint(pos):
                self.human_player = 1  # 흑 선택
                self.game_started = True
                return
            elif self.white_button_rect.collidepoint(pos):
                self.human_player = 2  # 백 선택
                self.game_started = True
                return
            return
        
        # AI가 생각 중이면 입력 무시
        if self.ai_thinking:
            return
        
        # 버튼 클릭 확인
        if self.pass_button_rect.collidepoint(pos):
            if not self.done:
                self.last_human_action = self.env.pass_action  # 패스 기록
                self.obs, reward, self.done, self.info = self.env.step(self.env.pass_action)
                if self.done:
                    print("Game Over:", self.info)
            return
            
        if self.reset_button_rect.collidepoint(pos):
            self.obs = self.env.reset()
            self.done = False
            self.info = {}
            self.message = "Game Reset"
            self.last_human_action = None  # 리셋 시 초기화
            self.last_ai_action = None
            self.last_moves = (None, None)
            self.game_started = False  # reset to color selection
            self.human_player = None
            # AI 재초기화
            self._init_ai()
            return

        if self.overlay_button_rect.collidepoint(pos):
            self.show_ownership_overlay = not self.show_ownership_overlay
            return

        if self.done:
            return
        
        # AI 사용 시 자신의 턴만 입력 받음
        if self.ai_type != 'human' and self.env.current_player != self.human_player:
            return

        # 보드 클릭 확인
        x, y = pos
        # 마진 제외
        board_width = self.cell_size * self.board_size
        if x < self.board_origin_x or x > self.board_origin_x + board_width:
            return
        if y < self.board_origin_y or y > self.board_origin_y + board_width:
            return
            
        # 좌표 변환
        c = (x - self.board_origin_x) // self.cell_size
        r = (y - self.board_origin_y) // self.cell_size
        
        if 0 <= r < self.board_size and 0 <= c < self.board_size:
            action = r * self.board_size + c
            
            # 유효성 검사 (미리 체크해서 메시지 띄우기)
            legal_moves = self.env.get_legal_moves()
            if legal_moves[action] == 0:
                self.message = "Illegal Move!"
                return
            
            self.last_human_action = action  # 사람의 수 기록
            self.obs, reward, self.done, self.info = self.env.step(action)
            if action != self.env.pass_action:
                self.last_moves = (action, self.last_moves[0])
            if self.done:
                print("Game Over:", self.info)

    def ai_move(self):
        """AI? ?? ?? ??"""
        if self.done or self.ai_type == 'human' or not self.game_started:
            return

        base_state = (
            self.env.board.copy(),
            self.env.current_player,
            self.env.consecutive_passes
        )
        az_state = base_state
        if self.az_use_last_moves:
            az_state = (
                self.env.board.copy(),
                self.env.current_player,
                self.env.consecutive_passes,
                self.last_moves
            )

        # ?? ??: AlphaZero vs MCTS
        if self.ai_type == 'spectate':
            if self.spectate_mode == 'az_vs_az' and self.alphazero_mcts_b is not None:
                if self.alphazero_black:
                    black_mcts = self.alphazero_mcts
                    black_state = self._make_state(self.az_use_last_moves)
                    black_name = self.az_label
                    white_mcts = self.alphazero_mcts_b
                    white_state = self._make_state(self.azb_use_last_moves)
                    white_name = self.az_label_b
                else:
                    black_mcts = self.alphazero_mcts_b
                    black_state = self._make_state(self.azb_use_last_moves)
                    black_name = self.az_label_b
                    white_mcts = self.alphazero_mcts
                    white_state = self._make_state(self.az_use_last_moves)
                    white_name = self.az_label
                if self.env.current_player == 1:
                    action, _ = black_mcts.run(black_state, temperature=0)
                    ai_name = black_name
                else:
                    action, _ = white_mcts.run(white_state, temperature=0)
                    ai_name = white_name
            else:
                is_alphazero_turn = (self.env.current_player == 1) == self.alphazero_black
                if is_alphazero_turn:
                    action, _ = self.alphazero_mcts.run(self._make_state(self.az_use_last_moves), temperature=0)
                    ai_name = self.az_label
                else:
                    action = self.mcts.run(self._make_state(False), opponent_last_action=self.last_ai_action)
                    ai_name = 'MCTS'

            # ?? ??
            self.obs, reward, self.done, self.info = self.env.step(action)
            if action != self.env.pass_action:
                self.last_moves = (action, self.last_moves[0])
            self.last_ai_action = action  # ?? AI? ?? ??

            # ? ??
            if action == self.env.pass_action:
                print(f"{ai_name}: Pass")
            else:
                r, c = action // self.board_size, action % self.board_size
                print(f"{ai_name}: ({r}, {c})")

            if self.done:
                print('Game Over:', self.info)
            return
        ai_player = 1 if self.human_player == 2 else 2  # AI? ??? ???

        if self.env.current_player == ai_player:  # AI? ?
            # AI ??? ?? ?? ??
            if self.ai_type == 'alphazero':
                action, _ = self.alphazero_mcts.run(az_state, temperature=0)
            else:  # mcts
                action = self.mcts.run(base_state, opponent_last_action=self.last_human_action)

            # ?? ??
            self.obs, reward, self.done, self.info = self.env.step(action)
            if action != self.env.pass_action:
                self.last_moves = (action, self.last_moves[0])

            if self.done:
                print("Game Over:", self.info)

    def run(self):
        clock = pygame.time.Clock()
        running = True
        last_move_time = 0  # 관전 모드용 타이머
        
        while running:
            current_time = pygame.time.get_ticks()
            
            # 관전 모드: AI vs AI
            if self.ai_type == 'spectate' and self.game_started and not self.done:
                if not self.ai_thinking and (current_time - last_move_time) >= self.spectate_delay:
                    self.ai_thinking = True
                    
                    # 게임 화면 업데이트 (AI 생각 중 표시)
                    self.draw_board()
                    self.draw_spectate_ui()
                    self.draw_ui()
                    pygame.display.flip()
                    
                    # AI 실행
                    self.ai_move()
                    self.ai_thinking = False
                    last_move_time = pygame.time.get_ticks()
            
            # 색상 선택 화면이 아니고, AI 턴이면 AI가 수를 둠 (일반 모드)
            elif self.game_started and self.ai_type not in ['human', 'spectate'] and not self.done:
                ai_player = 1 if self.human_player == 2 else 2
                if self.env.current_player == ai_player and not self.ai_thinking:
                    self.ai_thinking = True
                    
                    # 게임 화면 업데이트 (AI 생각 중 표시)
                    self.draw_board()
                    self.draw_ui()
                    pygame.display.flip()
                    
                    # AI 실행
                    self.ai_move()
                    self.ai_thinking = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p: # P key for Pass
                         if not self.done and not self.ai_thinking and self.game_started:
                            # AI 모드에서는 자신의 턴만 Pass 가능
                            if self.ai_type == 'human' or self.env.current_player == self.human_player:
                                self.last_human_action = self.env.pass_action
                                self.obs, reward, self.done, self.info = self.env.step(self.env.pass_action)
                    elif event.key == pygame.K_r: # R key for Reset
                        if not self.ai_thinking:
                            self.obs = self.env.reset()
                            self.done = False
                            self.info = {}
                            self.message = "Game Reset"
                            self.last_human_action = None
                            self.last_ai_action = None
                            self.last_moves = (None, None)
                            self.game_started = False  # reset to color selection
                            self.human_player = None
                            self._init_ai()
                    elif event.key == pygame.K_o:  # O key for ownership overlay
                        self.show_ownership_overlay = not self.show_ownership_overlay

            # 화면 그리기
            if not self.game_started:
                self.draw_color_selection_screen()
            else:
                self.draw_board()
                if self.ai_type == 'spectate':
                    self.draw_spectate_ui()
                self.draw_ui()
                
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Great Kingdom GUI')
    parser.add_argument('--ai', type=str, default='alphazero',
                        choices=['alphazero', 'mcts', 'human', 'spectate'],
                        help='AI 타입: alphazero, mcts, human (2인 대전), spectate (AI vs AI 관전)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/board_9/center_wall_on/b8c96/alphazero_latest.pt',
                        help='AlphaZero ???????????')
    parser.add_argument('--checkpoint_b', type=str, default='checkpoints/board_7/center_wall_on/b5c96/alphazero_latest.pt',
                        help='Second AlphaZero checkpoint for az_vs_az spectate')
    parser.add_argument('--spectate_mode', type=str, default='az_vs_az',
                        choices=['az_vs_mcts', 'az_vs_az'],
                        help='Spectate mode: az_vs_mcts or az_vs_az')
    parser.add_argument('--fast_env', type=str, default='auto',
                        help='Fast env/mcts ???? (True/False/auto)')
    parser.add_argument('--mcts_sims', type=int, default=200,
                        help='Pure MCTS 시뮬레이션 횟수')
    parser.add_argument('--alphazero_sims', type=int, default=100,
                        help='AlphaZero MCTS 시뮬레이션 횟수')
    parser.add_argument('--board_size', type=int, default=9,
                        help='보드 크기')
    parser.add_argument('--profile', type=str, default='auto',
                        help='Use training config profile (name or auto)')
    parser.add_argument('--center_wall', type=str, default='auto',
                        help='center wall setting (True/False/auto)')
    parser.add_argument('--komi', type=str, default='auto',
                        help='komi setting (number/auto)')
    parser.add_argument('--alphazero_black', type=str, default='True',
                        help='관전 모드에서 AlphaZero가 흑을 잡을지 여부 (True/False/auto). False면 MCTS가 흑')
    parser.add_argument('--az_use_last_moves', type=str, default='auto',
                        help='AlphaZero feature override: last moves (true/false/auto)')
    parser.add_argument('--az_use_liberty_features', type=str, default='auto',
                        help='AlphaZero feature override: liberty features (true/false/auto)')
    parser.add_argument('--az_liberty_bins', type=str, default='auto',
                        help='AlphaZero feature override: liberty bins (int/auto)')
    parser.add_argument('--az_use_last_moves_b', type=str, default='auto',
                        help='Second AlphaZero override: last moves (true/false/auto)')
    parser.add_argument('--az_use_liberty_features_b', type=str, default='auto',
                        help='Second AlphaZero override: liberty features (true/false/auto)')
    parser.add_argument('--az_liberty_bins_b', type=str, default='auto',
                        help='Second AlphaZero override: liberty bins (int/auto)')
    parser.add_argument('--debug_model', action='store_true',
                        help='Print extra model load and aux head diagnostics')
    
    args = parser.parse_args()

    def _parse_bool_auto(value):
        v = value.lower()
        if v == 'auto':
            return None
        if v in ('true', '1', 'yes', 'y'):
            return True
        if v in ('false', '0', 'no', 'n'):
            return False
        raise ValueError(f"Invalid boolean value: {value}")

    def _parse_int_auto(value):
        v = value.lower()
        if v == 'auto':
            return None
        return int(value)
    
    center_wall_arg = args.center_wall.lower()
    if center_wall_arg == 'auto':
        center_wall = None
        if os.path.exists(args.checkpoint):
            try:
                ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            except TypeError:
                ckpt = torch.load(args.checkpoint, map_location='cpu')
            center_wall = ckpt.get('center_wall')
        if center_wall is None:
            if 'center_wall_on' in args.checkpoint:
                center_wall = True
            elif 'center_wall_off' in args.checkpoint:
                center_wall = False
        if center_wall is None:
            center_wall = True
    else:
        center_wall = center_wall_arg == 'true'

    komi_arg = args.komi.lower()
    if komi_arg == 'auto':
        komi = None
        if os.path.exists(args.checkpoint):
            try:
                ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            except TypeError:
                ckpt = torch.load(args.checkpoint, map_location='cpu')
            komi = ckpt.get('komi')
        if komi is None and args.board_size == 7:
            komi = 2
        if komi is None:
            komi = 0
    else:
        komi = float(args.komi)

    az_use_last_moves = _parse_bool_auto(args.az_use_last_moves)
    az_use_liberty_features = _parse_bool_auto(args.az_use_liberty_features)
    az_liberty_bins = _parse_int_auto(args.az_liberty_bins)
    az_use_last_moves_b = _parse_bool_auto(args.az_use_last_moves_b)
    az_use_liberty_features_b = _parse_bool_auto(args.az_use_liberty_features_b)
    az_liberty_bins_b = _parse_int_auto(args.az_liberty_bins_b)

    profile = None
    profile_name = None
    if _CONFIG_AVAILABLE:
        if args.profile and args.profile.lower() != 'auto':
            try:
                profile = get_profile(args.profile)
                profile_name = args.profile
            except Exception as e:
                print(f"Warning: failed to load profile '{args.profile}': {e}")
        elif args.profile and args.profile.lower() == 'auto':
            candidates = []
            for name in list_profiles():
                try:
                    p = get_profile(name)
                except Exception:
                    continue
                if p.get('board_size') == args.board_size and p.get('center_wall') == center_wall:
                    candidates.append(name)
            if len(candidates) == 1:
                profile_name = candidates[0]
                profile = get_profile(profile_name)
            elif len(candidates) > 1:
                print(f"Warning: multiple profiles match board/center_wall: {candidates}.")
    else:
        if args.profile and args.profile.lower() != 'auto':
            print("Warning: train.config not available; ignoring --profile.")

    if profile:
        print(f"Using profile: {profile_name}")
        if profile.get('board_size') is not None and profile.get('board_size') != args.board_size:
            print(
                f"Warning: profile board_size={profile.get('board_size')} "
                f"!= args.board_size={args.board_size}"
            )
        if komi_arg == 'auto' and profile.get('komi') is not None:
            komi = profile['komi']
        if az_use_last_moves is None and profile.get('use_last_moves') is not None:
            az_use_last_moves = profile.get('use_last_moves')
        if az_use_liberty_features is None and profile.get('use_liberty_features') is not None:
            az_use_liberty_features = profile.get('use_liberty_features')
        if az_liberty_bins is None and profile.get('liberty_bins') is not None:
            az_liberty_bins = profile.get('liberty_bins')
        if az_use_last_moves_b is None:
            az_use_last_moves_b = az_use_last_moves
        if az_use_liberty_features_b is None:
            az_use_liberty_features_b = az_use_liberty_features
        if az_liberty_bins_b is None:
            az_liberty_bins_b = az_liberty_bins

    game = GreatKingdomGUI(
        board_size=args.board_size,
        ai_type=args.ai,
        mcts_simulations=args.mcts_sims,
        alphazero_simulations=args.alphazero_sims,
        checkpoint_path=args.checkpoint,
        checkpoint_path_b=args.checkpoint_b or None,
        center_wall=center_wall,
        komi=komi,
        alphazero_black=args.alphazero_black.lower() == 'true',
        use_fast_env=args.fast_env.lower() == 'true',
        spectate_mode=args.spectate_mode,
        az_use_last_moves_override=az_use_last_moves,
        az_use_liberty_features_override=az_use_liberty_features,
        az_liberty_bins_override=az_liberty_bins,
        az_use_last_moves_override_b=az_use_last_moves_b,
        az_use_liberty_features_override_b=az_use_liberty_features_b,
        az_liberty_bins_override_b=az_liberty_bins_b,
        debug_model=args.debug_model
    )
    game.run()
