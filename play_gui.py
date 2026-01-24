import pygame
import numpy as np
import sys
import os
import torch
from env.env import GreatKingdomEnv
from env.env_fast import GreatKingdomEnvFast
from mcts import MCTS
from mcts_fast import MCTS as MCTSFast
from network import AlphaZeroNetwork, infer_head_type_from_state_dict
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
                 center_wall=True, komi=0, alphazero_black=True, use_fast_env=False):
        pygame.init()
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = int(self.cell_size * 0.5)
        self.info_panel_height = int(self.cell_size * 1.0)
        
        self.width = self.cell_size * self.board_size + 2 * self.margin
        self.height = self.cell_size * self.board_size + 2 * self.margin + self.info_panel_height
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Great Kingdom")
        
        self.font = pygame.font.SysFont("malgungothic", 24)
        self.large_font = pygame.font.SysFont("malgungothic", 48)
        
        self.use_fast_env = use_fast_env
        env_cls = GreatKingdomEnvFast if use_fast_env else GreatKingdomEnv
        self.env = env_cls(board_size=board_size, center_wall=center_wall, komi=komi)
        self.obs = self.env.reset()
        self.done = False
        self.info = {}
        self.message = ""
        
        # AI 설정
        self.ai_type = ai_type  # 'mcts', 'alphazero', 'human' (2인 대전), 'spectate' (AI vs AI 관전)
        self.mcts = None
        self.alphazero_mcts = None
        self.alphazero_network = None
        self.mcts_simulations = mcts_simulations
        self.alphazero_simulations = alphazero_simulations
        self.checkpoint_path = checkpoint_path
        self.ai_thinking = False  # AI가 생각 중인지 표시
        self.last_human_action = None  # 사람의 마지막 수 기록 (AI에게 전달)
        self.last_ai_action = None  # AI의 마지막 수 기록 (관전 모드용)
        self.last_moves = (None, None)
        self.az_use_last_moves = False
        self.human_player = None  # 1=흑, 2=백 (None이면 색상 선택 화면)
        self.game_started = False  # 게임 시작 여부
        
        # 관전 모드 설정
        self.alphazero_black = alphazero_black  # True: AlphaZero=흑, MCTS=백 / False: 반대
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
            # 관전 모드: AlphaZero vs MCTS
            self._load_alphazero()
            self.mcts = mcts_cls(self.env, simulations_per_move=self.mcts_simulations)
            print(f"Spectate Mode: AlphaZero({self.alphazero_simulations} sims) vs MCTS({self.mcts_simulations} sims)")
            if self.alphazero_black:
                print("  AlphaZero plays Black, MCTS plays White")
            else:
                print("  MCTS plays Black, AlphaZero plays White")
            self.game_started = True  # 관전 모드는 바로 시작
        else:  # human
            print("Mode: Human vs Human")
    
    def _load_alphazero(self):
        """AlphaZero model load."""
        mcts_cls = MCTSFast if self.use_fast_env else MCTS
        if not os.path.exists(self.checkpoint_path):
            print(f"Warning: Checkpoint not found: {self.checkpoint_path}")
            print("Falling back to MCTS")
            self.ai_type = 'mcts'
            self.mcts = mcts_cls(self.env, simulations_per_move=self.mcts_simulations)
            return

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)

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

        self.az_use_last_moves = use_last_moves

        print(
            f"Network config: num_res_blocks={num_res_blocks}, num_channels={num_channels}, "
            f"use_liberty_features={use_liberty_features}, liberty_bins={liberty_bins}, "
            f"use_last_moves={use_last_moves}, head_type={head_type}"
        )

        self.alphazero_network = AlphaZeroNetwork(
            board_size=self.board_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            use_liberty_features=use_liberty_features,
            liberty_bins=liberty_bins,
            use_last_moves=use_last_moves,
            head_type=head_type
        )

        self.alphazero_network.load_state_dict(checkpoint['network'])
        self.alphazero_network.eval()

        self.alphazero_mcts = AlphaZeroMCTS(
            self.alphazero_network,
            self.env,
            num_simulations=self.alphazero_simulations,
            c_puct=1.5,
            use_liberty_features=use_liberty_features,
            liberty_bins=liberty_bins,
            use_last_moves=use_last_moves
        )

        print(f"Loaded AlphaZero from: {self.checkpoint_path}")
        if 'iteration' in checkpoint:
            print(f"  Iteration: {checkpoint['iteration']}")
        if 'total_games' in checkpoint:
            print(f"  Total games: {checkpoint['total_games']}")

    def draw_board(self):
        self.screen.fill(WOOD)
        
        # 그리드 그리기
        for i in range(self.board_size):
            # 가로줄
            start_pos = (self.margin + self.cell_size // 2, self.margin + self.cell_size // 2 + i * self.cell_size)
            end_pos = (self.width - self.margin - self.cell_size // 2, self.margin + self.cell_size // 2 + i * self.cell_size)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 2)
            
            # 세로줄
            start_pos = (self.margin + self.cell_size // 2 + i * self.cell_size, self.margin + self.cell_size // 2)
            end_pos = (self.margin + self.cell_size // 2 + i * self.cell_size, self.height - self.info_panel_height - self.margin - self.cell_size // 2)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 2)

        # 영토 표시 (반투명)
        if not self.done:
            black_territory = self.env._get_territory_mask(1)
            white_territory = self.env._get_territory_mask(2)
            
            surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            
            for r in range(self.board_size):
                for c in range(self.board_size):
                    cx = self.margin + c * self.cell_size
                    cy = self.margin + r * self.cell_size
                    
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

        # 돌 그리기
        for r in range(self.board_size):
            for c in range(self.board_size):
                cx = self.margin + c * self.cell_size + self.cell_size // 2
                cy = self.margin + r * self.cell_size + self.cell_size // 2
                
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
        """관전 모드 UI 그리기"""
        # 상단에 관전 모드 표시
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
                # 관전 모드: 어떤 AI의 턴인지 표시
                if self.alphazero_black:
                    ai_name = "AlphaZero" if self.env.current_player == 1 else "MCTS"
                else:
                    ai_name = "MCTS" if self.env.current_player == 1 else "AlphaZero"
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
            self.game_started = False  # 색상 선택 화면으로 돌아가기
            self.human_player = None
            # AI 재초기화
            self._init_ai()
            return

        if self.done:
            return
        
        # AI 사용 시 자신의 턴만 입력 받음
        if self.ai_type != 'human' and self.env.current_player != self.human_player:
            return

        # 보드 클릭 확인
        x, y = pos
        # 마진 제외
        if x < self.margin or x > self.width - self.margin:
            return
        if y < self.margin or y > self.height - self.info_panel_height - self.margin:
            return
            
        # 좌표 변환
        c = (x - self.margin) // self.cell_size
        r = (y - self.margin) // self.cell_size
        
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
            # alphazero_black=True: AlphaZero=?(1), MCTS=?(2)
            # alphazero_black=False: MCTS=?(1), AlphaZero=?(2)
            is_alphazero_turn = (self.env.current_player == 1) == self.alphazero_black

            if is_alphazero_turn:
                action, _ = self.alphazero_mcts.run(az_state, temperature=0)
                ai_name = "AlphaZero"
            else:
                action = self.mcts.run(base_state, opponent_last_action=self.last_ai_action)
                ai_name = "MCTS"

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
                print("Game Over:", self.info)
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
                            self.game_started = False
                            self.human_player = None
                            self._init_ai()

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
    parser.add_argument('--checkpoint', type=str, default='checkpoints/board_9/center_wall_on/alphazero_latest.pt',
                        help='AlphaZero 체크포인트 경로')
    parser.add_argument('--fast_env', type=str, default='auto',
                        help='Fast env/mcts ???? (True/False/auto)')
    parser.add_argument('--mcts_sims', type=int, default=200,
                        help='Pure MCTS 시뮬레이션 횟수')
    parser.add_argument('--alphazero_sims', type=int, default=600,
                        help='AlphaZero MCTS 시뮬레이션 횟수')
    parser.add_argument('--board_size', type=int, default=9,
                        help='보드 크기')
    parser.add_argument('--center_wall', type=str, default='auto',
                        help='center wall setting (True/False/auto)')
    parser.add_argument('--komi', type=str, default='auto',
                        help='komi setting (number/auto)')
    parser.add_argument('--alphazero_black', type=str, default='True',
                        help='관전 모드에서 AlphaZero가 흑을 잡을지 여부 (True/False/auto). False면 MCTS가 흑')
    
    args = parser.parse_args()
    
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

    game = GreatKingdomGUI(
        board_size=args.board_size,
        ai_type=args.ai,
        mcts_simulations=args.mcts_sims,
        alphazero_simulations=args.alphazero_sims,
        checkpoint_path=args.checkpoint,
        center_wall=center_wall,
        komi=komi,
        alphazero_black=args.alphazero_black.lower() == 'true',
        use_fast_env=args.fast_env.lower() == 'true'
    )
    game.run()
