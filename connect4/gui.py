import pygame
import sys
import os
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from connect4.env import Connect4Env
from connect4.mcts import MCTS
from connect4.network import AlphaZeroNetwork
from connect4.mcts_alphazero import AlphaZeroMCTS


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (80, 80, 80)
BLUE = (30, 70, 200)
RED = (200, 50, 50)
YELLOW = (240, 210, 50)


class Connect4GUI:
    def __init__(
        self,
        rows=6,
        cols=7,
        cell_size=80,
        mode="human",
        mcts_sims=200,
        az_sims=200,
        az_ckpt="connect4/checkpoints/alphazero_latest.pt",
        az_res_blocks=3,
        az_channels=64,
        az_device=None,
        human_player=1
    ):
        pygame.init()
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.margin = 20
        self.info_height = 60
        self.mode = mode
        self.mcts_sims = mcts_sims
        self.az_sims = az_sims
        self.az_ckpt = az_ckpt
        self.az_res_blocks = az_res_blocks
        self.az_channels = az_channels
        self.az_device = az_device
        self.human_player = human_player  # 1: human first (Red), 2: human second (Yellow)

        self.width = self.cols * self.cell_size + self.margin * 2
        self.height = self.rows * self.cell_size + self.margin * 2 + self.info_height

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Connect4")
        self.font = pygame.font.SysFont("malgungothic", 24)

        self.env = Connect4Env(rows=rows, cols=cols)
        self.mcts = None
        self.az_mcts = None
        if self.mode in ("mcts_vs_human", "mcts_vs_mcts"):
            self.mcts = MCTS(self.env, simulations_per_move=self.mcts_sims)
        if self.mode in ("az_vs_human", "az_vs_az"):
            self._load_alphazero()
        self.done = False
        self.info = {}
        self.message = ""

        self.reset_button = pygame.Rect(self.width - 140, self.height - 50, 120, 35)

    def _load_alphazero(self):
        if not os.path.exists(self.az_ckpt):
            raise FileNotFoundError(f"AlphaZero checkpoint not found: {self.az_ckpt}")

        device = torch.device(
            self.az_device if self.az_device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        network = AlphaZeroNetwork(
            rows=self.rows,
            cols=self.cols,
            num_res_blocks=self.az_res_blocks,
            num_channels=self.az_channels
        ).to(device)
        checkpoint = torch.load(self.az_ckpt, map_location=device)
        network.load_state_dict(checkpoint["network"])
        network.eval()

        self.az_mcts = AlphaZeroMCTS(
            network,
            self.env,
            num_simulations=self.az_sims,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25
        )
        print(f"Loaded AlphaZero model: {self.az_ckpt} on {device}")

    def draw_board(self):
        self.screen.fill(BLUE)

        for r in range(self.rows):
            for c in range(self.cols):
                cx = self.margin + c * self.cell_size + self.cell_size // 2
                cy = self.margin + r * self.cell_size + self.cell_size // 2
                pygame.draw.circle(self.screen, BLACK, (cx, cy), self.cell_size // 2 - 5)

                val = self.env.board[r, c]
                if val == 1:
                    pygame.draw.circle(self.screen, RED, (cx, cy), self.cell_size // 2 - 8)
                elif val == 2:
                    pygame.draw.circle(self.screen, YELLOW, (cx, cy), self.cell_size // 2 - 8)

    def draw_ui(self):
        pygame.draw.rect(self.screen, GRAY, (0, self.height - self.info_height, self.width, self.info_height))
        if not self.done:
            turn_text = "Red's Turn" if self.env.current_player == 1 else "Yellow's Turn"
            if self.mode != "human":
                if self.mode == "mcts_vs_human":
                    turn_text += " (You)" if self.env.current_player == self.human_player else " (MCTS)"
                elif self.mode == "mcts_vs_mcts":
                    turn_text += " (MCTS)"
                elif self.mode == "az_vs_human":
                    turn_text += " (You)" if self.env.current_player == self.human_player else " (AlphaZero)"
                elif self.mode == "az_vs_az":
                    turn_text += " (AlphaZero)"
        else:
            if "result" in self.info:
                turn_text = f"Game Over: {self.info['result']}"
            else:
                turn_text = "Game Over"

        text = self.font.render(turn_text, True, WHITE)
        self.screen.blit(text, (20, self.height - 45))

        pygame.draw.rect(self.screen, WHITE, self.reset_button)
        reset_text = self.font.render("RESET", True, BLACK)
        self.screen.blit(reset_text, reset_text.get_rect(center=self.reset_button.center))

        if self.message:
            msg = self.font.render(self.message, True, WHITE)
            self.screen.blit(msg, (20, 10))

    def handle_click(self, pos):
        self.message = ""
        if self.reset_button.collidepoint(pos):
            self.env.reset()
            self.done = False
            self.info = {}
            return

        if self.done:
            return
        if self.mode in ("mcts_vs_mcts", "az_vs_az"):
            return
        if self.mode in ("mcts_vs_human", "az_vs_human") and self.env.current_player != self.human_player:
            return

        x, y = pos
        if y > self.height - self.info_height:
            return
        if x < self.margin or x > self.width - self.margin:
            return

        col = (x - self.margin) // self.cell_size
        legal = self.env.get_legal_moves()
        if not legal[col]:
            self.message = "Column Full!"
            return

        _, reward, done, info = self.env.step(int(col))
        self.done = done
        self.info = info
        if self.done:
            winner = info.get("winner", 0)
            if winner == 1:
                self.info["result"] = "Red Wins"
            elif winner == 2:
                self.info["result"] = "Yellow Wins"
            else:
                self.info["result"] = "Draw"

    def ai_move(self):
        if self.done:
            return
        if self.mode in ("mcts_vs_human", "az_vs_human") and self.env.current_player == self.human_player:
            return

        state = (
            self.env.board.copy(),
            self.env.current_player,
            self.env.consecutive_passes
        )
        if self.mode in ("mcts_vs_human", "mcts_vs_mcts"):
            action = self.mcts.run(state)
        else:
            self.az_mcts.num_simulations = self.az_sims
            action, _ = self.az_mcts.run(state, temperature=0.0, add_root_noise=False)
        _, reward, done, info = self.env.step(action)
        self.done = done
        self.info = info
        if self.done:
            winner = info.get("winner", 0)
            if winner == 1:
                self.info["result"] = "Red Wins"
            elif winner == 2:
                self.info["result"] = "Yellow Wins"
            else:
                self.info["result"] = "Draw"

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            if self.mode in ("mcts_vs_human", "mcts_vs_mcts", "az_vs_human", "az_vs_az") and not self.done:
                if self.mode in ("mcts_vs_mcts", "az_vs_az") or self.env.current_player != self.human_player:
                    self.ai_move()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            self.draw_board()
            self.draw_ui()
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Connect4 GUI")
    parser.add_argument("--mode", type=str, default="az_vs_human",
                        choices=["human", "mcts_vs_human", "mcts_vs_mcts", "az_vs_human", "az_vs_az"],
                        help="Game mode")
    parser.add_argument("--mcts_sims", type=int, default=5000,
                        help="MCTS simulations per move")
    parser.add_argument("--az_sims", type=int, default=300,
                        help="AlphaZero MCTS simulations per move")
    parser.add_argument("--az_ckpt", type=str, default="connect4/checkpoints/alphazero_iter200.pt",
                        help="AlphaZero checkpoint path")
    parser.add_argument("--az_res_blocks", type=int, default=3,
                        help="AlphaZero res blocks (must match checkpoint)")
    parser.add_argument("--az_channels", type=int, default=64,
                        help="AlphaZero channels (must match checkpoint)")
    parser.add_argument("--az_device", type=str, default=None,
                        help="AlphaZero device override (e.g. cpu or cuda)")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--human_player", type=int, default=2, choices=[1, 2],
                        help="Human player: 1=Red(first), 2=Yellow(second)")
    args = parser.parse_args()

    gui = Connect4GUI(
        rows=args.rows,
        cols=args.cols,
        mode=args.mode,
        mcts_sims=args.mcts_sims,
        az_sims=args.az_sims,
        az_ckpt=args.az_ckpt,
        az_res_blocks=args.az_res_blocks,
        az_channels=args.az_channels,
        az_device=args.az_device,
        human_player=args.human_player
    )
    gui.run()
