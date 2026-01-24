import argparse
import os
import pickle

import numpy as np
import pygame

from env.env import GreatKingdomEnv
from game_result import winner_label_from_step
from network import AlphaZeroNetwork
from mcts_alphazero import AlphaZeroMCTS


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)
WOOD = (222, 184, 135)
WALL_COLOR = (139, 69, 19)


def load_checkpoint(network, path, device="cpu"):
    import torch
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    network.load_state_dict(checkpoint["network"])
    network.eval()
    return checkpoint


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


def simulate_game(env, black_fn, white_fn, max_moves=200):
    env.reset()
    done = False
    moves = 0
    info = {}
    reward = 0

    states = []
    states.append({
        "board": env.board.copy(),
        "current_player": env.current_player,
        "consecutive_passes": env.consecutive_passes
    })

    while not done and moves < max_moves:
        if env.current_player == 1:
            action = black_fn(env)
        else:
            action = white_fn(env)

        _, reward, done, info = env.step(action)
        moves += 1
        states.append({
            "board": env.board.copy(),
            "current_player": env.current_player,
            "consecutive_passes": env.consecutive_passes
        })

    winner = winner_label_from_step(info, reward, env.current_player)

    return {
        "winner": winner,
        "moves": moves,
        "states": states
    }


def generate_replays(
    board_size=5,
    num_games=5,
    mcts_simulations=50,
    checkpoint_path="checkpoints/alphazero_latest.pt",
    seed=0,
    center_wall=True
):
    np.random.seed(seed)

    env = GreatKingdomEnv(board_size=board_size, center_wall=center_wall)
    network = AlphaZeroNetwork(board_size=board_size, num_res_blocks=3, num_channels=64)
    load_checkpoint(network, checkpoint_path)
    mcts = AlphaZeroMCTS(network, env, num_simulations=mcts_simulations, c_puct=1.5)

    games = []
    groups = []

    # Model vs Model
    indices = []
    for i in range(num_games):
        result = simulate_game(
            env,
            black_fn=lambda e: mcts_player(e, mcts, temperature=0),
            white_fn=lambda e: mcts_player(e, mcts, temperature=0)
        )
        result["label"] = f"Model vs Model #{i+1}"
        result["model_color"] = "Both"
        games.append(result)
        indices.append(len(games) - 1)
    groups.append({"name": "Model vs Model", "game_indices": indices})

    # Model vs Random (alternate colors)
    indices = []
    for i in range(num_games):
        model_is_black = (i % 2 == 0)
        if model_is_black:
            black_fn = lambda e: mcts_player(e, mcts, temperature=0)
            white_fn = random_player
            label = f"Model (Black) vs Random #{i+1}"
            model_color = "Black"
        else:
            black_fn = random_player
            white_fn = lambda e: mcts_player(e, mcts, temperature=0)
            label = f"Model (White) vs Random #{i+1}"
            model_color = "White"

        result = simulate_game(env, black_fn=black_fn, white_fn=white_fn)
        result["label"] = label
        result["model_color"] = model_color
        games.append(result)
        indices.append(len(games) - 1)
    groups.append({"name": "Model vs Random", "game_indices": indices})

    return {
        "board_size": board_size,
        "num_games": num_games,
        "mcts_simulations": mcts_simulations,
        "checkpoint_path": checkpoint_path,
        "center_wall": center_wall,
        "games": games,
        "groups": groups
    }


def save_replays(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_replays(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class ReplayGUI:
    def __init__(self, data, cell_size=100):
        pygame.init()
        self.data = data
        self.board_size = data["board_size"]
        self.center_wall = data.get("center_wall", True)
        self.cell_size = cell_size
        self.margin = 50
        self.info_panel_height = 120

        self.width = self.cell_size * self.board_size + 2 * self.margin
        self.height = self.cell_size * self.board_size + 2 * self.margin + self.info_panel_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Replay Viewer")

        self.font = pygame.font.SysFont("malgungothic", 22)
        self.large_font = pygame.font.SysFont("malgungothic", 32)

        self.env = GreatKingdomEnv(board_size=self.board_size, center_wall=self.center_wall)

        self.game_index = 0
        self.move_index = 0

        self.prev_game_rect = pygame.Rect(20, self.height - 90, 130, 40)
        self.next_game_rect = pygame.Rect(160, self.height - 90, 130, 40)
        self.prev_move_rect = pygame.Rect(self.width - 300, self.height - 90, 130, 40)
        self.next_move_rect = pygame.Rect(self.width - 150, self.height - 90, 130, 40)

    def _current_game(self):
        return self.data["games"][self.game_index]

    def _current_state(self):
        game = self._current_game()
        return game["states"][self.move_index]

    def _set_env_state(self, state):
        self.env.board = state["board"].copy()
        self.env.current_player = state["current_player"]
        self.env.consecutive_passes = state["consecutive_passes"]

    def draw_board(self):
        self.screen.fill(WOOD)

        for i in range(self.board_size):
            start_pos = (self.margin + self.cell_size // 2, self.margin + self.cell_size // 2 + i * self.cell_size)
            end_pos = (self.width - self.margin - self.cell_size // 2, self.margin + self.cell_size // 2 + i * self.cell_size)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 2)

            start_pos = (self.margin + self.cell_size // 2 + i * self.cell_size, self.margin + self.cell_size // 2)
            end_pos = (self.margin + self.cell_size // 2 + i * self.cell_size, self.height - self.info_panel_height - self.margin - self.cell_size // 2)
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 2)

        state = self._current_state()
        self._set_env_state(state)

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
                    pygame.draw.circle(self.screen, BLACK, (cx + self.cell_size // 2, cy + self.cell_size // 2), 5)
                elif white_territory[r, c]:
                    pygame.draw.rect(surface, (255, 255, 255, 50), surface.get_rect())
                    self.screen.blit(surface, (cx, cy))
                    pygame.draw.circle(self.screen, WHITE, (cx + self.cell_size // 2, cy + self.cell_size // 2), 5)

        for r in range(self.board_size):
            for c in range(self.board_size):
                cx = self.margin + c * self.cell_size + self.cell_size // 2
                cy = self.margin + r * self.cell_size + self.cell_size // 2

                stone = self.env.board[r, c]
                if stone == 1:
                    pygame.draw.circle(self.screen, BLACK, (cx, cy), self.cell_size // 2 - 5)
                elif stone == 2:
                    pygame.draw.circle(self.screen, WHITE, (cx, cy), self.cell_size // 2 - 5)
                    pygame.draw.circle(self.screen, BLACK, (cx, cy), self.cell_size // 2 - 5, 1)
                elif stone == 3:
                    rect = pygame.Rect(0, 0, self.cell_size // 2, self.cell_size // 2)
                    rect.center = (cx, cy)
                    pygame.draw.rect(self.screen, WALL_COLOR, rect)

    def draw_ui(self):
        pygame.draw.rect(self.screen, GRAY, (0, self.height - self.info_panel_height, self.width, self.info_panel_height))

        game = self._current_game()
        title = f"{game['label']} | Winner: {game['winner']} | Moves: {game['moves']}"
        title_surf = self.font.render(title, True, BLACK)
        self.screen.blit(title_surf, (20, self.height - 115))

        move_text = f"Game {self.game_index + 1}/{len(self.data['games'])} | Move {self.move_index}/{len(game['states']) - 1}"
        move_surf = self.font.render(move_text, True, BLACK)
        self.screen.blit(move_surf, (20, self.height - 75))

        turn_text = "Black's Turn" if self.env.current_player == 1 else "White's Turn"
        turn_surf = self.font.render(turn_text, True, BLACK)
        self.screen.blit(turn_surf, (20, self.height - 45))

        pygame.draw.rect(self.screen, BLUE, self.prev_game_rect)
        pygame.draw.rect(self.screen, BLUE, self.next_game_rect)
        pygame.draw.rect(self.screen, GREEN, self.prev_move_rect)
        pygame.draw.rect(self.screen, GREEN, self.next_move_rect)

        self.screen.blit(self.font.render("PREV GAME", True, WHITE), self.prev_game_rect.move(10, 8))
        self.screen.blit(self.font.render("NEXT GAME", True, WHITE), self.next_game_rect.move(10, 8))
        self.screen.blit(self.font.render("PREV MOVE", True, WHITE), self.prev_move_rect.move(10, 8))
        self.screen.blit(self.font.render("NEXT MOVE", True, WHITE), self.next_move_rect.move(10, 8))

    def handle_click(self, pos):
        if self.prev_game_rect.collidepoint(pos):
            if self.game_index > 0:
                self.game_index -= 1
                self.move_index = 0
        elif self.next_game_rect.collidepoint(pos):
            if self.game_index < len(self.data["games"]) - 1:
                self.game_index += 1
                self.move_index = 0
        elif self.prev_move_rect.collidepoint(pos):
            if self.move_index > 0:
                self.move_index -= 1
        elif self.next_move_rect.collidepoint(pos):
            if self.move_index < len(self._current_game()["states"]) - 1:
                self.move_index += 1

    def handle_key(self, key):
        if key == pygame.K_LEFT:
            if self.move_index > 0:
                self.move_index -= 1
        elif key == pygame.K_RIGHT:
            if self.move_index < len(self._current_game()["states"]) - 1:
                self.move_index += 1
        elif key == pygame.K_UP:
            if self.game_index > 0:
                self.game_index -= 1
                self.move_index = 0
        elif key == pygame.K_DOWN:
            if self.game_index < len(self.data["games"]) - 1:
                self.game_index += 1
                self.move_index = 0

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)

            self.draw_board()
            self.draw_ui()
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--mcts-sims", type=int, default=50)
    parser.add_argument("--checkpoint", default="checkpoints/alphazero_latest.pt")
    parser.add_argument("--save", default="checkpoints/eval_replay.pkl")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--center_wall", type=str, default="True",
                        help="Center neutral wall placement (True/False).")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.regenerate or not os.path.exists(args.save):
        data = generate_replays(
            board_size=args.board_size,
            num_games=args.num_games,
            mcts_simulations=args.mcts_sims,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            center_wall=args.center_wall.lower() == "true"
        )
        save_replays(data, args.save)
    else:
        data = load_replays(args.save)

    gui = ReplayGUI(data)
    gui.run()


if __name__ == "__main__":
    main()
