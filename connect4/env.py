import numpy as np


class Connect4Env:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.board_size = rows  # kept for compatibility in some helpers
        self.action_space_size = cols
        self.board = None
        self.current_player = 1
        self.consecutive_passes = 0
        self.reset()

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.consecutive_passes = 0
        return self.get_observation()

    def get_observation(self):
        return self.board.copy()

    def get_legal_moves(self):
        legal = np.zeros(self.action_space_size, dtype=bool)
        for c in range(self.cols):
            legal[c] = (self.board[0, c] == 0)
        return legal

    def _drop_piece(self, col):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = self.current_player
                return r, col
        return None

    def _check_winner_from(self, r, c):
        player = self.board[r, c]
        if player == 0:
            return False
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                nr, nc = r + sign * dr, c + sign * dc
                while 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr, nc] == player:
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= 4:
                return True
        return False

    def step(self, action):
        if not (0 <= action < self.cols):
            return self.board, -1, True, {"error": "Invalid Move: Out of bounds"}

        if self.board[0, action] != 0:
            return self.board, -1, True, {"error": "Invalid Move: Column Full"}

        placed = self._drop_piece(action)
        if placed is None:
            return self.board, -1, True, {"error": "Invalid Move: Column Full"}

        r, c = placed
        if self._check_winner_from(r, c):
            return self.board, 1, True, {"result": "Win by Connect4", "winner": self.current_player}

        if np.all(self.board != 0):
            return self.board, 0, True, {"result": "Draw (Board Full)", "winner": 0}

        self.current_player = 2 if self.current_player == 1 else 1
        return self.board, 0, False, {}

    def fast_copy(self):
        cls = self.__class__
        new_env = cls.__new__(cls)
        new_env.rows = self.rows
        new_env.cols = self.cols
        new_env.board_size = self.board_size
        new_env.action_space_size = self.action_space_size
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.consecutive_passes = self.consecutive_passes
        return new_env

    def render(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        for r in range(self.rows):
            row = " ".join(symbols[self.board[r, c]] for c in range(self.cols))
            print(row)
        print(" " + " ".join(str(c) for c in range(self.cols)))
        print(f"Turn: {'X' if self.current_player == 1 else 'O'}")
        print("-" * 20)
