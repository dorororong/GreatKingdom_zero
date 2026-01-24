import numpy as np
from collections import OrderedDict

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

_DIR_R = np.array([-1, 1, 0, 0], dtype=np.int8)
_DIR_C = np.array([0, 0, -1, 1], dtype=np.int8)


@njit(cache=True)
def _group_liberties(board, player, start_r, start_c, visited_global):
    n = board.shape[0]
    max_cells = n * n
    q_r = np.empty(max_cells, dtype=np.int32)
    q_c = np.empty(max_cells, dtype=np.int32)
    lib_seen = np.zeros((n, n), dtype=np.uint8)
    head = 0
    tail = 1
    q_r[0] = start_r
    q_c[0] = start_c
    visited_global[start_r, start_c] = 1
    lib_count = 0
    lib_r = -1
    lib_c = -1

    while head < tail:
        cr = q_r[head]
        cc = q_c[head]
        head += 1

        for k in range(4):
            nr = cr + _DIR_R[k]
            nc = cc + _DIR_C[k]
            if nr < 0 or nr >= n or nc < 0 or nc >= n:
                continue
            val = board[nr, nc]
            if val == 0:
                if lib_seen[nr, nc] == 0:
                    lib_seen[nr, nc] = 1
                    lib_count += 1
                    lib_r = nr
                    lib_c = nc
            elif val == player and visited_global[nr, nc] == 0:
                visited_global[nr, nc] = 1
                q_r[tail] = nr
                q_c[tail] = nc
                tail += 1

    return lib_count, lib_r, lib_c


@njit(cache=True)
def _any_group_no_liberties(board, player):
    n = board.shape[0]
    visited = np.zeros((n, n), dtype=np.uint8)
    for r in range(n):
        for c in range(n):
            if board[r, c] == player and visited[r, c] == 0:
                lib_count, _, _ = _group_liberties(board, player, r, c, visited)
                if lib_count == 0:
                    return True
    return False


@njit(cache=True)
def _kill_moves_list(board, player):
    opp = 2 if player == 1 else 1
    n = board.shape[0]
    visited = np.zeros((n, n), dtype=np.uint8)
    out = np.empty(n * n, dtype=np.int32)
    out_len = 0
    for r in range(n):
        for c in range(n):
            if board[r, c] == opp and visited[r, c] == 0:
                lib_count, lib_r, lib_c = _group_liberties(board, opp, r, c, visited)
                if lib_count == 1 and lib_r >= 0:
                    out[out_len] = lib_r * n + lib_c
                    out_len += 1
    return out, out_len


@njit(cache=True)
def _territory_mask_numba(board, player):
    n = board.shape[0]
    territory = np.zeros((n, n), dtype=np.uint8)
    visited = np.zeros((n, n), dtype=np.uint8)
    opp = 2 if player == 1 else 1
    max_cells = n * n

    for r in range(n):
        for c in range(n):
            if board[r, c] != 0 or visited[r, c] == 1:
                continue

            q_r = np.empty(max_cells, dtype=np.int32)
            q_c = np.empty(max_cells, dtype=np.int32)
            region_r = np.empty(max_cells, dtype=np.int32)
            region_c = np.empty(max_cells, dtype=np.int32)
            head = 0
            tail = 1
            region_len = 1
            q_r[0] = r
            q_c[0] = c
            region_r[0] = r
            region_c[0] = c
            visited[r, c] = 1

            my_found = False
            opp_found = False
            hit_top = False
            hit_bottom = False
            hit_left = False
            hit_right = False
            neutral_seen = np.zeros((n, n), dtype=np.uint8)
            neutral_count = 0

            while head < tail:
                cr = q_r[head]
                cc = q_c[head]
                head += 1

                for k in range(4):
                    nr = cr + _DIR_R[k]
                    nc = cc + _DIR_C[k]

                    if nr < 0:
                        hit_top = True
                        continue
                    if nr >= n:
                        hit_bottom = True
                        continue
                    if nc < 0:
                        hit_left = True
                        continue
                    if nc >= n:
                        hit_right = True
                        continue

                    val = board[nr, nc]
                    if val == 0:
                        if visited[nr, nc] == 0:
                            visited[nr, nc] = 1
                            q_r[tail] = nr
                            q_c[tail] = nc
                            tail += 1
                            region_r[region_len] = nr
                            region_c[region_len] = nc
                            region_len += 1
                    elif val == player:
                        my_found = True
                    elif val == opp:
                        opp_found = True
                    elif val == 3:
                        if neutral_seen[nr, nc] == 0:
                            neutral_seen[nr, nc] = 1
                            neutral_count += 1

            if not my_found:
                continue
            if opp_found:
                continue

            total_walls = neutral_count
            if hit_top:
                total_walls += 1
            if hit_bottom:
                total_walls += 1
            if hit_left:
                total_walls += 1
            if hit_right:
                total_walls += 1
            if total_walls >= 3:
                continue

            for i in range(region_len):
                territory[region_r[i], region_c[i]] = 1

    return territory


class GreatKingdomEnvFast:
    def __init__(self, board_size=5, center_wall=True, komi=0, territory_cache_max=200000):
        self.board_size = board_size
        self.center_wall = center_wall
        self.komi = komi
        self.action_space_size = board_size * board_size + 1
        self.pass_action = board_size * board_size

        self.board = None
        self.current_player = 1
        self.consecutive_passes = 0

        self._territory_cache = OrderedDict()
        self._territory_cache_max = territory_cache_max

        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.consecutive_passes = 0
        self._territory_cache.clear()

        if self.center_wall:
            mid = self.board_size // 2
            self.board[mid, mid] = 3

        return self.get_observation()

    def get_observation(self):
        return self.board.copy()

    def _territory_cache_key(self, player):
        return (self.board.tobytes(), int(player))

    def _get_territory_mask(self, player):
        key = self._territory_cache_key(player)
        cached = self._territory_cache.get(key)
        if cached is not None:
            self._territory_cache.move_to_end(key)
            return cached

        mask = _territory_mask_numba(self.board, player).astype(bool)
        self._territory_cache[key] = mask
        if self._territory_cache_max is not None and len(self._territory_cache) > self._territory_cache_max:
            self._territory_cache.popitem(last=False)
        return mask

    def get_legal_moves(self):
        legal_mask = (self.board == 0)
        opp_player = 2 if self.current_player == 1 else 1
        opp_territory = self._get_territory_mask(opp_player)
        legal_mask[opp_territory] = False

        candidates = np.argwhere(legal_mask)
        for x, y in candidates:
            self.board[x, y] = self.current_player

            capture_flag = _any_group_no_liberties(self.board, opp_player)
            if not capture_flag:
                visited = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
                my_libs, _, _ = _group_liberties(self.board, self.current_player, x, y, visited)
                if my_libs == 0:
                    legal_mask[x, y] = False

            self.board[x, y] = 0

        legal_flat = legal_mask.flatten()
        legal_with_pass = np.append(legal_flat, True)
        return legal_with_pass

    def step(self, action_idx):
            opp_player = 2 if self.current_player == 1 else 1
    
            if action_idx == self.pass_action:
                self.consecutive_passes += 1
                if self.consecutive_passes >= 2:
                    return self._end_game_by_territory()
                self.current_player = opp_player
                return self.board, 0, False, {"action": "pass"}
    
            self.consecutive_passes = 0
            x, y = divmod(action_idx, self.board_size)
    
            if self.board[x, y] != 0:
                return self.board, -1, True, {"error": "Invalid Move: Occupied"}
    
            if self._get_territory_mask(opp_player)[x, y]:
                return self.board, -1, True, {"error": "Invalid Move: Opponent Territory"}
    
            self.board[x, y] = self.current_player
            self._territory_cache.clear()
    
            if _any_group_no_liberties(self.board, opp_player):
                return self.board, 1, True, {"result": "Win by Capture"}
    
            visited = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
            my_libs, _, _ = _group_liberties(self.board, self.current_player, x, y, visited)
            if my_libs == 0:
                return self.board, -1, True, {"result": "Loss by Suicide"}
    
            if np.sum(self.board == 0) == 0:
                return self.board, 0, True, {"result": "Draw (Board Full)"}
    
            self.current_player = opp_player
            return self.board, 0, False, {}
    
    def _end_game_by_territory(self):
        black_territory = self._get_territory_mask(1)
        white_territory = self._get_territory_mask(2)

        black_score = np.sum(black_territory)
        white_score = np.sum(white_territory)

        info = {
            "result": "Territory Count",
            "black_territory": int(black_score),
            "white_territory": int(white_score)
        }

        if black_score >= white_score + self.komi:
            reward = 1 if self.current_player == 1 else -1
            info["winner"] = "Black"
        elif black_score < white_score + self.komi:
            reward = 1 if self.current_player == 2 else -1
            info["winner"] = "White"
        else:
            reward = 0
            info["winner"] = "Draw"

        return self.board, reward, True, info

    def get_territory_scores(self):
        black_territory = np.sum(self._get_territory_mask(1))
        white_territory = np.sum(self._get_territory_mask(2))
        return {"black": int(black_territory), "white": int(white_territory)}

    def get_kill_moves(self, player):
        moves, count = _kill_moves_list(self.board, player)
        return [int(m) for m in moves[:count]]

    def get_kill_moves_bfs(self, player):
        return self.get_kill_moves(player)

    def fast_copy(self):
        cls = self.__class__
        new_env = cls.__new__(cls)

        new_env.board_size = self.board_size
        new_env.center_wall = self.center_wall
        new_env.komi = self.komi
        new_env.action_space_size = self.action_space_size
        new_env.pass_action = self.pass_action
        new_env.current_player = self.current_player
        new_env.consecutive_passes = self.consecutive_passes

        new_env.board = self.board.copy()
        new_env._territory_cache = OrderedDict()
        new_env._territory_cache_max = self._territory_cache_max
        return new_env

    def render(self):
        symbols = {0: '.', 1: 'B', 2: 'W', 3: '#'}
        print("  " + " ".join([str(i) for i in range(self.board_size)]))
        for i in range(self.board_size):
            row_str = f"{i} "
            for j in range(self.board_size):
                row_str += symbols[self.board[i, j]] + " "
            print(row_str)
        scores = self.get_territory_scores()
        print(f"Current Turn: {'Black (B)' if self.current_player == 1 else 'White (W)'}")
        print(f"Territory - Black: {scores['black']}, White: {scores['white']}")
        print("-" * 20)
