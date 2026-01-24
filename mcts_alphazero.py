"""
AlphaZero MCTS - 신경망 기반 Monte Carlo Tree Search

기존 MCTS와의 차이점:
1. 랜덤 롤아웃 대신 Value Network로 상태 평가
2. UCB1 대신 PUCT 알고리즘 사용
3. Prior 확률을 Policy Network에서 가져옴
"""

import numpy as np
import math
import time
from collections import OrderedDict
from network import encode_board_from_state, encode_board_batch


class AlphaZeroNode:
    """AlphaZero MCTS 노드"""
    
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior  # Policy network에서 온 사전 확률
        self.children = {}  # action -> AlphaZeroNode
        self.state = None   # (board, player, passes) - 확장 시 설정
        self.player = None  # player to move at this node
        self.is_expanded = False
    
    def value(self):
        """평균 가치 반환 (방문하지 않았으면 0)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class AlphaZeroMCTS:
    """
    AlphaZero 스타일 MCTS
    
    - Selection: PUCT 알고리즘으로 노드 선택
    - Expansion: Policy Network로 prior 확률 설정
    - Evaluation: Value Network로 상태 평가 (롤아웃 없음)
    - Backpropagation: 가치 역전파
    """
    
    def __init__(
        self,
        network,
        env,
        c_puct=1.5,
        num_simulations=100,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        eval_batch_size=1,
        profile=False,
        profile_every=0,
        profile_tag="MCTS",
        use_forced_playouts=False,
        use_policy_target_pruning=False,
        forced_playout_k=2.0,
        cache_max_size=100000,
        legal_cache_max_size=200000,
        use_vector_puct=True,
        use_liberty_features=True,
        liberty_bins=2,
        use_last_moves=False
    ):
        """
        Args:
            network: AlphaZeroNetwork 인스턴스
            env: GreatKingdomEnv 인스턴스 (규칙 참조용)
            c_puct: PUCT 탐색 상수 (exploration-exploitation 균형)
            num_simulations: 시뮬레이션 횟수
        """
        self.network = network
        self.env = env
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.eval_batch_size = eval_batch_size
        self._profile_enabled = bool(profile) and (profile_every is None or profile_every > 0)
        self._profile_every = 1 if profile_every is None else int(profile_every)
        self._profile_tag = profile_tag
        self._profile_active = False
        self._profile_runs = 0
        self._profile_time = {}
        self._profile_calls = {}
        self.use_forced_playouts = use_forced_playouts
        self.use_policy_target_pruning = use_policy_target_pruning
        self.forced_playout_k = forced_playout_k
        self.use_vector_puct = use_vector_puct
        self.use_liberty_features = use_liberty_features
        self.liberty_bins = liberty_bins
        self.use_last_moves = use_last_moves

        # Simulation environment copy
        self.sim_env = env.fast_copy()

        # Simple prediction cache (key: board+player)
        self.use_cache = True
        self._predict_cache = OrderedDict()
        self._predict_cache_hits = 0
        self._predict_cache_misses = 0
        self._cache_max_size = cache_max_size
        self.use_legal_cache = True
        self._legal_cache = OrderedDict()
        self._legal_cache_max_size = legal_cache_max_size

    def clear_cache(self):
        self._predict_cache.clear()
        self._predict_cache_hits = 0
        self._predict_cache_misses = 0
        self._legal_cache.clear()

    def _profile_reset(self):
        self._profile_time = {}
        self._profile_calls = {}

    def _profile_add(self, name, dt):
        if not self._profile_active:
            return
        self._profile_time[name] = self._profile_time.get(name, 0.0) + dt
        self._profile_calls[name] = self._profile_calls.get(name, 0) + 1

    def _profile_report(self, total_time):
        if not self._profile_active:
            return
        items = sorted(self._profile_time.items(), key=lambda x: x[1], reverse=True)
        print(f"[{self._profile_tag}] total={total_time:.4f}s")
        for name, t in items:
            calls = self._profile_calls.get(name, 0)
            avg_ms = (t / calls * 1000.0) if calls else 0.0
            pct = (t / total_time * 100.0) if total_time > 0 else 0.0
            print(f"  {name}: {t:.4f}s | calls={calls} | avg={avg_ms:.3f}ms | {pct:.1f}%")

    def _normalize_last_moves(self, last_moves):
        if last_moves is None:
            return (-1, -1)
        if isinstance(last_moves, (list, tuple)):
            lm0 = last_moves[0] if len(last_moves) > 0 else None
            lm1 = last_moves[1] if len(last_moves) > 1 else None
            return (int(lm0) if lm0 is not None else -1, int(lm1) if lm1 is not None else -1)
        return (int(last_moves), -1)

    def _update_last_moves(self, last_moves, action):
        if not self.use_last_moves:
            return last_moves
        if action == self.env.pass_action:
            return last_moves
        lm0, _ = self._normalize_last_moves(last_moves)
        return (action, lm0 if lm0 != -1 else None)

    def _cache_key(self, board, player, last_moves=None):
        if not self.use_last_moves:
            lm0, lm1 = (-1, -1)
        else:
            lm0, lm1 = self._normalize_last_moves(last_moves)
        return (board.tobytes(), int(player), lm0, lm1)

    def _predict(self, board, player, last_moves=None):
        if not self.use_cache:
            t0 = time.perf_counter() if self._profile_active else None
            encoded = encode_board_from_state(
                board, player, self.env.board_size, last_moves,
                use_liberty_features=self.use_liberty_features,
                liberty_bins=self.liberty_bins,
                use_last_moves=self.use_last_moves
            )
            if t0 is not None:
                self._profile_add("encode", time.perf_counter() - t0)
            t1 = time.perf_counter() if self._profile_active else None
            result = self.network.predict(encoded)
            if t1 is not None:
                self._profile_add("network_predict", time.perf_counter() - t1)
            return result

        key = self._cache_key(board, player, last_moves)
        cached = self._predict_cache.get(key)
        if cached is not None:
            self._predict_cache_hits += 1
            if self._cache_max_size is not None:
                self._predict_cache.move_to_end(key)
            return cached

        self._predict_cache_misses += 1
        t0 = time.perf_counter() if self._profile_active else None
        encoded = encode_board_from_state(
            board, player, self.env.board_size, last_moves,
            use_liberty_features=self.use_liberty_features,
            liberty_bins=self.liberty_bins,
            use_last_moves=self.use_last_moves
        )
        if t0 is not None:
            self._profile_add("encode", time.perf_counter() - t0)
        t1 = time.perf_counter() if self._profile_active else None
        result = self.network.predict(encoded)
        if t1 is not None:
            self._profile_add("network_predict", time.perf_counter() - t1)
        self._predict_cache[key] = result
        if self._cache_max_size is not None and len(self._predict_cache) > self._cache_max_size:
            self._predict_cache.popitem(last=False)
        return result

    def _predict_batch(self, boards, players, last_moves_list):
        results = [None] * len(boards)
        miss_indices = []
        miss_states = []
        if not self.use_last_moves:
            last_moves_list = [None] * len(boards)

        if self.use_cache:
            for i, (board, player, last_moves) in enumerate(zip(boards, players, last_moves_list)):
                key = self._cache_key(board, player, last_moves)
                cached = self._predict_cache.get(key)
                if cached is not None:
                    self._predict_cache_hits += 1
                    if self._cache_max_size is not None:
                        self._predict_cache.move_to_end(key)
                    results[i] = cached
                else:
                    self._predict_cache_misses += 1
                    miss_indices.append(i)
                    miss_states.append((board, player, last_moves))
        else:
            miss_indices = list(range(len(boards)))
            miss_states = list(zip(boards, players, last_moves_list))

        if miss_indices:
            boards_miss = [s[0] for s in miss_states]
            players_miss = [s[1] for s in miss_states]
            last_moves_miss = [s[2] for s in miss_states]
            t0 = time.perf_counter() if self._profile_active else None
            encoded_batch = encode_board_batch(
                boards_miss, players_miss, self.env.board_size, last_moves_miss,
                use_liberty_features=self.use_liberty_features,
                liberty_bins=self.liberty_bins,
                use_last_moves=self.use_last_moves
            )
            if t0 is not None:
                self._profile_add("encode_batch", time.perf_counter() - t0)

            if hasattr(self.network, "predict_batch"):
                t1 = time.perf_counter() if self._profile_active else None
                policies, values = self.network.predict_batch(encoded_batch)
                if t1 is not None:
                    self._profile_add("network_predict_batch", time.perf_counter() - t1)
            else:
                policies = []
                values = []
                for state in encoded_batch:
                    t1 = time.perf_counter() if self._profile_active else None
                    p, v = self.network.predict(state)
                    if t1 is not None:
                        self._profile_add("network_predict", time.perf_counter() - t1)
                    policies.append(p)
                    values.append(v)
                policies = np.array(policies)
                values = np.array(values)

            for idx, policy, value in zip(miss_indices, policies, values):
                results[idx] = (policy, float(value))
                if self.use_cache:
                    board, player, last_moves = boards[idx], players[idx], last_moves_list[idx]
                    cache_key = self._cache_key(board, player, last_moves)
                    self._predict_cache[cache_key] = results[idx]
                    if self._cache_max_size is not None and len(self._predict_cache) > self._cache_max_size:
                        self._predict_cache.popitem(last=False)

        policies = np.array([r[0] for r in results])
        values = np.array([r[1] for r in results])
        return policies, values
        
        # 시뮬레이션용 환경 복사본
        self.sim_env = env.fast_copy()
    
    def _unpack_state(self, state):
        if len(state) == 4:
            board, player, passes, last_moves = state
        else:
            board, player, passes = state
            last_moves = None
        if not self.use_last_moves:
            last_moves = None
        return board, player, passes, last_moves

    def _set_env_state(self, env, state):
        """???????? ???"""
        board, player, passes, _ = self._unpack_state(state)
        if env.board is None or env.board.shape != board.shape:
            env.board = board.copy()
        else:
            np.copyto(env.board, board)
        env.current_player = player
        env.consecutive_passes = passes

    def _get_legal_actions(self, env, exclude_own_territory=True):
        """??? ?? ??? ??"""
        t0 = time.perf_counter() if self._profile_active else None
        if self.use_legal_cache:
            key = (env.board.tobytes(), int(env.current_player), bool(exclude_own_territory))
            cached = self._legal_cache.get(key)
            if cached is not None:
                if self._legal_cache_max_size is not None:
                    self._legal_cache.move_to_end(key)
                if t0 is not None:
                    self._profile_add("get_legal_actions", time.perf_counter() - t0)
                return cached

        mask = env.get_legal_moves()

        # MCTS ??: ??? ??? ?? ??
        if exclude_own_territory:
            my_territory = env._get_territory_mask(env.current_player)
            my_territory_flat = my_territory.flatten()
            mask[:-1] = mask[:-1] & ~my_territory_flat

        legal_actions = np.where(mask)[0]
        if self.use_legal_cache:
            self._legal_cache[key] = legal_actions
            if self._legal_cache_max_size is not None and len(self._legal_cache) > self._legal_cache_max_size:
                self._legal_cache.popitem(last=False)
        if t0 is not None:
            self._profile_add("get_legal_actions", time.perf_counter() - t0)
        return legal_actions
    
    def _q_value(self, parent, child):
        """Convert child value to the parent's perspective."""
        if child.player is None or parent.player is None:
            return -child.value()
        return child.value() if child.player == parent.player else -child.value()

    def _puct_score(self, parent, child, action):
        """
        PUCT (Polynomial Upper Confidence Trees) 점수 계산
        
        Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        - Q(s,a): 행동의 평균 가치
        - P(s,a): Prior 확률 (Policy Network에서)
        - N(s): 부모 방문 횟수
        - N(s,a): 자식 방문 횟수
        """
        q_value = self._q_value(parent, child)
        
        # UCB 탐색 항
        exploration = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        
        return q_value + exploration
    
    def _select_child(self, node, root=None, force_root=False):
        """PUCT ???e ?ê????? ?ì ?­??ì?ì? í?"""
        t0 = time.perf_counter() if self._profile_active else None
        if force_root and root is node and node.children:
            total_n = max(1, node.visit_count)
            forced_actions = []
            forced_children = []
            forced_scores = []
            for action, child in node.children.items():
                n_forced = self.forced_playout_k * child.prior * math.sqrt(total_n)
                if child.visit_count < n_forced:
                    forced_actions.append(action)
                    forced_children.append(child)
                    forced_scores.append(n_forced - child.visit_count)
            if forced_actions:
                pick = int(np.argmax(forced_scores))
                return forced_actions[pick], forced_children[pick]

        if self.use_vector_puct:
            result = self._select_child_vector(node)
        else:
            result = self._select_child_py(node)
        if t0 is not None:
            self._profile_add("select_child", time.perf_counter() - t0)
        return result

    def _select_child_py(self, node):
        best_score = -float('inf')
        best_actions = []
        best_children = []

        for action, child in node.children.items():
            score = self._puct_score(node, child, action)
            if score > best_score + 1e-12:
                best_score = score
                best_actions = [action]
                best_children = [child]
            elif abs(score - best_score) <= 1e-12:
                best_actions.append(action)
                best_children.append(child)

        if len(best_actions) == 1:
            return best_actions[0], best_children[0]
        pick = np.random.randint(len(best_actions))
        return best_actions[pick], best_children[pick]

    def _select_child_vector(self, node):
        actions = list(node.children.keys())
        children = [node.children[a] for a in actions]
        if not actions:
            return None, None

        visit_counts = np.array([child.visit_count for child in children], dtype=np.float32)
        priors = np.array([child.prior for child in children], dtype=np.float32)

        q_values = []
        for child in children:
            q_values.append(self._q_value(node, child))
        q_values = np.array(q_values, dtype=np.float32)

        exploration = self.c_puct * priors * math.sqrt(max(1, node.visit_count)) / (1.0 + visit_counts)
        scores = q_values + exploration

        best_score = scores.max()
        best_idx = np.flatnonzero(np.abs(scores - best_score) <= 1e-12)
        if len(best_idx) == 1:
            idx = int(best_idx[0])
            return actions[idx], children[idx]
        pick = int(np.random.randint(len(best_idx)))
        idx = int(best_idx[pick])
        return actions[idx], children[idx]


    def _expand(self, node, state, policy=None):
        """
        노드 확장: Policy Network로 prior 확률 설정
        
        Args:
            node: 확장할 노드
            state: (board, player, passes) 상태
        """
        node.state = state
        board, player, passes, last_moves = self._unpack_state(state)
        node.player = player
        
        # 환경 설정
        self._set_env_state(self.sim_env, state)
        
        # 유효한 행동 가져오기
        legal_actions = self._get_legal_actions(self.sim_env)
        
        if len(legal_actions) == 0:
            node.is_expanded = True
            return
        
        # 신경망으로 policy와 value 예측
        if policy is None:
            policy, _ = self._predict(board, player, last_moves)
        
        # 유효한 행동에 대해서만 prior 설정 (마스킹 후 재정규화)
        valid_policy = np.zeros_like(policy)
        valid_policy[legal_actions] = policy[legal_actions]
        
        # 재정규화 (합이 1이 되도록)
        policy_sum = valid_policy.sum()
        if policy_sum > 0:
            valid_policy /= policy_sum
        else:
            # 모든 확률이 0이면 균등 분포
            valid_policy[legal_actions] = 1.0 / len(legal_actions)
        
        # 자식 노드 생성
        for action in legal_actions:
            node.children[action] = AlphaZeroNode(prior=valid_policy[action])
        
        node.is_expanded = True
    
    def _simulate(self, node, state, root=None, force_root=False):
        """
        한 번의 시뮬레이션 (Selection -> Expansion -> Evaluation -> Backprop)
        
        Returns:
            value: 리프 노드의 가치 (현재 플레이어 관점)
        """
        path = [node]
        current_state = state
        _, current_player, _, last_moves = self._unpack_state(state)
        
        # === Selection ===
        while node.is_expanded and node.children:
            action, node = self._select_child(node, root=root, force_root=force_root)
            
            # 환경에서 행동 수행
            t_state = time.perf_counter() if self._profile_active else None
            self._set_env_state(self.sim_env, current_state)
            if t_state is not None:
                self._profile_add("set_env_state", time.perf_counter() - t_state)
            t_step = time.perf_counter() if self._profile_active else None
            _, reward, done, info = self.sim_env.step(action)
            if t_step is not None:
                self._profile_add("env_step", time.perf_counter() - t_step)
            
            next_player = self.sim_env.current_player
            if done:
                next_player = 2 if next_player == 1 else 1
            last_moves = self._update_last_moves(last_moves, action)
            current_state = (
                self.sim_env.board.copy(),
                next_player,
                self.sim_env.consecutive_passes,
                last_moves
            )
            node.state = current_state
            node.player = next_player
            
            path.append(node)
            
            # 게임 종료 체크
            if done:
                # 승패 결정
                if "winner" in info:
                    if info["winner"] == "Black":
                        winner = 1
                    elif info["winner"] == "White":
                        winner = 2
                    else:
                        winner = 0
                elif reward > 0:
                    winner = self.sim_env.current_player
                elif reward < 0:
                    winner = 2 if self.sim_env.current_player == 1 else 1
                else:
                    winner = 0

                _, leaf_player, _, _ = self._unpack_state(current_state)
                if winner == 0:
                    value = 0.0
                else:
                    value = 1.0 if winner == leaf_player else -1.0
                
                # Backpropagation
                t_bp = time.perf_counter() if self._profile_active else None
                self._backpropagate(path, value, current_player)
                if t_bp is not None:
                    self._profile_add("backprop", time.perf_counter() - t_bp)
                return value
        
        # === Expansion ===
        if not node.is_expanded:
            t_ex = time.perf_counter() if self._profile_active else None
            self._expand(node, current_state)
            if t_ex is not None:
                self._profile_add("expand", time.perf_counter() - t_ex)
        
        # === Evaluation ===
        # Value Network로 평가
        board, player, _, last_moves = self._unpack_state(current_state)
        t_pred = time.perf_counter() if self._profile_active else None
        _, value = self._predict(board, player, last_moves)
        if t_pred is not None:
            self._profile_add("predict", time.perf_counter() - t_pred)
        
        # value는 현재 플레이어(player) 관점
        
        # === Backpropagation ===
        t_bp = time.perf_counter() if self._profile_active else None
        self._backpropagate(path, value, current_player)
        if t_bp is not None:
            self._profile_add("backprop", time.perf_counter() - t_bp)
        
        return value

    def _simulate_batch(self, root, state, batch_size, force_root=False):
        eval_items = []

        for _ in range(batch_size):
            node = root
            current_state = state
            _, current_player, _, last_moves = self._unpack_state(state)
            path = [node]
            done = False

            while node.is_expanded and node.children:
                action, node = self._select_child(node, root=root, force_root=force_root)
                t_state = time.perf_counter() if self._profile_active else None
                self._set_env_state(self.sim_env, current_state)
                if t_state is not None:
                    self._profile_add("set_env_state", time.perf_counter() - t_state)
                t_step = time.perf_counter() if self._profile_active else None
                _, reward, done, info = self.sim_env.step(action)
                if t_step is not None:
                    self._profile_add("env_step", time.perf_counter() - t_step)

                next_player = self.sim_env.current_player
                if done:
                    next_player = 2 if next_player == 1 else 1
                last_moves = self._update_last_moves(last_moves, action)

                current_state = (
                    self.sim_env.board.copy(),
                    next_player,
                    self.sim_env.consecutive_passes,
                    last_moves
                )
                node.state = current_state
                node.player = next_player
                path.append(node)

                if done:
                    if "winner" in info:
                        if info["winner"] == "Black":
                            winner = 1
                        elif info["winner"] == "White":
                            winner = 2
                        else:
                            winner = 0
                    elif reward > 0:
                        winner = self.sim_env.current_player
                    elif reward < 0:
                        winner = 2 if self.sim_env.current_player == 1 else 1
                    else:
                        winner = 0

                    _, leaf_player, _, _ = self._unpack_state(current_state)
                    if winner == 0:
                        value = 0.0
                    else:
                        value = 1.0 if winner == leaf_player else -1.0

                    t_bp = time.perf_counter() if self._profile_active else None
                    self._backpropagate(path, value, current_player)
                    if t_bp is not None:
                        self._profile_add("backprop", time.perf_counter() - t_bp)
                    break

            if done:
                continue

            if not node.is_expanded:
                eval_items.append((node, current_state, path, current_player, False))
            else:
                eval_items.append((node, current_state, path, current_player, True))

        if not eval_items:
            return

        boards = [item[1][0] for item in eval_items]
        players = [item[1][1] for item in eval_items]
        last_moves_list = [item[1][3] if len(item[1]) > 3 else None for item in eval_items]
        t_pred = time.perf_counter() if self._profile_active else None
        policies, values = self._predict_batch(boards, players, last_moves_list)
        if t_pred is not None:
            self._profile_add("predict_batch", time.perf_counter() - t_pred)

        for (node, current_state, path, current_player, already_expanded), policy, value in zip(eval_items, policies, values):
            if not already_expanded:
                t_ex = time.perf_counter() if self._profile_active else None
                self._expand(node, current_state, policy=policy)
                if t_ex is not None:
                    self._profile_add("expand", time.perf_counter() - t_ex)
            t_bp = time.perf_counter() if self._profile_active else None
            self._backpropagate(path, float(value), current_player)
            if t_bp is not None:
                self._profile_add("backprop", time.perf_counter() - t_bp)

    def _counts_to_probs(self, counts, temperature):
        if temperature == 0:
            action = int(np.argmax(counts))
            action_probs = np.zeros_like(counts)
            action_probs[action] = 1.0
            return action, action_probs
        counts_temp = counts ** (1 / temperature)
        total = counts_temp.sum()
        if total > 0:
            action_probs = counts_temp / total
        else:
            action_probs = np.ones_like(counts) / len(counts)
        action = np.random.choice(len(action_probs), p=action_probs)
        return action, action_probs

    def _prune_policy_target(self, root, visit_counts):
        pruned = visit_counts.astype(np.float64).copy()
        if root is None or not root.children:
            return pruned

        total_n = max(1.0, float(np.sum(visit_counts)))
        actions = list(root.children.keys())
        if not actions:
            return pruned

        best_action = max(actions, key=lambda a: root.children[a].visit_count)
        best_child = root.children[best_action]
        best_puct = self._puct_score(root, best_child, best_action)

        for action, child in root.children.items():
            if action == best_action:
                continue
            n = child.visit_count
            if n <= 0:
                continue
            n_forced = self.forced_playout_k * child.prior * math.sqrt(total_n)
            if n_forced <= 0:
                continue

            q_val = self._q_value(root, child)
            denom = best_puct - q_val
            if denom <= 0:
                continue
            min_visits = self.c_puct * child.prior * math.sqrt(root.visit_count) / denom - 1.0
            min_visits = max(0.0, min_visits)

            target = max(min_visits, n - n_forced)
            pruned_count = max(0.0, min(n, target))
            if pruned_count <= 1.0:
                pruned[action] = 0.0
            else:
                pruned[action] = pruned_count

        return pruned
    
    def _backpropagate(self, path, value, root_player):
        """
        가치를 트리 위로 역전파
        
        Args:
            path: 방문한 노드들의 리스트
            value: leaf 플레이어 관점의 가치
            root_player: root 노드의 플레이어
        """
        for node in reversed(path):
            node.visit_count += 1
            # 각 노드는 해당 노드의 플레이어 관점에서 가치 저장
            # 한 단계씩 올라가며 관점을 반전
            node.value_sum += value
            value = -value
    
    def _add_dirichlet_noise(self, node):
        """Add Dirichlet noise to root priors to encourage exploration."""
        if not node.children:
            return
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = node.children[action]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * n

    def run(self, state, temperature=1.0, add_root_noise=False):
        """
        MCTS 실행하여 최적의 행동 반환
        
        Args:
            state: (board, player, passes) 상태 튜플
            temperature: 행동 선택 온도
                - 0: 가장 많이 방문한 행동 (결정적)
                - 1: 방문 횟수에 비례한 확률적 선택
        
        Returns:
            action: 선택된 행동 인덱스
            action_probs: 각 행동의 선택 확률 (학습용)
        """
        if self._profile_enabled:
            self._profile_runs += 1
            self._profile_active = (self._profile_every is None) or (self._profile_runs % self._profile_every == 0)
            if self._profile_active:
                self._profile_reset()
        t_run = time.perf_counter() if self._profile_active else None
        root = AlphaZeroNode(prior=0.0)
        self._expand(root, state)
        if add_root_noise:
            self._add_dirichlet_noise(root)
        force_root = self.use_forced_playouts and add_root_noise
        
        # 시뮬레이션 실행
        if self.eval_batch_size <= 1:
            for _ in range(self.num_simulations):
                self._simulate(root, state, root=root, force_root=force_root)
        else:
            sims_done = 0
            while sims_done < self.num_simulations:
                batch_n = min(self.eval_batch_size, self.num_simulations - sims_done)
                self._simulate_batch(root, state, batch_n, force_root=force_root)
                sims_done += batch_n
        
        # 방문 횟수 기반 행동 선택
        visit_counts = np.zeros(self.env.action_space_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # 행동 선택은 raw 분포로, 학습용 정책은 pruning 적용
        action, _ = self._counts_to_probs(visit_counts, temperature)
        policy_counts = visit_counts
        if self.use_policy_target_pruning and add_root_noise:
            policy_counts = self._prune_policy_target(root, visit_counts)
        _, action_probs = self._counts_to_probs(policy_counts, temperature)
        
        if t_run is not None:
            self._profile_report(time.perf_counter() - t_run)
        return action, action_probs
    

    def run_with_info(self, state, temperature=1.0, add_root_noise=False, top_k=5):
        """run() 와 같지만 MCTS 디버그 메타데이터를 함께 반환"""
        if self._profile_enabled:
            self._profile_runs += 1
            self._profile_active = (self._profile_every is None) or (self._profile_runs % self._profile_every == 0)
            if self._profile_active:
                self._profile_reset()
        t_run = time.perf_counter() if self._profile_active else None
        root = AlphaZeroNode(prior=0.0)
        self._expand(root, state)

        priors_before = {a: c.prior for a, c in root.children.items()}
        if add_root_noise:
            self._add_dirichlet_noise(root)
        priors_after = {a: c.prior for a, c in root.children.items()}
        force_root = self.use_forced_playouts and add_root_noise

        if self.eval_batch_size <= 1:
            for _ in range(self.num_simulations):
                self._simulate(root, state, root=root, force_root=force_root)
        else:
            sims_done = 0
            while sims_done < self.num_simulations:
                batch_n = min(self.eval_batch_size, self.num_simulations - sims_done)
                self._simulate_batch(root, state, batch_n, force_root=force_root)
                sims_done += batch_n

        visit_counts = np.zeros(self.env.action_space_size)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        action, _ = self._counts_to_probs(visit_counts, temperature)
        policy_counts = visit_counts
        if self.use_policy_target_pruning and add_root_noise:
            policy_counts = self._prune_policy_target(root, visit_counts)
        _, action_probs = self._counts_to_probs(policy_counts, temperature)

        self._set_env_state(self.sim_env, state)
        legal_mask_all = self.sim_env.get_legal_moves()
        legal_all = np.where(legal_mask_all)[0]
        legal_mcts = np.array(sorted(root.children.keys()), dtype=int)
        excluded = np.setdiff1d(legal_all, legal_mcts, assume_unique=False)

        def _top_actions(values_dict, k):
            items = sorted(values_dict.items(), key=lambda x: x[1], reverse=True)
            return items[:k]

        visit_dict = {a: float(visit_counts[a]) for a in root.children.keys()}

        info = {
            'legal_total': int(len(legal_all)),
            'legal_mcts': int(len(legal_mcts)),
            'excluded_by_own_territory': int(len(excluded)),
            'visited_actions': int(np.sum(visit_counts > 0)),
            'root_visit_sum': int(np.sum(visit_counts)),
            'top_priors_before': _top_actions(priors_before, top_k),
            'top_priors_after': _top_actions(priors_after, top_k),
            'top_visits': _top_actions(visit_dict, top_k),
        }

        if t_run is not None:
            self._profile_report(time.perf_counter() - t_run)
        return action, action_probs, info

    def get_action_probs(self, state, temperature=1.0):
        """행동 확률만 반환 (학습 데이터 생성용)"""
        _, action_probs = self.run(state, temperature)
        return action_probs


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    
    from env.env import GreatKingdomEnv
    from network import AlphaZeroNetwork
    
    print("=== AlphaZero MCTS Test ===")
    
    # 환경 및 네트워크 생성
    env = GreatKingdomEnv(board_size=5)
    network = AlphaZeroNetwork(board_size=5, num_res_blocks=3, num_channels=64)
    
    # MCTS 생성
    mcts = AlphaZeroMCTS(network, env, num_simulations=50)
    
    # 테스트 게임
    env.reset()
    print("\n초기 보드:")
    env.render()
    
    # 첫 수 선택
    state = (env.board.copy(), env.current_player, env.consecutive_passes)
    action, probs = mcts.run(state, temperature=1.0)
    
    print(f"\n선택된 행동: {action}")
    if action == env.pass_action:
        print("  -> PASS")
    else:
        r, c = divmod(action, env.board_size)
        print(f"  -> ({r}, {c})")
    
    # 상위 5개 행동 확률
    top_5 = np.argsort(probs)[-5:][::-1]
    print("\n상위 5개 행동 확률:")
    for a in top_5:
        if a == env.pass_action:
            print(f"  PASS: {probs[a]:.4f}")
        else:
            r, c = divmod(a, env.board_size)
            print(f"  ({r},{c}): {probs[a]:.4f}")
    
    print("\n=== Test Passed! ===")
