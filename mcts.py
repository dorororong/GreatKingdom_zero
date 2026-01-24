import numpy as np
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, action=None, action_player=None):
        """
        state: (board, current_player, consecutive_passes) 튜플
        action: 이 노드에 도달하기 위해 수행한 행동
        action_player: 이 노드로 오는 액션을 취한 플레이어 (부모의 current_player)
        """
        self.state = state
        self.parent = parent
        self.action = action
        self.action_player = action_player  # 이 수를 둔 플레이어
        
        self.children = []
        self.visits = 0
        self.wins = 0  # action_player가 이긴 횟수
        
        self.untried_actions = None 

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        """UCB1 공식을 사용하여 가장 유망한 자식 노드를 선택
        
        child.wins는 "이 수를 둔 플레이어(=부모의 current_player)"가 이긴 횟수
        부모가 자식을 선택할 때, 부모의 current_player가 이긴 자식을 선택해야 함
        """
        choices_weights = []
        for child in self.children:
            # child.wins는 부모의 current_player가 이긴 횟수
            # 부모 입장에서 승률이 높은 자식을 선택
            exploitation = child.wins / child.visits
            exploration = c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            choices_weights.append(exploitation + exploration)
        return self.children[np.argmax(choices_weights)]

class MCTS:
    def __init__(self, env, simulations_per_move=1000, use_scipy_kill=True):
        self.env = env  # 원본 환경 (규칙 참조용)
        self.sim_env = env.fast_copy() if hasattr(env, 'fast_copy') else self._manual_copy(env) # 시뮬레이션용 복사본
        self.simulations_per_move = simulations_per_move
        self.last_opponent_action = None  # 상대방의 마지막 수 기록
        self.use_scipy_kill = use_scipy_kill  # True: scipy 버전, False: BFS 버전

    def _manual_copy(self, env):
        """fast_copy가 없는 경우를 위한 폴백(fallback)"""
        import copy
        # 전체 deepcopy 대신 껍데기만 만들고 필요한 것만 복사 (앞선 대화의 최적화 적용)
        new_env = copy.copy(env) 
        new_env.board = env.board.copy()
        return new_env

    def _set_env_state(self, env, state_snapshot):
        """
        환경에 저장된 상태 튜플을 주입합니다. (가장 중요한 최적화 부분)
        state_snapshot: (board, player, passes[, last_moves])
        """
        board, player, passes = state_snapshot[:3]
        if env.board is None or env.board.shape != board.shape:
            env.board = board.copy()
        else:
            np.copyto(env.board, board)
        env.current_player = player
        env.consecutive_passes = passes

    def get_legal_actions(self, env, exclude_own_territory=True):
        """환경에서 유효한 행동 인덱스 리스트를 반환
        
        Args:
            env: 게임 환경
            exclude_own_territory: True면 자신의 영토에도 착수하지 않음 (MCTS 전략)
        """
        mask = env.get_legal_moves()  # 마지막은 PASS
        
        # MCTS 전략: 자신의 영토에 착수하면 집을 줄이는 비효율적 행동이므로 제외
        if exclude_own_territory:
            my_territory = env._get_territory_mask(env.current_player)
            my_territory_flat = my_territory.flatten()
            # 보드 위치만 마스킹 (패스 액션은 제외)
            mask[:-1] = mask[:-1] & ~my_territory_flat
        
        return np.where(mask)[0]

    def run(self, current_state_snapshot, opponent_last_action=None):
        """MCTS 메인 루프"""
        board, current_player, _ = current_state_snapshot[:3]
        
        # === Decisive Move Check (킬 스위치 확인) ===
        # 환경(env)에 구현된 get_kill_moves 사용
        # 현재 상태를 시뮬레이션 환경에 설정
        self._set_env_state(self.sim_env, current_state_snapshot)
        
        # 킬 스위치 탐색 (상대방 돌을 잡을 수 있는 수)
        if self.use_scipy_kill:
            kill_moves = self.sim_env.get_kill_moves(current_player)
        else:
            kill_moves = self.sim_env.get_kill_moves_bfs(current_player)
        
        if kill_moves:
            # 유효한 수인지 확인
            legal_moves = self.sim_env.get_legal_moves()
            for km in kill_moves:
                if legal_moves[km]:
                    return km
        
        # === 킬 스위치가 없으면 기존 MCTS 실행 ===
        root = MCTSNode(state=current_state_snapshot)
        
        for _ in range(self.simulations_per_move):
            node = root
            
            # 1. Selection (선택)
            # 트리를 따라 내려가며 확장이 필요한 노드(Leaf에 가까운)를 찾음
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            
            # 2. Expansion (확장)
            # 아직 해보지 않은 행동이 있다면 자식 노드를 생성
            if node.untried_actions is None:
                # 노드에 처음 도달했을 때 유효 행동 목록 계산
                self._set_env_state(self.sim_env, node.state)
                node.untried_actions = list(self.get_legal_actions(self.sim_env))
            
            if node.untried_actions:
                action = node.untried_actions.pop()
                
                # 액션을 취하는 플레이어 기록
                self._set_env_state(self.sim_env, node.state)
                action_player = self.sim_env.current_player  # 이 수를 둔 플레이어
                
                obs, reward, done, _ = self.sim_env.step(action)
                
                # 새 상태 스냅샷 생성
                new_state = (self.sim_env.board.copy(), self.sim_env.current_player, self.sim_env.consecutive_passes)
                child_node = MCTSNode(state=new_state, parent=node, action=action, action_player=action_player)
                node.children.append(child_node)
                node = child_node # 시뮬레이션은 이 자식 노드에서 시작
            
            # 3. Simulation (시뮬레이션/롤아웃)
            # 무작위로 게임 끝까지 진행
            self._set_env_state(self.sim_env, node.state)
            is_done = False
            winner = 0
            winner_map = {"Black": 1, "White": 2, "Draw": 0}
            
            # 롤아웃 시작
            temp_steps = 0
            while not is_done and temp_steps < 100:
                action_player = self.sim_env.current_player  # 이번 턴 플레이어
                legal_moves = self.get_legal_actions(self.sim_env)
                if len(legal_moves) == 0: 
                    break
                
                random_action = np.random.choice(legal_moves)
                _, r, is_done, info = self.sim_env.step(random_action)
                
                if is_done:
                    # 승자 판정: info > reward > 무승부
                    if "winner" in info:
                        winner = winner_map.get(info["winner"], 0)
                    elif r != 0:
                        opp_player = 2 if action_player == 1 else 1
                        winner = action_player if r > 0 else opp_player
                    else:
                        winner = 0
                    break
                temp_steps += 1

            # 4. Backpropagation (역전파)
            # 각 노드는 action_player (이 노드로 이동시킨 플레이어) 관점에서 wins를 저장
            while node is not None:
                node.visits += 1
                # action_player가 승리했으면 wins 증가 (루트 노드는 action_player가 None)
                if node.action_player is not None:
                    if winner == node.action_player:
                        node.wins += 1
                    elif winner == 0:
                        node.wins += 0.5  # 무승부는 0.5점
                    # winner가 상대방이면 wins 증가 없음
                node = node.parent

        # 가장 많이 방문한 자식 노드를 최적의 수로 선택 (Robust Child)
        return max(root.children, key=lambda c: c.visits).action
