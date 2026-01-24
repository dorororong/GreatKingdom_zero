import numpy as np
from scipy.ndimage import label, binary_dilation
from collections import deque

class GreatKingdomEnv:
    def __init__(self, board_size=5, center_wall=True, komi=0):
        self.board_size = board_size
        self.center_wall = center_wall
        self.komi = komi
        # 액션 공간: board_size^2 (착수) + 1 (패스)
        self.action_space_size = board_size * board_size + 1
        self.pass_action = board_size * board_size  # 패스 액션 인덱스
        
        # 보드 표현: 0=빈칸, 1=흑, 2=백, 3=벽(중립)
        self.board = None
        self.current_player = 1 # 1: 흑, 2: 백
        
        # 연속 패스 카운터 (2회 연속 패스 시 게임 종료)
        self.consecutive_passes = 0
        
        # scipy 연산용 구조체 (십자 모양 연결)
        self.structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        
        self.reset()

    def reset(self):
        """게임을 초기화합니다."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.consecutive_passes = 0
        
        # 규칙 4: 중앙에 중립 기물(벽) 배치
        if self.center_wall:
            mid = self.board_size // 2
            self.board[mid, mid] = 3
        
        return self.get_observation()

    def get_observation(self):
        """현재 보드 상태와 플레이어 정보를 반환합니다."""
        return self.board.copy()

    def _get_territory_mask(self, player):
        board_size = self.board_size
        board = self.board
        opp_player = 2 if player == 1 else 1
        
        # 결과 마스크
        territory_mask = np.zeros((board_size, board_size), dtype=bool)
        
        # 방문 체크 (전역)
        visited = np.zeros((board_size, board_size), dtype=bool)
        
        # 4방향 델타
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 보드 전체 순회
        for r in range(board_size):
            for c in range(board_size):
                # 빈칸이 아니거나 이미 방문했으면 스킵
                if board[r, c] != 0 or visited[r, c]:
                    continue
                
                # --- 새로운 빈 공간 그룹(Region) 발견: BFS 시작 ---
                region_cells = []  # 현재 영역에 포함된 좌표들
                queue = deque([(r, c)])
                visited[r, c] = True
                region_cells.append((r, c))
                
                # 영역 판단을 위한 플래그들
                my_stones_found = False     # 내 돌 접촉 여부
                opp_stones_found = False    # 상대 돌 접촉 여부
                
                edge_walls_hit = set()      # 접촉한 보드 가장자리 ('top', 'left' 등)
                neutral_walls_hit = set()   # 접촉한 중립 벽(3)의 좌표 (중복 카운트 방지)
                
                while queue:
                    curr_r, curr_c = queue.popleft()
                    
                    for dr, dc in deltas:
                        nr, nc = curr_r + dr, curr_c + dc
                        
                        # 1. 보드 밖 (가장자리 벽) 체크
                        if not (0 <= nr < board_size and 0 <= nc < board_size):
                            if nr < 0: edge_walls_hit.add('top')
                            elif nr >= board_size: edge_walls_hit.add('bottom')
                            if nc < 0: edge_walls_hit.add('left')
                            elif nc >= board_size: edge_walls_hit.add('right')
                            continue
                        
                        val = board[nr, nc]
                        
                        # 2. 빈칸 (같은 영역 확장)
                        if val == 0:
                            if not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                                region_cells.append((nr, nc))
                        
                        # 3. 내 돌
                        elif val == player:
                            my_stones_found = True
                            
                        # 4. 상대 돌
                        elif val == opp_player:
                            opp_stones_found = True
                            
                        # 5. 중립 벽 (3)
                        elif val == 3:
                            # Scipy의 binary_dilation은 '벽 픽셀의 개수'를 셉니다.
                            # 따라서 좌표를 집합에 넣어 고유한 벽 개수만 셉니다.
                            neutral_walls_hit.add((nr, nc))

                # --- BFS 종료 후 영역 판정 (기존 규칙 적용) ---
                
                # 조건 1: 내 돌이 하나라도 접촉해야 함
                if not my_stones_found:
                    continue
                    
                # 조건 2: 상대 돌이 하나라도 있으면 안 됨
                if opp_stones_found:
                    continue
                    
                # 조건 3: 벽 개수 계산 (가장자리 면 수 + 중립 벽 돌 개수)
                total_walls = len(edge_walls_hit) + len(neutral_walls_hit)
                if total_walls >= 3:
                    continue
                    
                # 조건 4: (암묵적) 여기까지 왔으면 '둘러싸인' 것임
                # 모든 조건 통과 -> 현재 영역을 결과 마스크에 True로 설정
                for tr, tc in region_cells:
                    territory_mask[tr, tc] = True
                    
        return territory_mask

    def get_legal_moves(self):
        """
        ????? ????????????? ????????? ?????? ????????1, ????? ?????? 0????? ?????? (Flatten)
        ?????????????????? ????? ????? (???? ?????
        """
        # 1. ??????: ??????????????
        legal_mask = (self.board == 0)

        # 2. ????????????????? ?????? ?????
        opp_player = 2 if self.current_player == 1 else 1
        opp_territory = self._get_territory_mask(opp_player)
        legal_mask[opp_territory] = False

        # 3. ???????????? (Win by Capture??? ?????)
        candidates = np.argwhere(legal_mask)
        for x, y in candidates:
            self.board[x, y] = self.current_player

            # capture check
            opp_stones = (self.board == opp_player)
            capture_flag = False
            if np.any(opp_stones):
                labeled_opp, num_opp = label(opp_stones, structure=self.structure)
                for i in range(1, num_opp + 1):
                    group = (labeled_opp == i)
                    dilated = binary_dilation(group, structure=self.structure)
                    liberties = np.logical_and(dilated, self.board == 0)
                    if np.sum(liberties) == 0:
                        capture_flag = True
                        break

            if not capture_flag:
                my_stones = (self.board == self.current_player)
                labeled_my, num_my = label(my_stones, structure=self.structure)
                suicide = False
                for i in range(1, num_my + 1):
                    group = (labeled_my == i)
                    if group[x, y]:
                        dilated = binary_dilation(group, structure=self.structure)
                        liberties = np.logical_and(dilated, self.board == 0)
                        if np.sum(liberties) == 0:
                            suicide = True
                        break
                if suicide:
                    legal_mask[x, y] = False

            self.board[x, y] = 0

        # ????? ????? ????? (???? ?????
        legal_flat = legal_mask.flatten()
        legal_with_pass = np.append(legal_flat, True)  # ??????????? ????? ?????

        return legal_with_pass

    def step(self, action_idx):
            """
            행동을 수행하고 상태를 업데이트합니다.
            return: (obs, reward, done, info)
            reward: 승리 +1, 패배 -1, 진행중 0
            """
            opp_player = 2 if self.current_player == 1 else 1
            
            # 패스 액션 처리
            if action_idx == self.pass_action:
                self.consecutive_passes += 1
                
                # 양쪽 모두 패스하면 게임 종료, 영토 계산
                if self.consecutive_passes >= 2:
                    return self._end_game_by_territory()
                
                # 턴 넘기기
                self.current_player = opp_player
                return self.board, 0, False, {"action": "pass"}
            
            # 일반 착수 - 연속 패스 카운터 초기화
            self.consecutive_passes = 0
            
            x, y = divmod(action_idx, self.board_size)
            
            # 유효성 검사 (이미 있는 곳이나 금지 구역)
            if self.board[x, y] != 0:
                return self.board, -1, True, {"error": "Invalid Move: Occupied"}
            
            # 영토(착수 금지 구역) 체크
            if self._get_territory_mask(opp_player)[x, y]:
                 return self.board, -1, True, {"error": "Invalid Move: Opponent Territory"}
    
            # 1. 착수
            self.board[x, y] = self.current_player
            
            # 2. 승리 조건 체크 (상대방 돌을 잡았는가?)
            # 상대방 돌들의 활로 계산
            opp_stones = (self.board == opp_player)
            if np.any(opp_stones):
                labeled_opp, num_opp = label(opp_stones, structure=self.structure)
                capture_flag = False
                
                for i in range(1, num_opp + 1):
                    group = (labeled_opp == i)
                    dilated = binary_dilation(group, structure=self.structure)
                    # 활로(빈칸)가 하나도 없으면 잡힘
                    liberties = np.logical_and(dilated, self.board == 0)
                    if np.sum(liberties) == 0:
                        capture_flag = True
                        break
                
                if capture_flag:
                    return self.board, 1, True, {"result": "Win by Capture"}
    
            # 3. 자살수 체크 (내가 뒀는데 내 활로가 0인가?)
            # 상대를 못 잡았는데 내가 죽으면 -> 반칙패 또는 무효. 여기선 반칙패 처리
            my_stones = (self.board == self.current_player)
            labeled_my, num_my = label(my_stones, structure=self.structure)
            for i in range(1, num_my + 1):
                group = (labeled_my == i)
                # 방금 둔 돌이 포함된 그룹만 확인하면 더 빠름
                if group[x, y]:
                    dilated = binary_dilation(group, structure=self.structure)
                    liberties = np.logical_and(dilated, self.board == 0)
                    if np.sum(liberties) == 0:
                        # 자살수 -> 패배 처리
                        return self.board, -1, True, {"result": "Loss by Suicide"}
    
            # 4. 게임 종료 여부 (더 이상 둘 곳이 없는지)
            # 간단히 보드가 꽉 찼는지만 체크 (더 엄밀히는 양쪽 다 pass해야 함)
            if np.sum(self.board == 0) == 0:
                 # 집 계산 후 승패 판정 (간략화: 무승부 처리)
                 return self.board, 0, True, {"result": "Draw (Board Full)"}
                 
            # 5. 턴 넘기기
            self.current_player = opp_player
            return self.board, 0, False, {}
    
    def _end_game_by_territory(self):
        """
        양쪽 패스로 게임 종료 시 영토 크기로 승패 결정
        return: (obs, reward, done, info)
        """
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
            # 흑 승리
            # 현재 플레이어 기준으로 reward 결정
            reward = 1 if self.current_player == 1 else -1
            info["winner"] = "Black"
        elif black_score < white_score + self.komi:
            # 백 승리
            reward = 1 if self.current_player == 2 else -1
            info["winner"] = "White"
        else:
            # 무승부
            reward = 0
            info["winner"] = "Draw"
        
        return self.board, reward, True, info
    
    def get_territory_scores(self):
        """현재 양측 영토 점수를 반환합니다."""
        black_territory = np.sum(self._get_territory_mask(1))
        white_territory = np.sum(self._get_territory_mask(2))
        return {"black": int(black_territory), "white": int(white_territory)}
    
    def get_kill_moves(self, player):
        """
        해당 플레이어가 두어서 상대방 돌을 잡을 수 있는 모든 수(킬 스위치)를 반환합니다.
        상대방 그룹 중 활로가 1개인 그룹의 그 활로 위치를 찾습니다.
        [scipy 버전 - label과 binary_dilation 사용]
        """
        opp_player = 2 if player == 1 else 1
        opp_stones = (self.board == opp_player)
        
        kill_moves = []
        
        # 상대방 돌이 없으면 빈 리스트 반환
        if not np.any(opp_stones):
            return kill_moves
            
        labeled_opp, num_opp = label(opp_stones, structure=self.structure)
        
        for i in range(1, num_opp + 1):
            group = (labeled_opp == i)
            dilated = binary_dilation(group, structure=self.structure)
            # 활로(빈칸) 마스크
            liberties_mask = np.logical_and(dilated, self.board == 0)
            num_liberties = np.sum(liberties_mask)
            
            if num_liberties == 1:
                # 활로가 딱 하나라면 그곳이 킬 스위치
                # np.where는 (row_array, col_array)를 반환
                rows, cols = np.where(liberties_mask)
                r, c = rows[0], cols[0]
                action = r * self.board_size + c
                
                # 유효한 수인지 확인 (자살수 금지 등 규칙 위반 여부)
                # 여기서는 간단히 빈칸인지만 확인했으므로(liberties_mask 생성 시),
                # get_legal_moves()를 통해 최종 검증하는 것이 안전함.
                # 하지만 성능을 위해 호출 측에서 검증하도록 인덱스만 반환.
                kill_moves.append(action)
                
        return kill_moves

    def get_kill_moves_bfs(self, player):
        """
        해당 플레이어가 두어서 상대방 돌을 잡을 수 있는 모든 수(킬 스위치)를 반환합니다.
        [순수 BFS 버전 - scipy 미사용]
        """
        opp_player = 2 if player == 1 else 1
        board_size = self.board_size
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상하좌우
        
        kill_moves = []
        visited_groups = set()
        
        for start_r in range(board_size):
            for start_c in range(board_size):
                # 상대방 돌이 아니거나 이미 방문한 그룹이면 스킵
                if self.board[start_r, start_c] != opp_player:
                    continue
                if (start_r, start_c) in visited_groups:
                    continue
                
                # BFS로 연결된 그룹과 활로 찾기
                group = set()
                liberties = set()
                queue = deque([(start_r, start_c)])
                group.add((start_r, start_c))
                
                while queue:
                    r, c = queue.popleft()
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        
                        if 0 <= nr < board_size and 0 <= nc < board_size:
                            if self.board[nr, nc] == opp_player and (nr, nc) not in group:
                                group.add((nr, nc))
                                queue.append((nr, nc))
                            elif self.board[nr, nc] == 0:
                                liberties.add((nr, nc))
                
                # 방문 처리
                visited_groups.update(group)
                
                # 활로가 1개인 경우 킬 스위치
                if len(liberties) == 1:
                    kill_r, kill_c = liberties.pop()
                    action = kill_r * board_size + kill_c
                    kill_moves.append(action)
        
        return kill_moves

    def fast_copy(self):
        """
        deepcopy보다 훨씬 빠른 경량 복제 메서드입니다.
        빈 인스턴스를 생성하고 핵심 데이터만 복사하여 할당합니다.
        """
        # 1. 클래스의 새 인스턴스를 생성하되, __init__은 호출하지 않음 (속도 최적화)
        cls = self.__class__
        new_env = cls.__new__(cls)
        
        # 2. 변경 불가능한 값(Immutable)이나 단순 값은 얕은 복사(할당)
        new_env.board_size = self.board_size
        new_env.center_wall = self.center_wall
        new_env.komi = self.komi
        new_env.action_space_size = self.action_space_size
        new_env.pass_action = self.pass_action
        new_env.current_player = self.current_player
        new_env.consecutive_passes = self.consecutive_passes
        
        # 3. 변경 가능한 객체(Numpy Array)는 반드시 복사(Copy)해야 함
        # 구조체(structure)는 읽기 전용이므로 참조만 해도 무방
        new_env.board = self.board.copy() 
        new_env.structure = self.structure 
        
        return new_env

    def render(self):
        """터미널에 보드를 예쁘게 출력합니다."""
        symbols = {0: '.', 1: '●', 2: '○', 3: '■'}
        print("  " + " ".join([str(i) for i in range(self.board_size)]))
        for i in range(self.board_size):
            row_str = f"{i} "
            for j in range(self.board_size):
                row_str += symbols[self.board[i, j]] + " "
            print(row_str)
        scores = self.get_territory_scores()
        print(f"Current Turn: {'Black (●)' if self.current_player == 1 else 'White (○)'}")
        print(f"Territory - Black: {scores['black']}, White: {scores['white']}")
        print("-" * 20)
