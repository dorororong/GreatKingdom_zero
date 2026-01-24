import numpy as np
from env.env import GreatKingdomEnv
from mcts import MCTS
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


# ========== 멀티프로세싱용 전역 함수 ==========

def _init_game_worker(board_size, mcts_simulations, use_scipy_kill, center_wall):
    """워커 프로세스 초기화"""
    global _worker_env, _worker_mcts
    _worker_env = GreatKingdomEnv(board_size=board_size, center_wall=center_wall)
    _worker_mcts = MCTS(_worker_env, simulations_per_move=mcts_simulations, use_scipy_kill=use_scipy_kill)


def _play_single_game(args):
    """워커에서 한 게임 수행 (멀티프로세싱용)"""
    mcts_is_black = args  # True면 MCTS가 흑, False면 MCTS가 백
    
    global _worker_env, _worker_mcts
    
    _worker_env.reset()
    # MCTS의 sim_env도 리셋
    _worker_mcts.sim_env.reset()
    
    done = False
    moves = 0
    max_moves = _worker_env.board_size * _worker_env.board_size * 3
    last_action = None
    
    while not done and moves < max_moves:
        current_player_before_step = _worker_env.current_player
        
        # MCTS 또는 랜덤 선택
        if (mcts_is_black and _worker_env.current_player == 1) or \
           (not mcts_is_black and _worker_env.current_player == 2):
            # MCTS 플레이어
            current_state = (
                _worker_env.board.copy(),
                _worker_env.current_player,
                _worker_env.consecutive_passes
            )
            action = _worker_mcts.run(current_state, opponent_last_action=last_action)
        else:
            # 랜덤 플레이어
            legal_moves = _worker_env.get_legal_moves()
            possible_indices = np.where(legal_moves == 1)[0]
            if len(possible_indices) == 0:
                return "Draw", moves, mcts_is_black
            non_pass_moves = possible_indices[possible_indices != _worker_env.pass_action]
            if len(non_pass_moves) > 0 and np.random.random() > 0.1:
                action = np.random.choice(non_pass_moves)
            else:
                action = np.random.choice(possible_indices)
        
        if action is None:
            return "Draw", moves, mcts_is_black
        
        _, reward, done, info = _worker_env.step(action)
        moves += 1
        last_action = action
    
    if not done:
        return "Draw", moves, mcts_is_black
    
    # 결과 판정
    if "winner" in info:
        winner = info["winner"]
    elif reward != 0:
        winner = "Black" if current_player_before_step == 1 else "White"
    else:
        winner = "Draw"
    
    return winner, moves, mcts_is_black


def random_player(env, opponent_last_action=None):
    """랜덤으로 행동을 선택하는 플레이어"""
    legal_moves = env.get_legal_moves()
    possible_indices = np.where(legal_moves == 1)[0]
    
    if len(possible_indices) == 0:
        return None
    
    # 패스가 아닌 수가 있으면 패스 확률 낮추기
    non_pass_moves = possible_indices[possible_indices != env.pass_action]
    if len(non_pass_moves) > 0 and np.random.random() > 0.1:
        return np.random.choice(non_pass_moves)
    else:
        return np.random.choice(possible_indices)


def mcts_player(env, mcts_agent, opponent_last_action=None):
    """MCTS로 행동을 선택하는 플레이어"""
    current_state = (
        env.board.copy(),
        env.current_player,
        env.consecutive_passes
    )
    return mcts_agent.run(current_state, opponent_last_action=opponent_last_action)


def play_match(env, player1_func, player2_func, player1_name="Player1", player2_name="Player2", verbose=False, render=False):
    """두 플레이어 간의 한 판 게임"""
    obs = env.reset()
    done = False
    moves = 0
    max_moves = env.board_size * env.board_size * 3
    last_action = None  # 상대방의 마지막 수 기록
    
    if verbose:
        print(f"\n=== {player1_name} (Black) vs {player2_name} (White) ===")
        if render:
            env.render()
    
    while not done and moves < max_moves:
        # 현재 플레이어에 따라 선택
        current_player_before_step = env.current_player
        if env.current_player == 1:  # Black
            action = player1_func(env, last_action)
        else:  # White
            action = player2_func(env, last_action)
        
        if action is None:
            return "Draw", moves, "No Legal Moves"
        
        if verbose:
            player_name = "Black" if env.current_player == 1 else "White"
            if action == env.pass_action:
                print(f"Move {moves+1}: {player_name} passes")
            else:
                r, c = divmod(action, env.board_size)
                print(f"Move {moves+1}: {player_name} plays at ({r}, {c})")
        
        obs, reward, done, info = env.step(action)
        moves += 1
        last_action = action  # 이번 수를 기록 (다음 턴의 상대방이 참조)
        
        if verbose and render:
            env.render()
    
    if not done:
        return "Draw", moves, "Move Limit Exceeded"
    
    # 결과 반환
    if "winner" in info:
        winner = info["winner"]
        result_detail = f"B:{info['black_territory']} vs W:{info['white_territory']}"
        if verbose:
            print(f"\n{'='*50}")
            print(f"Game Over! {winner} Wins! ({result_detail})")
            print(f"{'='*50}")
        return winner, moves, result_detail
    else:
        # Capture 등으로 승리한 경우 (info에 winner가 없지만 reward가 있음)
        result_detail = info.get('result', 'Unknown')
        if reward != 0:
            # reward가 1이면 방금 수를 둔 플레이어(current_player_before_step)가 승리
            winner = "Black" if current_player_before_step == 1 else "White"
            if verbose:
                print(f"\n{'='*50}")
                print(f"Game Over! {winner} Wins! ({result_detail})")
                print(f"{'='*50}")
            return winner, moves, result_detail
        else:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Game Over! Draw - {result_detail}")
                print(f"{'='*50}")
            return "Draw", moves, result_detail


def run_tournament(num_games=100, mcts_simulations=50, board_size=5, verbose=False, render=False, use_scipy_kill=True, num_workers=10, center_wall=False):
    """랜덤 vs MCTS 토너먼트 실행 (멀티프로세싱 지원)"""
    method_name = "scipy" if use_scipy_kill else "BFS"
    print(f"{'='*60}")
    print(f"Tournament: Random vs MCTS ({mcts_simulations} sims, kill_method={method_name})")
    print(f"Board Size: {board_size}x{board_size}, Games: {num_games}, Workers: {num_workers}")
    print(f"{'='*60}\n")
    
    # 통계 초기화
    stats = {
        "random_black_wins": 0,
        "random_white_wins": 0,
        "mcts_black_wins": 0,
        "mcts_white_wins": 0,
        "draws": 0,
        "total_moves": 0
    }
    
    start_time = time.time()
    
    # 게임 작업 생성: 절반은 MCTS가 흑, 절반은 MCTS가 백
    games = [True] * (num_games // 2) + [False] * (num_games // 2)
    
    # 멀티프로세싱 실행
    print(f"Running {num_games} games with {num_workers} workers...")
    
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_game_worker,
        initargs=(board_size, mcts_simulations, use_scipy_kill, center_wall)
    ) as executor:
        futures = [executor.submit(_play_single_game, g) for g in games]
        
        # tqdm으로 진행 상황 표시
        for future in tqdm(as_completed(futures), total=num_games, desc="Games", unit="game"):
            try:
                winner, moves, mcts_is_black = future.result()
                stats["total_moves"] += moves
                
                if winner == "Draw":
                    stats["draws"] += 1
                elif mcts_is_black:
                    # MCTS가 흑일 때
                    if winner == "Black":
                        stats["mcts_black_wins"] += 1
                    else:
                        stats["random_white_wins"] += 1
                else:
                    # MCTS가 백일 때
                    if winner == "White":
                        stats["mcts_white_wins"] += 1
                    else:
                        stats["random_black_wins"] += 1
            except Exception as e:
                print(f"Game failed: {e}")
    
    elapsed_time = time.time() - start_time
    
    # 결과 출력
    print(f"\n{'='*60}")
    print("Tournament Results")
    print(f"{'='*60}")
    
    mcts_total_wins = stats["mcts_black_wins"] + stats["mcts_white_wins"]
    random_total_wins = stats["random_black_wins"] + stats["random_white_wins"]
    
    print(f"\nMCTS Total Wins:   {mcts_total_wins}/{num_games} ({mcts_total_wins/num_games*100:.1f}%)")
    print(f"  - As Black: {stats['mcts_black_wins']}/{num_games//2}")
    print(f"  - As White: {stats['mcts_white_wins']}/{num_games//2}")
    
    print(f"\nRandom Total Wins: {random_total_wins}/{num_games} ({random_total_wins/num_games*100:.1f}%)")
    print(f"  - As Black: {stats['random_black_wins']}/{num_games//2}")
    print(f"  - As White: {stats['random_white_wins']}/{num_games//2}")
    
    print(f"\nDraws: {stats['draws']}/{num_games}")
    print(f"\nAverage Moves per Game: {stats['total_moves']/num_games:.1f}")
    print(f"Total Time: {elapsed_time:.1f}s ({elapsed_time/num_games:.2f}s per game)")
    print(f"{'='*60}\n")
    
    return stats
    print(f"{'='*60}\n")


def play_random_match(size=5, center_wall=True):
    env = GreatKingdomEnv(board_size=size, center_wall=center_wall)
    obs = env.reset()
    done = False
    moves = 0
    
    # print(f"=== Random Match (Size {size}x{size}) Started! ===")
    # env.render()
    
    while not done:
        legal_moves = env.get_legal_moves()
        possible_indices = np.where(legal_moves == 1)[0]
        
        if len(possible_indices) == 0:
            # 둘 곳이 없음 -> 무승부 or 패배 처리
            return "Draw", moves, "No Legal Moves"
        
        # 패스가 아닌 수가 있으면 패스 확률 낮추기 (패스는 마지막 인덱스)
        non_pass_moves = possible_indices[possible_indices != env.pass_action]
        if len(non_pass_moves) > 0 and np.random.random() > 0.1:  # 90%는 일반 수
            action = np.random.choice(non_pass_moves)
        else:
            action = np.random.choice(possible_indices)
            
        # 액션 출력
        player_name = "Black" if env.current_player == 1 else "White"
        if action == env.pass_action:
            # print(f"Move {moves+1}: {player_name} passes")
            pass
        else:
            r, c = divmod(action, size)
            # print(f"Move {moves+1}: {player_name} plays at ({r}, {c})")
            
        obs, reward, done, info = env.step(action)
        moves += 1
        
        # env.render()
        
        if moves > size * size * 2: # 무한루프 방지
            return "Draw (Timeout)", moves, "Move Limit Exceeded"
    
    # 결과 처리
    if "winner" in info:  # 영토 계산으로 끝난 경우
        winner = info["winner"]
        result = f"{info['result']} (B:{info['black_territory']} vs W:{info['white_territory']})"
        # print(f"Game Over by Territory! {winner} Wins!  Black: {info['black_territory']}, White: {info['white_territory']}")
    else:
        # 캡처나 자살수로 끝난 경우
        if reward == 1:
            winner = "Black" if env.current_player == 1 else "White"
        elif reward == -1:
            winner = "White" if env.current_player == 1 else "Black"
        else:
            winner = "Draw"
        result = info.get('result', 'Unknown')


    # print(f"Game Over! {winner} wins in {moves} moves. Reason: {result}")
        
    return winner, moves, result


def play_human_vs_human(size=4, center_wall=False):
    env = GreatKingdomEnv(board_size=size, center_wall=center_wall)
    obs = env.reset()
    done = False
    
    print(f"=== Great Kingdom (Size {size}x{size}) Started! ===")
    print("Rule: Capture 1 stone to win, or pass twice to count territory.")
    print("Input Format: 'row col' (e.g., 2 3) or 'pass'")
    
    env.render()
    
    while not done:
        legal_moves = env.get_legal_moves()
        if np.sum(legal_moves) == 0:
            print("No legal moves! Draw.")
            break
            
        try:
            user_input = input(f"{'Black' if env.current_player==1 else 'White'}'s move > ").strip().lower()
            
            # 패스 처리
            if user_input == 'pass':
                action = env.pass_action
            else:
                r, c = map(int, user_input.split())
                action = r * size + c
                
                if not (0 <= r < size and 0 <= c < size):
                    print("Out of bounds.")
                    continue
                    
                if legal_moves[action] == 0:
                    print("Illegal move (Occupied or Territory). Try again.")
                    continue
                
            obs, reward, done, info = env.step(action)
            env.render()
            
            if done:
                if "winner" in info:  # 영토 계산
                    print(f"Game Over by Territory! {info['winner']} Wins!")
                    print(f"  Black: {info['black_territory']}, White: {info['white_territory']}")
                elif reward == 1:
                    winner = "Black" if env.current_player == 1 else "White"
                    print(f"Game Over! {winner} Wins! ({info.get('result', '')})")
                elif reward == -1:
                    loser = "Black" if env.current_player == 1 else "White"
                    print(f"Game Over! {loser} Lost! ({info.get('result', '')})")
                else:
                    print(f"Game Over! Draw.")
                    
        except ValueError:
            print("Invalid input format. Enter 'row col' or 'pass'.")
        except KeyboardInterrupt:
            print("\nGame Aborted.")
            break

        # 환경 초기화]


# 실행하려면 아래 주석을 해제하세요

# 테스트 실행
if __name__ == "__main__":
    # Windows 멀티프로세싱 보호
    mp.freeze_support()
    
    print("=== Random vs Random Test (100 Games) ===")
    black_wins = 0
    white_wins = 0
    for i in tqdm(range(100), desc="Random vs Random"):
        winner, moves, reason = play_random_match(5, center_wall=False)
        if winner == "Black": black_wins += 1
        elif winner == "White": white_wins += 1

    print(f"Final Score - Black: {black_wins}, White: {white_wins}")
    
    # === MCTS 토너먼트 (멀티프로세싱) ===
    print("\n" + "="*70)
    print("=== MCTS Performance Test (Multiprocessing with 10 workers) ===")
    print("="*70 + "\n")
    
    run_tournament(
        num_games=50, 
        mcts_simulations=200, 
        board_size=5, 
        use_scipy_kill=True,
        num_workers=10,
        center_wall=False
    )
