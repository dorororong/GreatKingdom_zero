"""
AlphaZero 모델 테스트 - 랜덤 플레이어 및 순수 MCTS와 대결

사용법:
    python test_alphazero.py --checkpoint alphazero_best.pt --opponent random --games 100
    python test_alphazero.py --checkpoint alphazero_best.pt --opponent mcts --mcts_sims 500 --games 50
"""

import numpy as np
import torch
import argparse
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

from env.env import GreatKingdomEnv
from network import AlphaZeroNetwork, encode_board_from_state
from mcts_alphazero import AlphaZeroMCTS
from mcts import MCTS

# GPU 설정 - 전역 변수로 초기화
device = None  # main()에서 설정


# ========== 멀티프로세싱용 전역 변수 ==========
_worker_network = None
_worker_env = None
_worker_mcts = None
_worker_config = None


def _init_worker(network_state_dict, config):
    """워커 프로세스 초기화 - 각 워커마다 네트워크/환경 생성"""
    global _worker_network, _worker_env, _worker_mcts, _worker_config
    
    _worker_config = config
    _worker_env = GreatKingdomEnv(
        board_size=config['board_size'],
        center_wall=config.get('center_wall', True)
    )
    _worker_network = AlphaZeroNetwork(
        board_size=config['board_size'],
        num_res_blocks=config['num_res_blocks'],
        num_channels=config['num_channels']
    )
    _worker_network.load_state_dict(network_state_dict)
    _worker_network.eval()
    
    _worker_mcts = AlphaZeroMCTS(
        _worker_network, _worker_env,
        c_puct=config['c_puct'],
        num_simulations=config['alphazero_sims']
    )


def _play_one_game_worker(args):
    """워커에서 한 게임 수행 (멀티프로세싱용)"""
    game_idx, opponent_type, mcts_sims = args
    
    global _worker_network, _worker_env, _worker_mcts, _worker_config
    
    # 환경 리셋
    _worker_env.reset()
    
    # 상대 플레이어 생성 (워커 내부에서)
    if opponent_type == 'random':
        opponent_get_action = lambda state: _random_action(_worker_env, state)
    else:  # mcts
        mcts_player = MCTS(_worker_env, simulations_per_move=mcts_sims)
        opponent_get_action = lambda state: mcts_player.run(state)
    
    # AlphaZero 행동 선택 함수
    def alphazero_get_action(state):
        action, _ = _worker_mcts.run(state, temperature=_worker_config['temperature'])
        return action
    
    # 선공/후공 결정
    alphazero_is_black = (game_idx % 2 == 0)
    
    move_count = 0
    max_moves = 200
    
    while move_count < max_moves:
        current_player = _worker_env.current_player
        state = (_worker_env.board.copy(), _worker_env.current_player, _worker_env.consecutive_passes)
        
        # 현재 플레이어의 행동 선택
        if (current_player == 1 and alphazero_is_black) or (current_player == 2 and not alphazero_is_black):
            action = alphazero_get_action(state)
        else:
            action = opponent_get_action(state)
        
        # 행동 수행
        obs, reward, done, info = _worker_env.step(action)
        move_count += 1
        
        if done:
            if "winner" in info:
                if info["winner"] == "Black":
                    winner = 1
                elif info["winner"] == "White":
                    winner = 2
                else:
                    winner = 0
            elif reward == 1:
                winner = 1 if _worker_env.current_player == 2 else 2
            else:
                winner = 0
            break
    else:
        winner = 0  # 최대 수 초과
    
    # 결과 반환
    alphazero_won = (alphazero_is_black and winner == 1) or (not alphazero_is_black and winner == 2)
    opponent_won = (alphazero_is_black and winner == 2) or (not alphazero_is_black and winner == 1)
    
    return {
        'alphazero_won': alphazero_won,
        'opponent_won': opponent_won,
        'draw': winner == 0,
        'alphazero_is_black': alphazero_is_black,
        'moves': move_count
    }


def _random_action(env, state):
    """랜덤 행동 선택 (워커용)"""
    board, player, passes = state
    env.board = board.copy()
    env.current_player = player
    env.consecutive_passes = passes
    
    legal_mask = env.get_legal_moves()
    legal_actions = np.where(legal_mask)[0]
    
    if len(legal_actions) == 0:
        return env.pass_action
    
    return np.random.choice(legal_actions)


class RandomPlayer:
    """랜덤 플레이어"""
    def __init__(self, env):
        self.env = env
    
    def get_action(self, state):
        """
        state: (board, current_player, consecutive_passes) 튜플
        """
        board, player, passes = state
        
        # 환경에 상태 설정
        self.env.board = board.copy()
        self.env.current_player = player
        self.env.consecutive_passes = passes
        
        # 합법적인 수 가져오기
        legal_mask = self.env.get_legal_moves()
        legal_actions = np.where(legal_mask)[0]
        
        if len(legal_actions) == 0:
            return self.env.pass_action
        
        return np.random.choice(legal_actions)


class MCTSPlayer:
    """순수 MCTS 플레이어"""
    def __init__(self, env, num_simulations=1000):
        self.mcts = MCTS(env, simulations_per_move=num_simulations)
    
    def get_action(self, state):
        """
        state: (board, current_player, consecutive_passes) 튜플
        """
        return self.mcts.run(state)


class AlphaZeroPlayer:
    """AlphaZero 플레이어"""
    def __init__(self, network, env, num_simulations=100, c_puct=1.5, temperature=0.0):
        self.mcts = AlphaZeroMCTS(
            network, env,
            c_puct=c_puct,
            num_simulations=num_simulations
        )
        self.temperature = temperature
    
    def get_action(self, state):
        """
        state: (board, current_player, consecutive_passes) 튜플
        """
        action, _ = self.mcts.run(state, temperature=self.temperature)
        return action


def load_model(checkpoint_path, board_size=5, num_res_blocks=3, num_channels=64):
    """체크포인트에서 모델 로드"""
    global device  # 전역 device 사용
    
    network = AlphaZeroNetwork(
        board_size=board_size,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 체크포인트 형식에 따라 로드
        if isinstance(checkpoint, dict):
            if 'network' in checkpoint:
                network.load_state_dict(checkpoint['network'])
                if 'train_step' in checkpoint:
                    print(f"  학습 스텝: {checkpoint['train_step']}")
                if 'total_games' in checkpoint:
                    print(f"  총 게임 수: {checkpoint['total_games']}")
            elif 'network_state_dict' in checkpoint:
                network.load_state_dict(checkpoint['network_state_dict'])
            elif 'model_state_dict' in checkpoint:
                network.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                network.load_state_dict(checkpoint['state_dict'])
            else:
                try:
                    network.load_state_dict(checkpoint)
                except:
                    print(f"✗ 알 수 없는 체크포인트 형식입니다.")
                    print(f"  사용 가능한 키: {checkpoint.keys()}")
                    return None
        else:
            network.load_state_dict(checkpoint)
        
        print(f"✓ 모델 로드 완료: {checkpoint_path}")
    else:
        print(f"✗ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return None
    
    network.eval()
    network.to(device)
    print(f"✓ 디바이스: {device}")
    
    return network


def play_game(player1, player2, env, max_moves=200, verbose=False):
    """
    두 플레이어 간 한 게임 진행
    
    Args:
        player1: 흑(1번 플레이어)
        player2: 백(2번 플레이어)
        env: 게임 환경
        max_moves: 최대 수
        verbose: 상세 출력 여부
    
    Returns:
        winner: 1 (player1 승), 2 (player2 승), 0 (무승부)
        move_count: 총 수
    """
    env.reset()
    players = {1: player1, 2: player2}
    move_count = 0
    
    while move_count < max_moves:
        current_player = env.current_player
        state = (env.board.copy(), env.current_player, env.consecutive_passes)
        
        # 현재 플레이어의 행동 선택
        action = players[current_player].get_action(state)
        
        if verbose:
            if action == env.pass_action:
                print(f"[{move_count+1}] Player {current_player}: PASS")
            else:
                row, col = action // env.board_size, action % env.board_size
                print(f"[{move_count+1}] Player {current_player}: ({row}, {col})")
        
        # 행동 수행
        obs, reward, done, info = env.step(action)
        move_count += 1
        
        if done:
            if "winner" in info:
                if info["winner"] == "Black":
                    return 1, move_count
                elif info["winner"] == "White":
                    return 2, move_count
                else:
                    return 0, move_count
            elif reward == 1:
                # 현재 플레이어 (전환 후)의 상대가 이김
                return 1 if env.current_player == 2 else 2, move_count
            else:
                return 0, move_count
    
    # 최대 수 초과 - 무승부
    return 0, move_count


def run_tournament(alphazero_player, opponent_player, env, num_games=100, verbose=False):
    """
    토너먼트 진행 (AlphaZero vs Opponent) - 단일 프로세스 버전
    
    각 게임마다 선공/후공을 번갈아가며 진행
    
    Returns:
        results: dict with win/loss/draw counts
    """
    results = {
        'alphazero_wins': 0,
        'opponent_wins': 0,
        'draws': 0,
        'alphazero_as_black_wins': 0,
        'alphazero_as_white_wins': 0,
        'total_moves': 0
    }
    
    print(f"\n{'='*50}")
    print(f"토너먼트 시작: {num_games} 게임 (단일 프로세스)")
    print(f"{'='*50}\n")
    
    for game_idx in tqdm(range(num_games), desc="게임 진행중"):
        # 번갈아가며 선공/후공 결정
        alphazero_is_black = (game_idx % 2 == 0)
        
        if alphazero_is_black:
            player1, player2 = alphazero_player, opponent_player
        else:
            player1, player2 = opponent_player, alphazero_player
        
        winner, moves = play_game(player1, player2, env, verbose=verbose)
        results['total_moves'] += moves
        
        if winner == 0:
            results['draws'] += 1
        elif alphazero_is_black:
            if winner == 1:  # AlphaZero(흑) 승리
                results['alphazero_wins'] += 1
                results['alphazero_as_black_wins'] += 1
            else:  # 상대 승리
                results['opponent_wins'] += 1
        else:  # AlphaZero is white
            if winner == 2:  # AlphaZero(백) 승리
                results['alphazero_wins'] += 1
                results['alphazero_as_white_wins'] += 1
            else:  # 상대 승리
                results['opponent_wins'] += 1
        
        if verbose:
            print(f"\n게임 {game_idx+1}: ", end="")
            if winner == 0:
                print("무승부")
            else:
                winner_name = "AlphaZero" if (alphazero_is_black and winner == 1) or (not alphazero_is_black and winner == 2) else "Opponent"
                print(f"{winner_name} 승리 (수: {moves})")
    
    return results


def run_tournament_parallel(network_state_dict, config, num_games=100, num_workers=None):
    """
    병렬 토너먼트 진행 (멀티프로세싱)
    
    Args:
        network_state_dict: 네트워크 가중치
        config: 설정 딕셔너리
        num_games: 총 게임 수
        num_workers: 워커 수 (None이면 CPU 코어 수 - 2)
    
    Returns:
        results: dict with win/loss/draw counts
    """
    if num_workers is None:
        # CPU 코어 수 - 2 (시스템 여유분 확보)
        num_workers = max(1, mp.cpu_count() - 2)
    num_workers = min(num_workers, num_games)  # 게임 수보다 많으면 안됨
    
    results = {
        'alphazero_wins': 0,
        'opponent_wins': 0,
        'draws': 0,
        'alphazero_as_black_wins': 0,
        'alphazero_as_white_wins': 0,
        'total_moves': 0
    }
    
    print(f"\n{'='*50}")
    print(f"토너먼트 시작: {num_games} 게임 (병렬 처리: {num_workers} 워커)")
    print(f"{'='*50}\n")
    
    # 게임 인자 준비
    game_args = [(i, config['opponent_type'], config['mcts_sims']) for i in range(num_games)]
    
    start_time = time.time()
    
    # ProcessPoolExecutor 사용
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(network_state_dict, config)
    ) as executor:
        # 모든 게임 제출
        futures = [executor.submit(_play_one_game_worker, args) for args in game_args]
        
        # 결과 수집 (진행 표시)
        for future in tqdm(as_completed(futures), total=num_games, desc="게임 진행중"):
            result = future.result()
            
            if result['draw']:
                results['draws'] += 1
            elif result['alphazero_won']:
                results['alphazero_wins'] += 1
                if result['alphazero_is_black']:
                    results['alphazero_as_black_wins'] += 1
                else:
                    results['alphazero_as_white_wins'] += 1
            elif result['opponent_won']:
                results['opponent_wins'] += 1
            
            results['total_moves'] += result['moves']
    
    elapsed = time.time() - start_time
    print(f"\n소요 시간: {elapsed:.1f}초 ({elapsed/num_games:.2f}초/게임)")
    
    return results


def print_results(results, opponent_name):
    """결과 출력"""
    total = results['alphazero_wins'] + results['opponent_wins'] + results['draws']
    
    print(f"\n{'='*50}")
    print(f"최종 결과: AlphaZero vs {opponent_name}")
    print(f"{'='*50}")
    print(f"총 게임 수: {total}")
    print(f"")
    print(f"AlphaZero 승리: {results['alphazero_wins']} ({100*results['alphazero_wins']/total:.1f}%)")
    print(f"  - 흑으로 승리: {results['alphazero_as_black_wins']}")
    print(f"  - 백으로 승리: {results['alphazero_as_white_wins']}")
    print(f"")
    print(f"{opponent_name} 승리: {results['opponent_wins']} ({100*results['opponent_wins']/total:.1f}%)")
    print(f"무승부: {results['draws']} ({100*results['draws']/total:.1f}%)")
    print(f"")
    print(f"평균 게임 길이: {results['total_moves']/total:.1f} 수")
    print(f"{'='*50}")
    
    # 승률 계산 (무승부는 0.5로 계산)
    win_rate = (results['alphazero_wins'] + 0.5 * results['draws']) / total
    print(f"\nAlphaZero 승률 (무승부=0.5): {100*win_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='AlphaZero 모델 테스트')
    
    # 체크포인트 설정
    parser.add_argument('--checkpoint', type=str, default='checkpoints/alphazero_best.pt',
                        help='체크포인트 파일 경로')
    
    # 상대 설정
    parser.add_argument('--opponent', type=str, choices=['random', 'mcts'], default='random',
                        help='상대 타입: random 또는 mcts')
    
    # 게임 설정
    parser.add_argument('--games', type=int, default=100,
                        help='테스트할 게임 수')
    parser.add_argument('--board_size', type=int, default=5,
                        help='보드 크기')
    parser.add_argument('--center_wall', type=str, default='True',
                        help='중앙 중립 기물(벽) 배치 여부 (True/False)')
    
    # AlphaZero 설정
    parser.add_argument('--alphazero_sims', type=int, default=100,
                        help='AlphaZero MCTS 시뮬레이션 횟수')
    parser.add_argument('--c_puct', type=float, default=1.5,
                        help='PUCT 탐색 상수')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='행동 선택 온도 (0=greedy)')
    
    # 네트워크 설정
    parser.add_argument('--num_res_blocks', type=int, default=3,
                        help='ResNet 블록 수')
    parser.add_argument('--num_channels', type=int, default=64,
                        help='컨볼루션 채널 수')
    
    # MCTS 상대 설정
    parser.add_argument('--mcts_sims', type=int, default=1000,
                        help='순수 MCTS 시뮬레이션 횟수')
    
    # 기타
    parser.add_argument('--verbose', action='store_true',
                        help='상세 출력 모드')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드')
    
    # GPU 설정 추가
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['auto', 'cpu', 'cuda'],
                        help='연산 장치 (기본: cpu, auto=자동 선택, cuda)')
    
    # 병렬 처리 설정
    parser.add_argument('--parallel', action='store_true',
                        help='멀티프로세싱 병렬 처리 사용')
    parser.add_argument('--workers', type=int, default=None,
                        help='워커 프로세스 수 (기본: CPU 코어 수 - 2)')
    
    args = parser.parse_args()
    
    # 디바이스 설정 - 전역 변수 업데이트
    global device
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("⚠ CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:  # auto
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*50}")
    print(f"사용 디바이스: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'='*50}")
    
    # 환경 생성
    env = GreatKingdomEnv(board_size=args.board_size, center_wall=args.center_wall.lower() == 'true')
    
    # 모델 로드
    network = load_model(
        args.checkpoint,
        board_size=args.board_size,
        num_res_blocks=args.num_res_blocks,
        num_channels=args.num_channels
    )
    
    if network is None:
        print("모델 로드 실패. 종료합니다.")
        return
    
    # AlphaZero 플레이어 생성
    alphazero_player = AlphaZeroPlayer(
        network, env,
        num_simulations=args.alphazero_sims,
        c_puct=args.c_puct,
        temperature=args.temperature
    )
    
    # 상대 설정
    if args.opponent == 'random':
        opponent_name = "Random"
        print(f"\n상대: 랜덤 플레이어")
    else:  # mcts
        opponent_name = f"MCTS ({args.mcts_sims} sims)"
        print(f"\n상대: 순수 MCTS ({args.mcts_sims} simulations)")
    
    print(f"AlphaZero: {args.alphazero_sims} simulations, c_puct={args.c_puct}, temp={args.temperature}")
    
    # 토너먼트 진행
    if args.parallel:
        # 병렬 처리 모드
        config = {
            'board_size': args.board_size,
            'num_res_blocks': args.num_res_blocks,
            'num_channels': args.num_channels,
            'alphazero_sims': args.alphazero_sims,
            'c_puct': args.c_puct,
            'temperature': args.temperature,
            'opponent_type': args.opponent,
            'mcts_sims': args.mcts_sims,
            'center_wall': args.center_wall.lower() == 'true'
        }
        results = run_tournament_parallel(
            network.state_dict(),
            config,
            num_games=args.games,
            num_workers=args.workers
        )
    else:
        # 단일 프로세스 모드
        alphazero_player = AlphaZeroPlayer(
            network, env,
            num_simulations=args.alphazero_sims,
            c_puct=args.c_puct,
            temperature=args.temperature
        )
        
        if args.opponent == 'random':
            opponent_player = RandomPlayer(env)
        else:
            opponent_player = MCTSPlayer(env, num_simulations=args.mcts_sims)
        
        results = run_tournament(
            alphazero_player, opponent_player, env,
            num_games=args.games,
            verbose=args.verbose
        )
    
    # 결과 출력
    print_results(results, opponent_name)


def test_single_game(checkpoint_path='checkpoints/alphazero_best.pt', opponent='random', verbose=True, center_wall=True):
    """
    단일 게임 테스트 (빠른 테스트용)
    """
    global device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n테스트 디바이스: {device}")
    
    env = GreatKingdomEnv(board_size=5, center_wall=center_wall)
    
    # 모델 로드
    network = load_model(checkpoint_path)
    if network is None:
        return
    
    # 플레이어 생성
    alphazero_player = AlphaZeroPlayer(network, env, num_simulations=50, temperature=0.0)
    
    if opponent == 'random':
        opponent_player = RandomPlayer(env)
    else:
        opponent_player = MCTSPlayer(env, num_simulations=500)
    
    # 게임 진행
    print("\n=== 단일 게임 테스트 ===")
    print("AlphaZero (흑) vs Opponent (백)\n")
    
    env.reset()
    move_count = 0
    
    while move_count < 100:
        current = env.current_player
        state = (env.board.copy(), env.current_player, env.consecutive_passes)
        
        if current == 1:  # AlphaZero
            action = alphazero_player.get_action(state)
            player_name = "AlphaZero"
        else:
            action = opponent_player.get_action(state)
            player_name = "Opponent"
        
        if verbose:
            if action == env.pass_action:
                print(f"[{move_count+1}] {player_name}: PASS")
            else:
                row, col = action // env.board_size, action % env.board_size
                print(f"[{move_count+1}] {player_name}: ({row}, {col})")
        
        obs, reward, done, info = env.step(action)
        move_count += 1
        
        if verbose:
            print_board(env.board)
            print()
        
        if done:
            print(f"\n게임 종료! (수: {move_count})")
            if "winner" in info:
                print(f"승자: {info['winner']}")
                if 'black_score' in info:
                    print(f"점수 - 흑: {info['black_score']}, 백: {info['white_score']}")
            break
    
    return info


def print_board(board):
    """보드 출력"""
    symbols = {0: '.', 1: '●', 2: '○', 3: '█'}
    print("  ", end="")
    for c in range(board.shape[1]):
        print(f"{c} ", end="")
    print()
    for r in range(board.shape[0]):
        print(f"{r} ", end="")
        for c in range(board.shape[1]):
            print(f"{symbols[board[r, c]]} ", end="")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 인자 없이 실행 시 기본 테스트
        print("사용법:")
        print("  python test_alphazero.py --checkpoint checkpoints/alphazero_iter5.pt --opponent random --games 20")
        print("  python test_alphazero.py --checkpoint checkpoints/alphazero_best.pt --opponent mcts --mcts_sims 500 --games 20")
        print("\n기본 테스트 (Random 상대로 20게임)를 실행합니다...")
        print("-" * 50)
        
        # 기본 테스트 실행
        sys.argv.extend(['--games', '10', '--opponent', 'random'])
        main()
    else:
        main()
