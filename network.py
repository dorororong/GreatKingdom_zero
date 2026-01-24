"""
AlphaZero 신경망 - 간단한 ResNet 구조
Policy Head: 각 행동의 확률 분포
Value Head: 현재 상태의 승률 평가 (-1 ~ 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration."""
    def __init__(self, num_channels, reduction=16):
        super().__init__()
        reduced = max(1, num_channels // reduction)
        self.fc1 = nn.Linear(num_channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, num_channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3))
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """Residual Block with SE."""
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.se = SEBlock(num_channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero 네트워크 (간단 버전)
    
    입력: (batch, channels, board_size, board_size)
        - 채널 0: 현재 플레이어 돌 (1이면 돌 있음)
        - 채널 1: 상대 플레이어 돌
        - 채널 2: 벽 (중립 기물)
        - 채널 3: 현재 플레이어 표시 (전체 1 또는 0)
    
    출력:
        - policy: (batch, action_space_size) - 각 행동의 로그 확률
        - value: (batch, 1) - 현재 상태 평가 (-1 ~ 1)
    """
    
    def __init__(self, board_size=5, num_res_blocks=3, num_channels=64,
                 use_liberty_features=True, liberty_bins=2, use_last_moves=False,
                 head_type="fc"):
        super().__init__()
        self.board_size = board_size
        self.action_space_size = board_size * board_size + 1  # +1 for pass
        self.use_liberty_features = use_liberty_features
        self.liberty_bins = liberty_bins
        self.use_last_moves = use_last_moves
        self.head_type = head_type
        
        # 입력 채널: 4 (현재 플레이어 돌, 상대 돌, 벽, 현재 플레이어 표시)
        self.input_channels = get_input_channels(
            use_liberty_features=use_liberty_features,
            liberty_bins=liberty_bins,
            use_last_moves=use_last_moves
        )
        
        # === 공통 Backbone ===
        # 초기 컨볼루션
        self.conv_input = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # === Policy / Value Head ===
        if self.head_type == "conv":
            self.policy_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
            self.policy_bn = nn.BatchNorm2d(1)
            self.pass_fc = nn.Linear(num_channels, 1)

            self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(1, 64)
            self.value_fc2 = nn.Linear(64, 1)
        else:
            self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_space_size)

            self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_fc1 = nn.Linear(board_size * board_size, 64)
            self.value_fc2 = nn.Linear(64, 1)

        # === Ownership Head ===
        self.owner_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.owner_bn = nn.BatchNorm2d(2)
    
    def forward(self, x):
        """
        Args:
            x: (batch, 4, board_size, board_size) 텐서
        Returns:
            policy: (batch, action_space_size) - log softmax 확률
            value: (batch, 1) - tanh 활성화된 상태 가치
        """
        # 공통 backbone
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
        
        # Policy / Value head
        if self.head_type == "conv":
            policy = F.relu(self.policy_bn(self.policy_conv(x)))
            policy = policy.flatten(1)
            pooled = x.mean(dim=(2, 3))
            pass_logit = self.pass_fc(pooled)
            policy = torch.cat([policy, pass_logit], dim=1)
            policy = F.log_softmax(policy, dim=1)

            value = F.relu(self.value_bn(self.value_conv(x)))
            value = value.mean(dim=(2, 3))
            value = F.relu(self.value_fc1(value))
            value = torch.tanh(self.value_fc2(value))
        else:
            policy = F.relu(self.policy_bn(self.policy_conv(x)))
            policy = policy.view(policy.size(0), -1)  # Flatten
            policy = self.policy_fc(policy)
            policy = F.log_softmax(policy, dim=1)

            value = F.relu(self.value_bn(self.value_conv(x)))
            value = value.view(value.size(0), -1)  # Flatten
            value = F.relu(self.value_fc1(value))
            value = torch.tanh(self.value_fc2(value))
        
        # Ownership head (logits)
        ownership = self.owner_bn(self.owner_conv(x))

        return policy, value, ownership
    
    def predict(self, state_tensor):
        """
        단일 상태에 대한 예측 (inference용)
        
        Args:
            state_tensor: (4, board_size, board_size) numpy array 또는 torch tensor
        Returns:
            policy: (action_space_size,) numpy array - 확률 분포
            value: float - 상태 가치
        """
        self.eval()
        with torch.no_grad():
            if isinstance(state_tensor, np.ndarray):
                state_tensor = torch.FloatTensor(state_tensor)
            
            # 배치 차원 추가
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            
            # GPU로 이동 (가능하면)
            device = next(self.parameters()).device
            state_tensor = state_tensor.to(device)
            
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    policy, value, _ = self.forward(state_tensor)
            else:
                policy, value, _ = self.forward(state_tensor)
            
            # log_softmax -> softmax
            policy = torch.exp(policy).float().squeeze(0).cpu().numpy()
            value = value.float().squeeze().item()
            
        return policy, value

    def predict_batch(self, state_tensors):
        """
        Batch prediction.

        Args:
            state_tensors: (batch, 4, board, board) numpy array or torch tensor
        Returns:
            policies: (batch, action_space_size) numpy array
            values: (batch,) numpy array
        """
        self.eval()
        with torch.no_grad():
            if isinstance(state_tensors, np.ndarray):
                state_tensors = torch.FloatTensor(state_tensors)

            if state_tensors.dim() == 3:
                state_tensors = state_tensors.unsqueeze(0)

            device = next(self.parameters()).device
            state_tensors = state_tensors.to(device)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    policy, value, _ = self.forward(state_tensors)
            else:
                policy, value, _ = self.forward(state_tensors)
            policy = torch.exp(policy).float().cpu().numpy()
            value = value.squeeze(1).float().cpu().numpy()

        return policy, value


def _compute_liberty_maps(board, player, board_size):
    visited = np.zeros((board_size, board_size), dtype=bool)
    lib1 = np.zeros((board_size, board_size), dtype=np.float32)
    lib2 = np.zeros((board_size, board_size), dtype=np.float32)
    lib3 = np.zeros((board_size, board_size), dtype=np.float32)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(board_size):
        for c in range(board_size):
            if board[r, c] != player or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            group = [(r, c)]
            liberties = set()

            while stack:
                cr, cc = stack.pop()
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < board_size and 0 <= nc < board_size):
                        continue
                    if board[nr, nc] == 0:
                        liberties.add((nr, nc))
                    elif board[nr, nc] == player and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
                        group.append((nr, nc))

            lib_count = len(liberties)
            if lib_count <= 1:
                target = lib1
            elif lib_count == 2:
                target = lib2
            else:
                target = lib3
            for gr, gc in group:
                target[gr, gc] = 1.0

    return lib1, lib2, lib3


def _compute_liberty_maps_bins(board, player, board_size, liberty_bins):
    if liberty_bins == 1:
        visited = np.zeros((board_size, board_size), dtype=bool)
        lib1 = np.zeros((board_size, board_size), dtype=np.float32)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] != player or visited[r, c]:
                    continue
                stack = [(r, c)]
                visited[r, c] = True
                group = [(r, c)]
                liberties = set()

                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < board_size and 0 <= nc < board_size):
                            continue
                        if board[nr, nc] == 0:
                            liberties.add((nr, nc))
                        elif board[nr, nc] == player and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                            group.append((nr, nc))

                if len(liberties) <= 1:
                    for gr, gc in group:
                        lib1[gr, gc] = 1.0

        return (lib1,)
    if liberty_bins == 2:
        visited = np.zeros((board_size, board_size), dtype=bool)
        lib1 = np.zeros((board_size, board_size), dtype=np.float32)
        lib2p = np.zeros((board_size, board_size), dtype=np.float32)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] != player or visited[r, c]:
                    continue
                stack = [(r, c)]
                visited[r, c] = True
                group = [(r, c)]
                liberties = set()

                while stack:
                    cr, cc = stack.pop()
                    for dr, dc in directions:
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < board_size and 0 <= nc < board_size):
                            continue
                        if board[nr, nc] == 0:
                            liberties.add((nr, nc))
                        elif board[nr, nc] == player and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                            group.append((nr, nc))

                lib_count = len(liberties)
                target = lib1 if lib_count <= 1 else lib2p
                for gr, gc in group:
                    target[gr, gc] = 1.0

        return (lib1, lib2p)
    return _compute_liberty_maps(board, player, board_size)


def _append_liberty_features(encoded, board, current_player, board_size, liberty_bins, start_idx):
    opponent = 2 if current_player == 1 else 1
    if liberty_bins == 1:
        (c1,) = _compute_liberty_maps_bins(board, current_player, board_size, liberty_bins)
        (o1,) = _compute_liberty_maps_bins(board, opponent, board_size, liberty_bins)
        encoded[start_idx] = c1
        encoded[start_idx + 1] = o1
        return start_idx + 2
    if liberty_bins == 2:
        c1, c2p = _compute_liberty_maps_bins(board, current_player, board_size, liberty_bins)
        o1, o2p = _compute_liberty_maps_bins(board, opponent, board_size, liberty_bins)
        encoded[start_idx] = c1
        encoded[start_idx + 1] = c2p
        encoded[start_idx + 2] = o1
        encoded[start_idx + 3] = o2p
        return start_idx + 4
    c1, c2, c3 = _compute_liberty_maps(board, current_player, board_size)
    o1, o2, o3 = _compute_liberty_maps(board, opponent, board_size)
    encoded[start_idx] = c1
    encoded[start_idx + 1] = c2
    encoded[start_idx + 2] = c3
    encoded[start_idx + 3] = o1
    encoded[start_idx + 4] = o2
    encoded[start_idx + 5] = o3
    return start_idx + 6


def _append_last_moves(encoded, last_moves, board_size, start_idx):
    if not last_moves:
        return
    moves = list(last_moves)[:2]
    for idx, action in enumerate(moves):
        if action is None:
            continue
        if action >= board_size * board_size:
            continue
        r, c = divmod(int(action), board_size)
        encoded[start_idx + idx, r, c] = 1.0


def get_input_channels(use_liberty_features=True, liberty_bins=2, use_last_moves=False):
    channels = 4
    if use_liberty_features:
        if liberty_bins not in (1, 2, 3):
            raise ValueError(f"liberty_bins must be 1, 2, or 3, got {liberty_bins}")
        channels += 2 * liberty_bins
    if use_last_moves:
        channels += 2
    return channels


def encode_board_batch(boards, players, board_size, last_moves_list=None,
                       use_liberty_features=True, liberty_bins=2, use_last_moves=False):
    if last_moves_list is None:
        last_moves_list = [None] * len(boards)

    num_channels = get_input_channels(
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )
    batch = np.zeros((len(boards), num_channels, board_size, board_size), dtype=np.float32)
    for i, (board, current_player, last_moves) in enumerate(zip(boards, players, last_moves_list)):
        opponent = 2 if current_player == 1 else 1
        batch[i, 0] = (board == current_player).astype(np.float32)
        batch[i, 1] = (board == opponent).astype(np.float32)
        batch[i, 2] = (board == 3).astype(np.float32)
        batch[i, 3] = 1.0 if current_player == 1 else 0.0
        next_idx = 4
        if use_liberty_features:
            next_idx = _append_liberty_features(batch[i], board, current_player, board_size, liberty_bins, next_idx)
        if use_last_moves:
            _append_last_moves(batch[i], last_moves, board_size, next_idx)
    return batch


def encode_board(env, use_liberty_features=True, liberty_bins=2, use_last_moves=False, last_moves=None):
    """
    환경 상태를 신경망 입력 형태로 인코딩합니다.
    
    Args:
        env: GreatKingdomEnv 인스턴스
    Returns:
        (4, board_size, board_size) numpy array (float32)
    """
    board = env.board
    current_player = env.current_player
    opponent = 2 if current_player == 1 else 1
    
    # 4채널 인코딩
    num_channels = get_input_channels(
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )
    encoded = np.zeros((num_channels, env.board_size, env.board_size), dtype=np.float32)
    
    # 채널 0: 현재 플레이어 돌
    encoded[0] = (board == current_player).astype(np.float32)
    
    # 채널 1: 상대 플레이어 돌
    encoded[1] = (board == opponent).astype(np.float32)
    
    # 채널 2: 벽 (중립 기물, 값 3)
    encoded[2] = (board == 3).astype(np.float32)
    
    # 채널 3: 현재 플레이어 표시 (1이면 흑, 0이면 백)
    # 전체를 동일한 값으로 채움 (positional encoding 역할)
    encoded[3] = 1.0 if current_player == 1 else 0.0
    next_idx = 4
    if use_liberty_features:
        next_idx = _append_liberty_features(encoded, board, current_player, env.board_size, liberty_bins, next_idx)
    if use_last_moves:
        _append_last_moves(encoded, last_moves, env.board_size, next_idx)
    return encoded


def encode_board_from_state(board, current_player, board_size=5, last_moves=None,
                            use_liberty_features=True, liberty_bins=2, use_last_moves=False):
    """
    보드 배열과 현재 플레이어 정보로 인코딩 (MCTS에서 사용)
    
    Args:
        board: (board_size, board_size) numpy array
        current_player: 1 또는 2
        board_size: 보드 크기
    Returns:
        (4, board_size, board_size) numpy array (float32)
    """
    opponent = 2 if current_player == 1 else 1
    
    num_channels = get_input_channels(
        use_liberty_features=use_liberty_features,
        liberty_bins=liberty_bins,
        use_last_moves=use_last_moves
    )
    encoded = np.zeros((num_channels, board_size, board_size), dtype=np.float32)
    encoded[0] = (board == current_player).astype(np.float32)
    encoded[1] = (board == opponent).astype(np.float32)
    encoded[2] = (board == 3).astype(np.float32)
    encoded[3] = 1.0 if current_player == 1 else 0.0
    next_idx = 4
    if use_liberty_features:
        next_idx = _append_liberty_features(encoded, board, current_player, board_size, liberty_bins, next_idx)
    if use_last_moves:
        _append_last_moves(encoded, last_moves, board_size, next_idx)
    return encoded


def infer_head_type_from_state_dict(state_dict):
    if any(k.startswith("pass_fc.") for k in state_dict.keys()):
        return "conv"
    return "fc"


if __name__ == "__main__":
    # 테스트
    print("=== AlphaZero Network Test ===")
    
    # 네트워크 생성
    net = AlphaZeroNetwork(board_size=5, num_res_blocks=3, num_channels=64)
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 테스트 입력
    batch_size = 4
    test_input = torch.randn(batch_size, net.input_channels, 5, 5)
    
    # Forward pass
    policy, value, ownership = net(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Policy output shape: {policy.shape}")  # (batch, 26)
    print(f"Value output shape: {value.shape}")    # (batch, 1)
    print(f"Ownership output shape: {ownership.shape}")  # (batch, 2, board, board)
    
    # Policy 확률 합 확인 (log_softmax이므로 exp 후 합이 1)
    policy_probs = torch.exp(policy)
    print(f"Policy probabilities sum: {policy_probs.sum(dim=1)}")  # 각 배치마다 1
    
    # 단일 예측 테스트
    single_state = np.random.randn(4, 5, 5).astype(np.float32)
    policy_pred, value_pred = net.predict(single_state)
    print(f"\nSingle prediction:")
    print(f"  Policy shape: {policy_pred.shape}, sum: {policy_pred.sum():.4f}")
    print(f"  Value: {value_pred:.4f}")
    
    print("\n=== Test Passed! ===")
