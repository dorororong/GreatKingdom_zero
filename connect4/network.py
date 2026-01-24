"""
Connect4 AlphaZero-style network (policy + value).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AlphaZeroNetwork(nn.Module):
    def __init__(self, rows=6, cols=7, num_res_blocks=3, num_channels=64):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.action_space_size = cols
        self.input_channels = 3

        self.conv_input = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * rows * cols, self.action_space_size)

        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(rows * cols, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def predict(self, state_tensor):
        self.eval()
        with torch.no_grad():
            if isinstance(state_tensor, np.ndarray):
                state_tensor = torch.FloatTensor(state_tensor)
            if state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(0)
            device = next(self.parameters()).device
            state_tensor = state_tensor.to(device)
            policy, value = self.forward(state_tensor)
            policy = torch.exp(policy).squeeze(0).cpu().numpy()
            value = value.squeeze().item()
        return policy, value

    def predict_batch(self, state_tensors):
        self.eval()
        with torch.no_grad():
            if isinstance(state_tensors, np.ndarray):
                state_tensors = torch.FloatTensor(state_tensors)
            if state_tensors.dim() == 3:
                state_tensors = state_tensors.unsqueeze(0)
            device = next(self.parameters()).device
            state_tensors = state_tensors.to(device)
            policy, value = self.forward(state_tensors)
            policy = torch.exp(policy).cpu().numpy()
            value = value.squeeze(1).cpu().numpy()
        return policy, value


def encode_board_from_state(board, current_player, rows=6, cols=7):
    opponent = 2 if current_player == 1 else 1
    encoded = np.zeros((3, rows, cols), dtype=np.float32)
    encoded[0] = (board == current_player).astype(np.float32)
    encoded[1] = (board == opponent).astype(np.float32)
    encoded[2] = 1.0 if current_player == 1 else 0.0
    return encoded
