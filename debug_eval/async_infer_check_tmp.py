import os
import sys
import time
import numpy as np
import torch
import multiprocessing as mp

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from network import AlphaZeroNetwork, encode_board_from_state
from train.selfplay_workers import BatchInferenceServer, AsyncNetworkClient
from env.env import GreatKingdomEnv


def infer_features(state_dict):
    conv_w = state_dict["conv_input.weight"]
    in_channels = conv_w.shape[1]
    if in_channels == 4:
        return False, 2, False
    if in_channels == 6:
        return True, 1, False
    if in_channels == 8:
        return True, 2, False
    if in_channels == 10:
        return True, 3, False
    if in_channels == 12:
        return True, 3, True
    raise RuntimeError(f"Unsupported input channels: {in_channels}")


def main():
    checkpoint_path = "checkpoints/board_4/center_wall_off/alphazero_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["network"]

    use_lib, lib_bins, use_last = infer_features(state_dict)
    print(f"Inferred: use_liberty={use_lib}, liberty_bins={lib_bins}, use_last_moves={use_last}")

    net = AlphaZeroNetwork(
        board_size=4,
        num_res_blocks=checkpoint.get("num_res_blocks", 2),
        num_channels=checkpoint.get("num_channels", 48),
        use_liberty_features=use_lib,
        liberty_bins=lib_bins,
        use_last_moves=use_last
    )
    net.load_state_dict(state_dict)
    net.eval()

    env = GreatKingdomEnv(board_size=4, center_wall=False)

    # build random encoded states
    states = []
    last_moves = (None, None)
    for _ in range(12):
        env.reset()
        # play a few random moves
        for _ in range(3):
            legal = env.get_legal_moves()
            idx = np.where(legal)[0]
            action = int(np.random.choice(idx))
            env.step(action)
            if use_last and action != env.pass_action:
                last_moves = (action, last_moves[0])
        encoded = encode_board_from_state(
            env.board.copy(),
            env.current_player,
            env.board_size,
            last_moves=last_moves,
            use_liberty_features=use_lib,
            liberty_bins=lib_bins,
            use_last_moves=use_last
        )
        states.append(encoded)

    states_np = np.array(states, dtype=np.float32)
    with torch.no_grad():
        ref_policy, ref_value = net.predict_batch(states_np)

    network_state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    server = BatchInferenceServer(
        network_state_dict=network_state_dict,
        board_size=4,
        num_res_blocks=checkpoint.get("num_res_blocks", 2),
        num_channels=checkpoint.get("num_channels", 48),
        device="cpu",
        batch_size=8,
        timeout=0.01,
        use_liberty_features=use_lib,
        liberty_bins=lib_bins,
        use_last_moves=use_last
    )
    output_queue = mp.Queue(maxsize=1000)
    server.output_queues = [output_queue]

    server_process = mp.Process(target=server.run)
    server_process.start()

    client = AsyncNetworkClient(server.input_queue, output_queue, worker_id=0)
    pol, val = client.predict_batch(states_np)

    server.input_queue.put(None)
    server_process.join()

    max_policy_diff = float(np.max(np.abs(pol - ref_policy)))
    max_value_diff = float(np.max(np.abs(val - ref_value)))
    print(f"Max policy diff: {max_policy_diff:.6f}")
    print(f"Max value diff: {max_value_diff:.6f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
