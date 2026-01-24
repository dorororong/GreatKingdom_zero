import argparse
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.env import GreatKingdomEnv
from network import encode_board_from_state


def _build_channel_names(use_liberty_features, liberty_bins, use_last_moves):
    names = [
        "current_player_stones",
        "opponent_stones",
        "neutral_walls",
        "current_player_flag",
    ]
    if use_liberty_features:
        if liberty_bins == 1:
            names += [
                "curr_lib_1",
                "opp_lib_1",
            ]
        elif liberty_bins == 2:
            names += [
                "curr_lib_1",
                "curr_lib_2plus",
                "opp_lib_1",
                "opp_lib_2plus",
            ]
        else:
            names += [
                "curr_lib_1",
                "curr_lib_2",
                "curr_lib_3plus",
                "opp_lib_1",
                "opp_lib_2",
                "opp_lib_3plus",
            ]
    if use_last_moves:
        names += ["last_move", "prev_move"]
    return names


def _format_positions(mask):
    coords = list(zip(*np.where(mask > 0)))
    if not coords:
        return "[]"
    return "[" + ", ".join(f"({r},{c})" for r, c in coords) + "]"


def _print_board(board):
    size = board.shape[0]
    symbols = {0: ".", 1: "B", 2: "W", 3: "#"}
    header = "  " + " ".join(str(i) for i in range(size))
    print(header)
    for r in range(size):
        row = " ".join(symbols[int(x)] for x in board[r])
        print(f"{r} {row}")


def _print_channels(encoded, channel_names):
    assert encoded.shape[0] == len(channel_names), (
        f"expected {len(channel_names)} channels, got {encoded.shape[0]}"
    )
    for idx, name in enumerate(channel_names):
        ch = encoded[idx]
        count = int(np.sum(ch))
        print(f"  [{idx:02d}] {name}: sum={count} positions={_format_positions(ch)}")
        if count > 0:
            print(ch.astype(int))


def _parse_moves(moves_str):
    if not moves_str:
        return []
    return [int(tok.strip()) for tok in moves_str.split(",") if tok.strip()]


def main():
    parser = argparse.ArgumentParser(description="Debug 12-channel feature encoding.")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--center-wall", action="store_true", help="Place neutral wall at center.")
    parser.add_argument("--moves", type=str, default="", help="Comma-separated action indices.")
    parser.add_argument("--turns", type=int, default=6, help="Number of turns to inspect.")
    parser.add_argument("--use-liberty-features", action=argparse.BooleanOptionalAction, default=True,
                        help="Include liberty feature channels.")
    parser.add_argument("--liberty-bins", type=int, choices=(2, 3), default=2)
    parser.add_argument("--use-last-moves", action=argparse.BooleanOptionalAction, default=False,
                        help="Include last-move feature channels.")
    args = parser.parse_args()

    env = GreatKingdomEnv(board_size=args.board_size, center_wall=args.center_wall)
    last_moves = (None, None) if args.use_last_moves else None

    if args.moves:
        move_list = _parse_moves(args.moves)
    else:
        # Simple default sequence that alternates without touching center.
        size = args.board_size
        move_list = [0, 1, size, size + 1, 2, size + 2, 3, size + 3]

    channel_names = _build_channel_names(
        args.use_liberty_features, args.liberty_bins, args.use_last_moves
    )
    print("=== Feature Channel Debug ===")
    print(f"board_size={args.board_size} center_wall={args.center_wall}")
    print(f"use_liberty_features={args.use_liberty_features} liberty_bins={args.liberty_bins} use_last_moves={args.use_last_moves}")

    for turn_idx in range(min(args.turns, len(move_list))):
        print("=" * 60)
        print(f"Turn {turn_idx + 1} | Current player: {env.current_player}")
        print(f"Last moves (most recent first): {last_moves}")
        _print_board(env.board)

        encoded = encode_board_from_state(
            env.board, env.current_player, env.board_size, last_moves=last_moves,
            use_liberty_features=args.use_liberty_features,
            liberty_bins=args.liberty_bins,
            use_last_moves=args.use_last_moves
        )
        print(f"Encoded shape: {encoded.shape}")
        _print_channels(encoded, channel_names)

        action = move_list[turn_idx]
        obs, reward, done, info = env.step(action)
        if info.get("error"):
            print(f"Move {action} failed: {info['error']}")
            break
        if args.use_last_moves and action != env.pass_action:
            last_moves = (action, last_moves[0])
        print(f"Applied action={action} reward={reward} done={done} info={info}")
        if done:
            break


if __name__ == "__main__":
    main()
