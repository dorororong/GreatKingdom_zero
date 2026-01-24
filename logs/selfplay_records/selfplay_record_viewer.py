import glob
import json
import os
import sys

RECORD_DIR = "logs/selfplay_records"


def _resolve_record_dir():
    if os.path.isdir(RECORD_DIR):
        return RECORD_DIR
    if os.path.basename(os.getcwd()) == "selfplay_records":
        return "."
    return RECORD_DIR


def _find_record_file(iteration, record_dir):
    pattern = f"selfplay_*_iter_{iteration:04d}.json"
    candidates = glob.glob(os.path.join(record_dir, pattern))
    if not candidates:
        raise FileNotFoundError(f"No record file for iteration {iteration} in {record_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def load_records(iteration, record_dir):
    path = _find_record_file(iteration, record_dir)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", []), data.get("meta", {}), path


def _render_board_from_actions(record):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from env.env import GreatKingdomEnv

    size = int(record.get("board_size", 0))
    center_wall = bool(record.get("center_wall", True))
    komi = float(record.get("komi", 0))
    actions = record.get("actions", [])
    if size <= 0:
        raise ValueError("Invalid board_size")

    env = GreatKingdomEnv(board_size=size, center_wall=center_wall, komi=komi)
    for action in actions:
        env.step(int(action))

    board = env.board
    cols = [chr(ord('A') + i) for i in range(size)]
    print("   " + " ".join(cols))
    for r in range(size):
        row_label = str(r + 1).rjust(2)
        row = []
        for c in range(size):
            val = board[r, c]
            if val == 1:
                row.append("X")
            elif val == 2:
                row.append("O")
            elif val == 3:
                row.append("#")
            else:
                row.append(".")
        print(f"{row_label} " + " ".join(row))


def _render_board_from_positions(record):
    size = int(record.get("board_size", 0))
    board = [["." for _ in range(size)] for _ in range(size)]
    if record.get("center_wall") and size > 0:
        mid = size // 2
        board[mid][mid] = "#"
    for r, c in record.get("black", []):
        board[int(r)][int(c)] = "X"
    for r, c in record.get("white", []):
        board[int(r)][int(c)] = "O"

    cols = [chr(ord('A') + i) for i in range(size)]
    print("   " + " ".join(cols))
    for r in range(size):
        row_label = str(r + 1).rjust(2)
        print(f"{row_label} " + " ".join(board[r]))


def show_game(iteration, index, record_dir):
    records, meta, path = load_records(iteration, record_dir)
    if index < 1 or index > len(records):
        raise IndexError(f"index out of range: 1..{len(records)}")
    record = records[index - 1]
    print(f"File: {path}")
    if meta:
        print(
            f"Iteration: {meta.get('iteration')} | Games: {meta.get('games')} | Record interval: {meta.get('record_interval')}"
        )
    print(
        f"Winner: {record.get('winner_label')} | Win type: {record.get('win_type')} | Moves: {record.get('moves')} | Territory diff: {record.get('territory_diff')}"
    )
    if record.get("actions") is not None:
        _render_board_from_actions(record)
    else:
        _render_board_from_positions(record)


def _parse_input(line):
    line = line.strip()
    if not line:
        return None
    if line.lower() in ("q", "quit", "exit"):
        return "quit"
    line = line.strip("()")
    parts = [p for p in line.replace(",", " ").split() if p]
    if len(parts) != 2:
        raise ValueError("Please enter: <iteration> <index>  (e.g. 71 25)")
    return int(parts[0]), int(parts[1])


def main():
    record_dir = _resolve_record_dir()
    print(f"Record dir: {os.path.abspath(record_dir)}")
    print("Enter: <iteration> <index> (e.g. 71 25). Type 'q' to quit.")
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        result = _parse_input(line)
        if result is None:
            continue
        if result == "quit":
            break
        iteration, index = result
        try:
            show_game(iteration, index, record_dir)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
