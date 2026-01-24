def winner_id_from_step(info, reward, current_player):
    """
    Determine winner id from env.step outputs.

    Returns:
        1 for Black, 2 for White, 0 for Draw/unknown.
    """
    if isinstance(info, dict) and "winner" in info:
        if info["winner"] == "Black":
            return 1
        if info["winner"] == "White":
            return 2
        return 0

    if reward == 0:
        return 0
    if reward > 0:
        return current_player
    return 2 if current_player == 1 else 1


def winner_label_from_step(info, reward, current_player):
    """Return winner label ('Black'/'White'/'Draw') from env.step outputs."""
    winner_id = winner_id_from_step(info, reward, current_player)
    if winner_id == 1:
        return "Black"
    if winner_id == 2:
        return "White"
    return "Draw"
