import numpy as np
import random


class MCTSNode:
    def __init__(self, state, parent=None, action=None, action_player=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.action_player = action_player
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = None

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        choices_weights = []
        for child in self.children:
            exploitation = child.wins / child.visits
            exploration = c_param * np.sqrt(2 * np.log(self.visits) / child.visits)
            choices_weights.append(exploitation + exploration)
        return self.children[int(np.argmax(choices_weights))]


class MCTS:
    def __init__(self, env, simulations_per_move=100):
        self.env = env
        self.sim_env = env.fast_copy()
        self.simulations_per_move = simulations_per_move

    def _set_env_state(self, env, state_snapshot):
        board, player, passes = state_snapshot
        env.board = board.copy()
        env.current_player = player
        env.consecutive_passes = passes

    def get_legal_actions(self, env):
        mask = env.get_legal_moves()
        return np.where(mask)[0]

    def _rollout(self, env):
        done = False
        info = {}
        while not done:
            legal = self.get_legal_actions(env)
            if len(legal) == 0:
                return 0, info
            action = random.choice(legal)
            _, _, done, info = env.step(action)
        return info.get("winner", 0), info

    def run(self, current_state_snapshot, opponent_last_action=None):
        root = MCTSNode(state=current_state_snapshot)
        for _ in range(self.simulations_per_move):
            node = root
            self._set_env_state(self.sim_env, current_state_snapshot)
            state = current_state_snapshot
            done = False
            winner = 0

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                _, _, done, info = self.sim_env.step(node.action)
                state = (
                    self.sim_env.board.copy(),
                    self.sim_env.current_player,
                    self.sim_env.consecutive_passes
                )
                if done:
                    winner = info.get("winner", 0)
                    break

            # Expansion
            if not done and node.untried_actions is None:
                node.untried_actions = self.get_legal_actions(self.sim_env).tolist()
            if not done and node.untried_actions:
                action = node.untried_actions.pop()
                _, _, done, info = self.sim_env.step(action)
                child = MCTSNode(state=(
                    self.sim_env.board.copy(),
                    self.sim_env.current_player,
                    self.sim_env.consecutive_passes
                ), parent=node, action=action, action_player=state[1])
                node.children.append(child)
                node = child
                if done:
                    winner = info.get("winner", 0)

            # Simulation
            if not done:
                winner, info = self._rollout(self.sim_env)

            # Backprop
            current = node
            while current is not None:
                current.visits += 1
                if winner != 0 and current.action_player == winner:
                    current.wins += 1
                current = current.parent

        # Choose best child
        if not root.children:
            return random.choice(self.get_legal_actions(self.sim_env))
        best = max(root.children, key=lambda c: c.visits)
        return best.action
