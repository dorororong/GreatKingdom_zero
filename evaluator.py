import numpy as np
import torch

from symmetry import canonicalize_encoded_state, invert_action_probs
from eval_cache import EvalCache


class Evaluator:
    """Canonicalized NN evaluator with cache."""

    def __init__(self, network, board_size, cache_max_entries=50000):
        self.network = network
        self.board_size = board_size
        self.net_version = 0
        self.cache = EvalCache(max_entries=cache_max_entries)

    def set_net_version(self, net_version):
        if int(net_version) != int(self.net_version):
            self.net_version = int(net_version)
            self.cache.clear()

    def eval_encoded(self, encoded_state):
        canon_state, transform_id, key = canonicalize_encoded_state(encoded_state)
        cache_key = (self.net_version, key)
        cached = self.cache.get(cache_key)
        if cached is not None:
            policy_canon, value, aux = cached
            policy = invert_action_probs(policy_canon, self.board_size, transform_id)
            return policy, value, aux, True

        device = next(self.network.parameters()).device
        x = torch.from_numpy(canon_state).unsqueeze(0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    policy, value, ownership, win_logit, win_type_logit = self.network(x)
            else:
                policy, value, ownership, win_logit, win_type_logit = self.network(x)

        policy = torch.exp(policy).float().squeeze(0).cpu().numpy()
        value = float(value.squeeze(0).cpu().numpy())
        aux = {
            "ownership": torch.sigmoid(ownership).squeeze(0).cpu().numpy(),
            "win_prob": torch.sigmoid(win_logit).squeeze(0).cpu().numpy().item(),
            "win_type_prob": torch.sigmoid(win_type_logit).squeeze(0).cpu().numpy().item(),
        }

        self.cache.store(cache_key, (policy, value, aux))
        policy = invert_action_probs(policy, self.board_size, transform_id)
        return policy, value, aux, False
