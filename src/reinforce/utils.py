import torch
import torch.nn.functional as F
import numpy as np


def convert_to_onehot(actions, nactions):
    out = np.zeros((actions.shape[0], nactions))
    out[range(len(out)), actions.astype('int')] = 1
    return out


def compute_discount_reward(rewards, gamma):
    """
    Args:
        rewards (NX1)
    """
    N = rewards.shape[0]
    triangle = np.tri(N, N)
    powers = np.cumsum(triangle, axis=0) - 1
    gammas = gamma**powers
    returns = gammas * triangle * rewards
    return np.sum(returns, axis=0).T


class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.f1 = torch.nn.Linear(4, 16)
        self.f2 = torch.nn.Linear(16, 32)
        self.logits = torch.nn.Linear(32, 2)

    def forward(self, X):
        X = F.relu(self.f1(X))
        X = F.relu(self.f2(X))
        return self.logits(X)
