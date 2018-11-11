import torch
import torch.nn.functional as F
import numpy as np
import gym


def compute_discount_reward_with_done(rewards, done, gamma):
    rewards = np.array(rewards)
    done = np.array(done)
    end = np.argwhere(np.array(done)).ravel()
    returns = np.zeros(np.array(rewards).shape[0])
    idx = 0
    for nidx in end:
        returns[idx:nidx + 1] = compute_discount_reward(
            rewards[idx:nidx + 1], gamma)
        idx = nidx
    return returns


def compute_discount_reward(rewards, gamma):
    """
    Args:
        rewards (NX1)
    """
    rewards = np.expand_dims(np.array(rewards), 1)
    N = len(rewards)
    triangle = np.tri(N, N)
    powers = np.cumsum(triangle, axis=0) - 1
    gammas = gamma**powers
    returns = gammas * triangle * rewards
    return np.sum(returns, axis=0).T


class StateTransformer(object):
    """
    The input of our Qvalue is the concatenation of the last four frames.
    """

    def transform(self, state):
        return state


class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.f1 = torch.nn.Linear(4, 16)
        self.f2 = torch.nn.Linear(16, 32)
        self.logits = torch.nn.Linear(32, 2)
        self.values = torch.nn.Linear(32, 1)

    def forward(self, X):
        X = F.relu(self.f1(X))
        X = F.relu(self.f2(X))
        return self.logits(X), self.values(X)
