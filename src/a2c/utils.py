import torch
import torch.nn.functional as F
import numpy as np
import numpy as np
import torch
import torch.nn.functional as F
from baselines.run import make_vec_env, get_env_type


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def compute_discount_reward_with_done(rewards, done, gamma):
    rewards = np.array(rewards)
    done = np.array(done)
    end = np.argwhere(np.array(done)).ravel()
    returns = np.zeros(np.array(rewards).shape[0])
    idx = 0
    for nidx in end:
        returns[idx:nidx + 1] = compute_discount_reward(
            rewards[idx:nidx + 1], gamma)
        idx = nidx + 1
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


class Rollout(object):
    def __init__(self, agent):
        self.agent = agent
        env_type, env_id = get_env_type(self.agent.env_name)
        self.nenv = self.agent.nenv
        self.env = make_vec_env(
            env_id, env_type, self.nenv, self.agent.seed, reward_scale=1)
        self.reset()

    @property
    def states(self):
        return np.stack([roll[0] for roll in self.rollout]).transpose(1, 0, 2)

    @property
    def actions(self):
        return np.stack(
            [roll[1] for roll in self.rollout]).astype(int).transpose(1, 0)

    @property
    def values(self):
        return np.stack([roll[2] for roll in self.rollout]).transpose(
            1, 0, 2).squeeze(-1)

    @property
    def nstates(self):
        return np.stack([roll[3] for roll in self.rollout]).transpose(1, 0, 2)

    @property
    def nvalues(self):
        return np.stack([roll[4] for roll in self.rollout]).transpose(
            1, 0, 2).squeeze(-1)

    @property
    def rewards(self):
        return np.stack([roll[5] for roll in self.rollout]).transpose(1, 0)

    @property
    def dones(self):
        return np.stack([roll[6] for roll in self.rollout]).transpose(1, 0)

    @property
    def explained_variance(self):
        return explained_variance(self.values.ravel(), self.returns.ravel())

    @property
    def returns(self):
        # If epsiodes are note done, use V(s+1) as guess for G(s+1)
        returns = []
        for reward, done, nvalue in zip(self.rewards, self.dones,
                                        self.nvalues):
            reward = list(reward)
            done = list(done)
            nvalue = nvalue[-1]
            if done[-1] == 0:
                ret = compute_discount_reward_with_done(
                    reward + [nvalue], done + [1], gamma=self.agent.gamma)[:-1]
            else:
                ret = compute_discount_reward_with_done(
                    reward, done, gamma=self.agent.gamma)
            returns.append(ret)
        return np.stack(returns)

    @property
    def advantages(self):
        return self.returns - self.values

    def reset(self):
        self.current_states = self.agent.stransformer.transform(
            self.env.reset())
        self.rollout = []
        self.steps_done = 0
        self._cumulative_reward = 0
        self.cumulative_reward = [0]
        self.rollout_done = 0

    def run(self):
        self.rollout = []
        for i in range(self.agent.nsteps):
            actions, values = self.agent.select_action(self.current_states)
            nstates, rewards, done, _ = self.env.step(actions)
            _, nvalues = self.agent.select_action(nstates)
            # keep track of the reward for one env
            self._cumulative_reward += rewards[0]
            if done[0] == 1:
                self.cumulative_reward.append(self._cumulative_reward)
                self._cumulative_reward = 0
            nstates = self.agent.stransformer.transform(nstates)
            self.rollout.append([
                self.current_states, actions, values, nstates, nvalues,
                rewards, done
            ])
            self.current_states = nstates
        self.agent.steps_done += self.nenv * self.agent.nsteps
        self.rollout_done += 1


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
