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
    def action_log_probs(self):
        return np.stack([roll[2] for roll in self.rollout]).transpose(1, 0)

    @property
    def values(self):
        return np.stack([roll[3] for roll in self.rollout]).transpose(
            1, 0, 2).squeeze(-1)

    @property
    def nstates(self):
        return np.stack([roll[4] for roll in self.rollout]).transpose(1, 0, 2)

    @property
    def nvalues(self):
        return np.stack([roll[5] for roll in self.rollout]).transpose(
            1, 0, 2).squeeze(-1)

    @property
    def rewards(self):
        return np.stack([roll[6] for roll in self.rollout]).transpose(1, 0)

    @property
    def dones(self):
        return np.stack([roll[7] for roll in self.rollout]).transpose(1, 0)

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

    def obs_to_state(self, obs):
        return torch.Tensor(obs).view(self.nenv, -1)

    def reset(self):
        self.current_obs = self.env.reset()
        self.rollout = []
        self.steps_done = 0
        self._cumulative_reward = 0
        self.cumulative_reward = [0]
        self.rollout_done = 0

    def run(self):
        self.rollout = []
        for i in range(self.agent.nsteps):
            # Choose action based on new state
            with torch.no_grad():
                states = self.obs_to_state(self.current_obs)
                actions, action_log_probs, values = self.agent.take_action(
                    states)

            # Take the action in the environment
            obs, rewards, done, _ = self.env.step(actions.tolist())

            # Evaluate V(s')
            with torch.no_grad():
                nstates = self.obs_to_state(obs)
                nvalues = self.agent.get_value(nstates)

            self.rollout.append([
                states, actions, action_log_probs, values, nstates, nvalues,
                rewards, done
            ])
            # update current obs
            self.current_obs = obs

            # keep track of the reward for one env
            self._cumulative_reward += rewards[0]
            if done[0] == 1:
                self.cumulative_reward.append(self._cumulative_reward)
                self._cumulative_reward = 0

        self.agent.steps_done += self.nenv * self.agent.nsteps
        self.rollout_done += 1
