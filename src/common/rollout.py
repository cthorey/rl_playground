import numpy as np
import torch
import visdom
from src.common import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rollout(object):
    def __init__(self, agent, nsteps=10, nenv=1, seed=223, gamma=1):
        self.agent = agent
        self.gamma = gamma
        self.nsteps = nsteps
        self.nenv = nenv
        self.env = utils.make_env(self.agent.env_name, nenv, seed, debug=True)
        self.reset()

    @property
    def states(self):
        return torch.stack([roll[0] for roll in self.rollout])

    @property
    def actions(self):
        return torch.stack([roll[1] for roll in self.rollout])

    @property
    def action_log_probs(self):
        return torch.stack([roll[2] for roll in self.rollout])

    @property
    def values(self):
        return torch.stack([roll[3] for roll in self.rollout])

    @property
    def entropy(self):
        return torch.stack([roll[4] for roll in self.rollout])

    @property
    def nstates(self):
        return torch.stack([roll[5] for roll in self.rollout])

    @property
    def nvalues(self):
        return torch.stack([roll[6] for roll in self.rollout])

    @property
    def rewards(self):
        return torch.stack([roll[7] for roll in self.rollout])

    @property
    def dones(self):
        return torch.stack([roll[8] for roll in self.rollout])

    @property
    def returns(self):
        s = self.rewards.shape
        returns = torch.zeros(s[0] + 1, s[1], s[2])
        returns[-1] = self.nvalues[-1]
        for step in reversed(range(self.rewards.shape[0])):
            returns[step] = self.rewards[step]
            returns[step] += returns[step + 1] * self.gamma * (
                1 - self.dones[step])
        return returns[:-1]

    def obs_to_state(self, obs):
        return torch.Tensor(obs).view(self.nenv, -1).to('cpu')

    def reset(self):
        self.obs = self.env.reset()
        self.rollout = []
        self.steps_done = 0
        self._cumulative_reward = dict(zip(range(self.nenv), [0] * self.nenv))
        self.cumulative_reward = dict.fromkeys(range(self.nenv), [])
        self.rollout_done = 0

    def insert(self, states, actions, action_log_probs, values, nstates,
               nvalues, entropy, rewards, done):
        self.rollout.append([
            states.to('cpu'),
            actions.to('cpu'),
            action_log_probs.to('cpu'),
            values.to('cpu'),
            entropy.to('cpu'),
            nstates.to('cpu'),
            nvalues.to('cpu'),
            torch.Tensor(rewards).view(self.nenv, -1).to('cpu'),
            torch.Tensor(np.array(done) * 1.0).view(self.nenv, -1).to('cpu')
        ])

    def track_cumulative_reward(self, rewards, dones):
        # keep track of the reward for one env
        for idx in range(self.nenv):
            self._cumulative_reward[idx] += rewards[idx]
            if dones[idx]:
                self.cumulative_reward[idx].append(
                    self._cumulative_reward[idx])
                self._cumulative_reward[idx] = 0

    def run(self):
        self.rollout = []
        self.agent.policy.to('cpu')
        for i in range(self.nsteps):
            # Choose action based on new state
            with torch.no_grad():
                states = self.obs_to_state(self.obs)
                values, actions, action_log_probs, entropy = self.agent.policy.act(
                    states)

            # Take the action in the env
            nobs, rewards, dones, _ = self.env.step(
                np.array(actions.flatten()))
            # Evaluate V(s')
            with torch.no_grad():
                nstates = self.obs_to_state(nobs)
                nvalues = self.agent.policy.get_value(nstates)

            # add to rollout
            self.insert(states, actions, action_log_probs, values, nstates,
                        nvalues, entropy, rewards, dones)

            self.track_cumulative_reward(rewards, dones)

            # update obs
            self.obs = nobs if not dones[0] else self.env.reset()

        self.agent.steps_done += self.nenv * self.nsteps
        self.rollout_done += 1
        self.agent.policy.to(DEVICE)
