import os
from multiprocessing import Pool
import numpy as np
from box import Box
import torch
from src.common.base_trainer import BasePersonalTrainer
from src.a2c.utils import compute_discount_reward_with_done
import torch.nn.functional as F
ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gym
import torch.nn.functional as F
from baselines.run import make_vec_env, get_env_type


class PersonalTrainer(BasePersonalTrainer):
    def parse_rollout(self, rollout):
        states = torch.from_numpy(np.stack(
            [roll[0] for roll in rollout])).to(DEVICE).float().permute(
                1, 0, 2)
        actions = torch.from_numpy(np.stack(
            [roll[1] for roll in rollout])).to(DEVICE).float().permute(1, 0)
        nstates = torch.from_numpy(np.stack(
            [roll[2] for roll in rollout])).to(DEVICE).float().permute(
                1, 0, 2)
        rewards = torch.from_numpy(np.stack(
            [roll[3] for roll in rollout])).to(DEVICE).float().permute(1, 0)
        done = torch.from_numpy(np.stack(
            [roll[4] * 1.0 for roll in rollout])).to(DEVICE).float().permute(
                1, 0)
        values = torch.from_numpy(np.stack(
            [roll[5] for roll in rollout])).to(DEVICE).float().permute(
                1, 0, 2).squeeze(-1)
        return states, actions, nstates, rewards, done, values

    def compute_returns(self, rewards, dones, values):
        # If epsiodes are note done, use V(s+1) as guess for G(s+1)
        returns = []
        for reward, done, value in zip(rewards, dones, values):
            reward = reward.tolist()
            done = done.tolist()
            value = value[-1].tolist()
            if done[-1] == 0:
                ret = compute_discount_reward_with_done(
                    reward + [value], done + [1], gamma=self.agent.gamma)[:-1]
            else:
                ret = compute_discount_reward_with_done(
                    reward, done, gamma=self.agent.gamma)
            returns.append(ret)
        return torch.from_numpy(np.stack(returns)).float()

    def update_agent(self, rollout):
        """
        Perform one step of gradient ascent.
        Use the Returns Gt directly. MC way.
        """
        states, actions, nstates, rewards, dones, values = self.parse_rollout(
            rollout)
        returns = self.compute_returns(rewards, dones, values).view(-1, 1)
        logits, values = self.agent.policy(
            states.reshape(-1, states.shape[-1]))
        advantages = returns.reshape(-1, 1) - values.reshape(-1, 1)
        cross_entropy = F.cross_entropy(logits, actions.reshape(-1).long())
        wcross_entropy = cross_entropy * advantages
        # policy loss
        policy_loss = wcross_entropy.sum()
        # value loss
        value_loss = F.smooth_l1_loss(values, returns)
        # loss
        loss = value_loss + policy_loss

        # additional loss to encourage exploration
        self.optimizer.zero_grad()
        loss.backward()

        # clip the gradient
        for params in self.agent.policy.parameters():
            params.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def rollout(self, env, states):
        data = []
        for i in range(self.agent.nsteps):
            actions, values = self.agent.select_action(states)
            nstates, rewards, done, _ = env.step(actions)
            nstates = self.agent.stransformer.transform(nstates)
            data.append([states, actions, nstates, rewards, done, values])
            states = nstates
        return data, states

    def train(self, total_steps=10, nenv=2):
        env_type, env_id = get_env_type('CartPole-v0')
        env = make_vec_env(
            env_id, env_type, nenv, self.agent.seed, reward_scale=1)
        states = env.reset()
        states = self.agent.stransformer.transform(states)
        for i in range(total_steps):
            rollout, states = self.rollout(env, states)
            return rollout, states
            update_agent(rollout)
