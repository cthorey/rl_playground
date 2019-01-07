import os

import numpy as np
from box import Box
import torch
from src.common.base_trainer import BasePersonalTrainer
from src.reinforce.utils import convert_to_onehot, compute_discount_reward
import torch.nn.functional as F
ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonalTrainer(BasePersonalTrainer):
    def update_agent(self, data):
        """
        Perform one step of gradient ascent.
        Use the Returns Gt directly. MC way.
        """
        rewards = np.array([d[-1] for d in data])
        returns = torch.Tensor(
            compute_discount_reward(rewards, self.agent.gamma))
        targets = torch.Tensor(np.array([d[1] for d in data])).long()
        states = torch.cat([d[0] for d in data])
        action_log_probs = torch.cat([d[2].view(-1) for d in data])
        loss = -action_log_probs * returns
        loss = loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_one_episode(self, env):
        """
        Train on episode
        """
        state = env.reset()
        state = self.agent.stransformer.transform(state)
        episode = Box(steps=0, reward=0)
        data = []
        # play one episode
        while 1:
            action, action_log_probs = self.agent.take_action(state)
            # env magic
            nstate, reward, done, _ = env.step(action.numpy())
            nstate = self.agent.stransformer.transform(nstate)
            # add to episode stats
            episode.steps += 1
            episode.reward += reward
            self.agent.steps_done += 1
            data.append([state, action, action_log_probs, reward])
            if done:
                break
            state = nstate
        # gradient ascent step
        self.update_agent(data)
        return episode
