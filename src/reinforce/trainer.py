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
        """
        rewards = np.array([d[-1] for d in data])
        returns = torch.Tensor(
            compute_discount_reward(rewards, self.agent.gamma))
        targets = torch.Tensor(np.array([d[1] for d in data])).long()
        states = torch.cat([d[0] for d in data])
        logits = self.agent.policy(states)
        cross_entropy = F.cross_entropy(logits, targets)
        wcross_entropy = cross_entropy * returns
        loss = wcross_entropy.sum()
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
            action = self.agent.select_action(state)
            # env magic
            nstate, reward, done, _ = env.step(action)
            nstate = self.agent.stransformer.transform(nstate)
            # add to episode stats
            episode.steps += 1
            episode.reward += reward
            self.agent.steps_done += 1
            data.append([state, action, reward])
            if done:
                break
            state = nstate

        # gradient ascent step
        self.update_agent(data)
        return episode
