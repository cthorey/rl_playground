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
    def train_one_episode(self, env):
        state = env.reset()
        state = self.agent.stransformer.transform(state)
        episode = Box(steps=0, reward=0)
        data = []
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

        # Compile the data into Gt and one hot action to compute the log
        rewards = np.array([d[-1] for d in data])
        returns = compute_discount_reward(rewards, self.agent.gamma)
        returns = torch.Tensor(returns)
        targets = torch.Tensor(np.array([d[1] for d in data])).long()
        states = torch.cat([d[0] for d in data])
        probs = self.agent.policy(states)
        negative_likelihoods = F.cross_entropy(probs, targets)
        weighted_negative_likelihoods = negative_likelihoods * returns

        loss = -weighted_negative_likelihoods.sum()

        # clean up grads
        self.optimizer.zero_grad()
        # compute gradients
        loss.backward()

        self.optimizer.step()
        return episode
