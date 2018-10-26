import os

import numpy as np

import torch
from src.common.base_agent import BaseAgent
from src.reinforce.utils import PolicyNetwork, StateTransformer

ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(BaseAgent):
    def setup_default_experiment(self):
        self._setup_default_experiment()
        self.env_name = 'CartPole-v0'
        self.nactions = 2

    def get_policy(self):
        return PolicyNetwork().to(DEVICE)

    def get_state_transformer(self):
        return StateTransformer()

    def select_action(self, state, **kwargs):
        action_prob = np.array(self.policy(state).detach()[0])
        action = np.random.choice(range(self.nactions), p=action_prob)
        return action
