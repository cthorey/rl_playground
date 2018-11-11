import os

import numpy as np

import torch
from src.common.base_agent import BaseAgent
from src.a2c.utils import PolicyNetwork, StateTransformer
from torch.functional import F
ROOT_DIR = os.environ['ROOT_DIR']
from box import Box
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.distributions import Categorical


class Agent(BaseAgent):
    def setup_default_experiment(self):
        self._setup_default_experiment()
        self.env_name = 'CartPole-v0'
        self.nactions = 2
        self.optimizer = 'Adam'
        self.optimizer_config = Box({'lr': 1e-3})
        self.nsteps = 5  # step to run before update
        self.seed = 547

    def get_policy(self):
        return PolicyNetwork().to(DEVICE)

    def get_state_transformer(self):
        return StateTransformer()

    def select_action(self, state, return_value=True, **kwargs):
        state = torch.from_numpy(state).float().to('cpu')
        logits, values = self.policy(state)
        logits = logits.detach()
        actions = Categorical(logits=logits).sample()
        out = np.array(actions.tolist())
        if return_value:
            out = out, values.detach()
        return out
