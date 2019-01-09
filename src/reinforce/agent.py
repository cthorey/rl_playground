import os

import torch
from src.common.base_agent import BaseAgent
from src.reinforce.utils import PolicyNetwork, StateTransformer
ROOT_DIR = os.environ['ROOT_DIR']
from box import Box
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(BaseAgent):
    def setup_default_experiment(self, env_name):
        self._setup_default_experiment(env_name=env_name)
        self.optimizer = 'Adam'
        self.optimizer_config = Box({'lr': 1e-3})

    def get_policy(self):
        return PolicyNetwork().to(DEVICE)

    def get_state_transformer(self):
        return StateTransformer()

    def take_action(self, state, **kwargs):
        logits = self.policy(state)
        dist = self.dist(logits=logits)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        return action, action_log_probs
