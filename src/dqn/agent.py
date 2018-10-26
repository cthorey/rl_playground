from src.common.base_agent import BaseAgent
import os
import random

import numpy as np

import torch
from box import Box
from src.dqn import utils
ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(BaseAgent):
    def get_policy(self):
        return utils.DeepQNetwork().to(DEVICE)

    def get_state_transformer(self):
        return utils.StateTransformer()

    def setup_default_experiment(self):
        # general
        self.batch_size = 32
        self.gamma = 0.999
        self.double_dqn = False

        # epsilon decay
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay = 250000
        self.target_update = 10000

        # memory
        self.capacity = 1000000
        self.min_memory_size = 50000

        # training
        self.update_freq = 4
        self.optimizer = 'Adam'
        self.optimizer_config = Box({'lr': 1e-5})

        # environmental
        self.env_name = 'BreakoutDeterministic-v4'
        self.nactions = 4

        # Global step
        self.steps_done = 0
        self.num_episodes = 0
        self.episodes_done = 0
        self.best_reward = 0

        # chekpoints
        self.gif_size = (128, 128)
        self.checkpoint = None

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = random.choice(range(self.nactions))
        else:
            state = state.float() / 255.0
            state = state.to(DEVICE)
            action = self.policy_dqn(state).detach().max(1)[1].view(1, 1)
        action = torch.tensor(action, device='cpu').view(1, -1)
        return action
