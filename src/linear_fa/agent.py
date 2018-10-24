from src.common.base_agent import BaseAgent
from src.linear_fa.utils import LinearApproximator, StateTransformer
import torch
from box import Box
import numpy as np
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "MountainCar-v0"


class Agent(BaseAgent):
    def get_policy(self):
        return LinearApproximator().to(DEVICE)

    def get_state_transformer(self):
        return StateTransformer(ENV_NAME)

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = [random.choice(range(self.nactions))]
        else:
            state = state.to(DEVICE)
            action = self.policy_dqn(state).detach().max(1)[1]
        action = np.array(action)
        return action[0]

    def setup_default_experiment(self):
        # training
        self.update_freq = 1
        self.optimizer = 'Adam'
        self.optimizer_config = Box({'lr': 1e-3})

        # epsilon decay
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay = 5000

        # environmental
        self.env_name = ENV_NAME
        self.nactions = 3
        self.gamma = 0.999

        # Global step
        self.steps_done = 0
        self.num_episodes = 0
        self.episodes_done = 0
        self.best_reward = -250

        # chekpoints
        self.checkpoint = None
