import os

import torch
import torch.nn.functional as F
from box import Box
from src.common.base_agent import BaseAgent

ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(ActorCritic, self).__init__()
        self.f1 = torch.nn.Linear(input_shape, 16)
        self.f2 = torch.nn.Linear(16, 32)
        self.actor = torch.nn.Linear(32, output_shape)
        self.critic = torch.nn.Linear(32, 1)

    def forward(self, X):
        X = F.relu(self.f1(X))
        X = F.relu(self.f2(X))
        return self.actor(X), self.critic(X)


class Agent(BaseAgent):
    def setup_default_experiment(self, env_name):
        self._setup_default_experiment(env_name=env_name)
        self.optimizer = 'RMSprop'
        self.optimizer_config = Box({'lr': 1e-4})
        self.nsteps = 1  # step to run before update
        self.seed = 400
        self.nenv = 1
        self.wloss = Box(value=.5, entropy=.01)
        self.max_grad_norm = 0.5

    def get_policy(self):
        print(DEVICE)
        return ActorCritic(
            input_shape=4, output_shape=self.num_outputs).to(DEVICE)

    def take_action(self, state):
        actor_logits, values = self.policy(state)
        dist = self.dist(logits=actor_logits)
        actions = dist.sample()
        action_log_probs = dist.log_prob(actions)
        return actions, action_log_probs, values

    def get_value(self, state):
        _, values = self.policy(state)
        return values

    def evaluate_actions(self, state, action):
        actor_logits, values = self.policy(state)
        dist = self.dist(actor_logits)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()
        return values, action_log_probs, dist_entropy

    def get_state_transformer(self):
        pass
