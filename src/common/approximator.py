import torch.nn as nn
from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli


class MLPBase(nn.Module):
    def __init__(self, input_shape, hidden_size=64):
        super(MLPBase, self).__init__()
        self._hidden_size = hidden_size
        self.actor = nn.Sequential(
            nn.Linear(input_shape, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.critic = nn.Sequential(
            nn.Linear(input_shape, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1))

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, inputs):
        critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)
        return critic, hidden_actor


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None):
        super(Policy, self).__init__()
        if base is None:
            if len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0])

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, dist_entropy

    def get_value(self, inputs):
        value, _, = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
