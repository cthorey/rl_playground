from collections import namedtuple
from torch import nn
import torch
import random
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StateTransformer(object):
    """
    The input of our Qvalue is the concatenation of the last four frames.
    """

    def __init__(self):
        self.reset()
        self.transforms = transforms.Compose([
            transforms.Resize(size=(84, 84)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def reset(self):
        self.state = None

    def preprocessing(self, state):
        """
        Input: 4X210X160X3
        Output: 4X84X84
        """
        state = state[210 - 170:200]
        out = self.transforms(Image.fromarray(state))
        out /= 255.
        return out

    def transform(self, state):
        state = self.preprocessing(state)
        if self.state is None:
            nstate = np.vstack([state for _ in range(4)])
        else:
            nstate = np.vstack([self.state[1:], state])
        nstate = torch.from_numpy(nstate).to(DEVICE)
        self.state = nstate
        return nstate.unsqueeze(0)


class ReplayMemory(object):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.idx = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # compute phi(state)
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done)
        self.memory[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return [random.choice(self.memory) for _ in range(batch_size)]


class DeepQNetwork(torch.nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.fc = torch.nn.Linear(5184, 256)
        self.head = torch.nn.Linear(256, 4)

    def forward(self, X):
        """
        Architecture of DQN
        """
        X = F.relu(self.bn1(self.conv1(X)))
        X = F.relu(self.bn2(self.conv2(X)))
        X = X.view(-1, 5184)
        X = self.fc(X)
        return self.head(X)


def plot_reward(eps):
    df = pd.DataFrame(eps)
    df['total_steps'] = df.steps.cumsum()
    df.plot(x='total_steps', y='reward')
