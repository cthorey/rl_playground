from collections import namedtuple
from torch import nn
import random
from torchvision import transforms
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.idx = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        transition = Transition(*args)
        self.memory[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return [random.choice(self.memory) for _ in range(batch_size)]


class QValueFunction(nn.Module):
    """
    Architectures
    Input - 84X84X4

    """

    def __init__(self):
        super(QValueFunction, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.nl1 = nn.Relu()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.nl2 = nn.Relu()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.nl3 = nn.Relu()
        self.head = nn.Linear(448, 2)

    def preprocessing(self, X):
        X = X - X.mean(axis=-1, keepdims=True)
        return X.transpose(0, 3, 1, 2)

    def forward(self, X):
        return self.conv1(X)
