from collections import namedtuple
import random
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
