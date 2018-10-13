from dqn import utils
from collections import namedtuple
import numpy as np
import torch
import math
import random
from box import Box

DEVICE = 'cpu'
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
CAPACITY = 10000


class DQNAgent(object):
    def __init__(self):
        # our dqn agent that we want to optimize
        self.policy_dqn = utils.DeepQNetwork()
        # the frozen target used to compute td target
        self.target_dqn = utils.DeepQNetwork()
        # init with policy dqn params
        self.update_target_dqn()

        #replay memory
        self.memory = utils.ReplayMemory(CAPACITY)
        # state transformers -- Phi in the paper
        self.stransformer = utils.StateTransformer()

        #optimizer
        self.optimizer = torch.optim.RMSprop(self.policy_dqn.parameters())

        # global step
        self.steps_done = 0
        self.num_episodes = 0
        self.episode_i = 0

    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()

    def select_action(self, state, epsilon, nactions):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        if np.random.rand() < eps_threshold:
            action = random.choice(range(nactions))
            action = torch.tensor([action]).view(-1, 1)
        else:
            with torch.no_grad():
                action = np.argmax(self.policy_dqn.forward(state)).view(-1, 1)
        return action

    def update_parameters(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = utils.Transition(*zip(*self.memory.sample(BATCH_SIZE)))
        # we need to compute the targets
        state_batch = torch.cat(batch.state)  #Bx4x64x64
        nstate_batch = torch.cat(batch.next_state)  #Bx4x64x64
        action_batch = torch.cat(batch.action)  #Bx1
        reward_batch = torch.cat(batch.reward)  #Bx1
        done_batch = 1.0 - torch.cat(batch.done)

        # Q(s,a)
        qsa = self.policy_dqn.forward(state_batch)  # BX4
        qsa = qsa.gather(1, action_batch)  #Bx1
        # TD target r + Q(s',argmax(Q(s',a))
        qnsa = self.target_dqn.forward(state_batch)
        qnsa = qnsa.max(1)[0]
        target = reward_batch + GAMMA * qnsa * done_batch.float()
        target = target.detach().view(-1, 1)

        # compute loss
        loss = torch.nn.functional.smooth_l1_loss(qsa, target)

        # clean up grads
        self.optimizer.zero_grad()
        # compute gradients
        loss.backward()
        # clip the gradient
        for params in self.policy_dqn.parameters():
            params.grad.data.clamp_(-1, 1)
        # one step of gradient descent
        self.optimizer.step()

    def train_one_episode(self, env):
        self.stransformer.reset()
        state = env.reset()
        state = self.stransformer.transform(state)
        episode = Box(steps=0, reward=0)
        while True:
            env.render()
            action = self.select_action(state, epsilon=0.1, nactions=4)
            nstate, reward, done, info = env.step(action)

            # add to episode stats
            episode.steps += 1
            episode.reward += reward
            last_reward = reward

            # convert to tensor and set nstate to None if end ep
            reward = torch.tensor([reward], device=DEVICE)
            done = torch.tensor([done], device=DEVICE)
            nstate = self.stransformer.transform(nstate)

            # push to memory
            self.memory.push(state, action, reward, nstate, done)

            # compute the loss
            self.update_parameters()

            # update target dqn from time to time
            if self.steps_done % TARGET_UPDATE == 0:
                self.update_target_dqn()

            print(
                "\rStep {} @ Episode {}/{} ({})".format(
                    episode.steps, self.episode_i + 1, self.num_episodes,
                    last_reward),
                end="")
            if done:
                break
            state = nstate

        return episode

    def train(self, env, num_episodes=50):
        self.steps_done = 0
        self.num_episodes = num_episodes
        episodes = []
        for i in range(num_episodes):
            self.episode_i = i
            episodes.append(self.train_one_episode(env))
        return episodes
