from dqn import utils
from collections import namedtuple
import numpy as np
import torch
import torch.nn.functional as F
import math
import random
from box import Box
import fire
from tqdm import tqdm
import json
import gym
from tensorboardX import SummaryWriter
import json
import os

ROOT_DIR = os.environ['ROOT_DIR']
CHECKPOINT_NAME = 'checkpoint.pth.tar'
CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'dqn', CHECKPOINT_NAME)
HISTORY_PATH = os.path.join(ROOT_DIR, 'history.json')
BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
CAPACITY = 10000
NACTIONS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent(object):
    def __init__(self):
        # our dqn agent that we want to optimize
        self.policy_dqn = utils.DeepQNetwork().to(DEVICE)
        # the frozen target used to compute td target
        self.target_dqn = utils.DeepQNetwork().to(DEVICE)
        # init with policy dqn params
        self.update_target_dqn()

        #replay memory
        self.memory = utils.ReplayMemory(CAPACITY)
        # state transformers -- Phi in the paper
        self.stransformer = utils.StateTransformer()

        #optimizer
        self.optimizer = torch.optim.RMSprop(self.policy_dqn.parameters())

        # writer
        self.writer = SummaryWriter()

        # global step
        self.steps_done = 0
        self.num_episodes = 0
        self.episode_i = 0

    def save_checkpoint(self):
        state = {
            'steps_done': self.steps_done,
            'episodes': self.episode_i,
            'state_dict': self.policy_dqn.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, CHECKPOINT_PATH)

    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        self.target_dqn.eval()

    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        if np.random.rand() < eps_threshold:
            action = random.choice(range(NACTIONS))
        else:
            with torch.no_grad():
                action = np.argmax(self.policy_dqn(state))
        action = torch.tensor(action, device=DEVICE).view(1, -1)
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
        done_batch = 1.0 - torch.cat(batch.done)  #Bx1

        # Q(s,a)
        qsa = self.policy_dqn(state_batch)  # BX4
        qsa = qsa.gather(1, action_batch)  #Bx1
        # TD target r + Q(s',argmax(Q(s',a))
        qnsa = self.target_dqn(nstate_batch)
        qnsa = qnsa.max(1)[0].detach()
        target = reward_batch + GAMMA * qnsa * done_batch.float()
        target = target.view(-1, 1)

        # compute loss
        loss = F.smooth_l1_loss(qsa, target)
        self.writer.add_scalar('data/loss', loss, self.steps_done)

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
            action = self.select_action(state)
            nstate, reward, done, info = env.step(action)
            self.writer.add_scalar('data/reward', reward, self.steps_done)

            # add to episode stats
            episode.steps += 1
            episode.reward += reward

            # convert to tensor and set nstate to None if end ep
            reward = torch.tensor([reward], device=DEVICE)
            done = torch.tensor([done], device=DEVICE)
            nstate = self.stransformer.transform(nstate)

            # push to memory
            self.memory.push(state, action, reward, nstate, done)

            # compute the loss
            self.update_parameters()

            if done:
                break
            state = nstate
            self.steps_done += 1

        return episode

    def train(self, env, num_episodes=50, resume=False):
        self.steps_done = 0
        if resume:
            checkpoint = Box(torch.load(CHECKPOINT_PATH))
            self.steps_done = checkpoint.steps_done
            self.episode_i = checkpoint.episodes
            self.policy_dqn.load_state_dict(checkpoint.state_dict)
            self.update_target_dqn()
            self.optimizer.load_state_dict(checkpoint.optimizer)
            episodes = json.load(open(HISTORY_PATH, 'r'))
            print("=> loaded checkpoint (steps_done {}/ episode {})".format(
                checkpoint.steps_done, checkpoint.episodes))

        num_episodes = num_episodes + self.episode_i + 1
        self.num_episodes = num_episodes
        episodes = []
        for i in tqdm(range(num_episodes)):
            self.episode_i += 1
            summary = self.train_one_episode(env)
            episodes.append(summary)
            self.save_checkpoint()
            json.dump(episodes, open(HISTORY_PATH, 'w+'))
            self.writer.add_scalars('data/episodes', summary, self.episode_i)
            # update target dqn from time to time
            if self.episode_i % TARGET_UPDATE == 0:
                self.update_target_dqn()
            self.writer.export_scalars_to_json("./all_scalars.json")
        self.writer.close()

        return episodes


def make_agent(num_episodes, resume=False):
    agent = DQNAgent()
    env = gym.envs.make("Breakout-v0")
    agent.train(env, num_episodes=num_episodes, resume=resume)


if __name__ == '__main__':
    fire.Fire(make_agent)
