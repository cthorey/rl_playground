import json
import math
import os
import random
import sys
from collections import namedtuple

import gym
import numpy as np

import fire
import torch
import torch.nn.functional as F
from box import Box
from src.dqn import utils
from tensorboardX import SummaryWriter
from tqdm import tqdm

ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonalTrainer(object):
    def __init__(self, agent):
        self.agent = agent
        # the frozen target used to compute td target
        self.target_dqn = utils.DeepQNetwork().to(DEVICE)
        # init with policy dqn params
        self.update_target_dqn()

        #replay memory
        self.memory = utils.ReplayMemory(self.agent.capacity)

        #optimizer
        self.optimizer = getattr(torch.optim, self.agent.optimizer)(
            self.agent.policy_dqn.parameters(), **self.agent.optimizer_config)
        if self.agent.checkpoint is not None:
            self.optimizer.load_state_dict(self.agent.checkpoint.optimizer)

    def epsilon_decay(self, num_steps):
        return  self.agent.eps_end + (self.agent.eps_start - self.agent.eps_end) * \
            math.exp(-1. * num_steps / self.agent.eps_decay)

    def update_config(self, **kwargs):
        self.agent.update_config(**kwargs)

    @property
    def agent_name(self):
        return self.agent.agent_name

    @property
    def agent_folder(self):
        return self.agent.agent_folder

    @property
    def expname(self):
        return self.agent.expname

    @property
    def steps_done(self):
        return self.agent.steps_done

    @property
    def episodes_done(self):
        return self.agent.episodes_done

    def save(self, prefix):
        checkpoint_path = os.path.join(self.agent.agent_folder,
                                       '{}_{}.pth.tar'.format(
                                           self.expname, prefix))
        state = {
            'agent_name': self.agent_name,
            'expname': self.expname,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'state_dict': self.agent.policy_dqn.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.agent.policy_dqn.state_dict())
        self.target_dqn.eval()

    def replay(self):
        if len(self.memory) < self.agent.min_memory_size:
            return
        batch = utils.Transition(
            *zip(*self.memory.sample(self.agent.batch_size)))

        # we need to compute the targets
        state_batch = torch.cat(batch.state).float() / 255.
        state_batch = state_batch.to(DEVICE)

        #Bx4x64x64
        nstate_batch = torch.cat(batch.next_state).float() / 255.
        nstate_batch = nstate_batch.to(DEVICE)  #Bx4x64x64
        action_batch = torch.cat(batch.action).to(DEVICE).view(-1, 1)  #Bx1
        reward_batch = torch.cat(batch.reward).to(DEVICE).view(-1, 1)  #Bx1
        done_batch = 1.0 - torch.cat(batch.done).to(DEVICE).view(-1, 1)  #Bx1

        # Q(s,a)
        action_values = self.agent.policy_dqn(state_batch)  # BX4
        action_values = action_values.gather(1, action_batch)  #Bx1

        # TD target r + Q(s',argmax(Q(s',a))
        naction_values = self.target_dqn(nstate_batch).detach()  #Bx4
        naction_values_tmp = naction_values
        if self.agent.double_dqn:
            # use the policy instead of the target
            naction_values_tmp = self.agent.policy_dqn(nstate_batch).detach()
        idx_best_actions = naction_values_tmp.max(1)[-1].view(-1, 1)  #Bx1
        naction_values = naction_values.gather(1, idx_best_actions)  #Bx1
        td_target = reward_batch + self.agent.gamma * naction_values * done_batch.float(
        )
        td_target = td_target.view(-1, 1)

        # compute loss
        loss = F.smooth_l1_loss(action_values, td_target)
        if hasattr(self, 'writer'):
            self.writer.add_scalar('data/loss', loss, self.steps_done)

        # clean up grads
        self.optimizer.zero_grad()
        # compute gradients
        loss.backward()
        # clip the gradient
        for params in self.agent.policy_dqn.parameters():
            params.grad.data.clamp_(-1, 1)
        # one step of gradient descent
        self.optimizer.step()

    def train_one_episode(self, env):
        self.agent.stransformer.reset()
        state = env.reset()
        state = self.agent.stransformer.transform(state)
        episode = Box(steps=0, reward=0)
        life = 5
        while True:
            epsilon = self.epsilon_decay(self.steps_done)
            if hasattr(self, 'writer'):
                self.writer.add_scalar('data/epsilon', epsilon,
                                       self.steps_done)
            action = self.agent.select_action(state.to(DEVICE), epsilon)
            nstate, reward, done, info = env.step(action)

            # add terminal state when the agent lose a life
            nlife = info['ale.lives']
            terminal = True if life != nlife else done
            life = nlife

            # add to episode stats
            episode.steps += 1
            episode.reward += reward
            self.agent.steps_done += 1

            # convert to tensor and set nstate to None if end ep
            reward = torch.tensor([reward], device='cpu')
            terminal = torch.tensor([terminal], device='cpu')
            nstate = self.agent.stransformer.transform(nstate)

            # push to memory
            self.memory.push(state, action, reward, nstate, terminal)

            # replay every four frames
            if self.steps_done % 4 == 0:
                self.replay()

            # update target netwrook
            if self.steps_done % self.agent.target_update == 0:
                self.update_target_dqn()

            if done:
                break
            state = nstate

        return episode

    def train(self, num_episodes=50):
        env = gym.envs.make(self.agent.env_name)
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.agent_folder, self.expname))
        self.current_best_reward = 0
        rewards = []
        for i in tqdm(range(num_episodes), file=sys.stderr):
            summary = self.train_one_episode(env)
            if summary.reward >= self.current_best_reward:
                self.save('best')
                self.current_best_reward = summary.reward
            rewards.append(summary.reward)

            self.writer.add_scalar('nb_steps', self.steps_done,
                                   self.episodes_done)
            self.writer.add_scalar('reward', summary.reward, self.steps_done)
            self.writer.add_scalar('100_reward', np.mean(rewards[-100:]),
                                   self.steps_done)
            self.writer.add_scalar('episodes_done', self.episodes_done,
                                   self.steps_done)
            history_path = os.path.join(self.agent.agent_folder,
                                        'history.json'.format(self.expname))
            self.writer.export_scalars_to_json(history_path)
            self.agent.episodes_done += 1
            self.save('current')

        self.writer.close()
