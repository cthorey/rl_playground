import math
import os
import sys

import gym
import numpy as np

import torch
from box import Box
from tensorboardX import SummaryWriter
from tqdm import tqdm

ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasePersonalTrainer(object):
    def __init__(self, agent):
        self.agent = agent

        #optimizer
        self.optimizer = getattr(torch.optim, self.agent.optimizer)(
            self.agent.policy.parameters(), **self.agent.optimizer_config)

        # scheduler
        self.scheduler = None

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
    def best_reward(self):
        return self.agent.best_reward

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
            'best_reward': self.best_reward,
            'expname': self.expname,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'state_dict': self.agent.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def train_one_episode(self, env):
        self.agent.stransformer.reset()
        state = env.reset()
        state = self.agent.stransformer.transform(state)
        episode = Box(steps=0, reward=0)
        raise NotImplementedError
        return episode

    def train(self, num_episodes=50):
        env = gym.envs.make(self.agent.env_name)
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.agent_folder, self.expname))
        rewards = []
        for i in tqdm(range(num_episodes), file=sys.stderr):
            summary = self.train_one_episode(env)
            if summary.reward >= self.best_reward:
                self.save('best')
                self.agent.best_reward = summary.reward
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
