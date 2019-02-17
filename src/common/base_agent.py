import json
import os
import random
import sys

import gym
import numpy as np

import torch
from box import Box
from PIL import Image
import imageio
ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseAgent(object):
    def __init__(self, agent_name, environment_name, expname=None):
        # agent_name
        self.agent_name = agent_name
        self.env_name = environment_name
        # common seutp
        self.common_setup(expname)

    def common_setup(self, expname):
        self.setup_approximator()
        self.setup_foldertree()
        self.setup_config()
        # reload previous
        if expname is not None:
            self.load_experiment(expname)

    def setup_approximator(self):
        raise NotImplementedError

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            print('Updating {}: {}'.format(key, value), file=sys.stderr)
            if key not in self.__dict__:
                raise ValueError('Parameter {} does not exist'.format(key))
            setattr(self, key, value)

    def load_exp_config(self, expname=None):
        """
        Loads configuration file for experiment
        """
        if expname is None:
            expname = self.expname
        expfile = os.path.join(self.agent_folder,
                               '{}_experiment.json'.format(expname))
        experiment = json.load(open(expfile, 'r'))
        return Box(experiment)

    def load_experiment(self, expname, prefix='best'):
        experiment = self.load_exp_config(expname)
        self.__dict__.update(experiment)
        checkpoint_path = os.path.join(self.agent_folder,
                                       '{}_{}.pth.tar'.format(expname, prefix))
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                'Cannot find the checkoint in {}'.format(checkpoint_path))
        self.checkpoint = Box(torch.load(checkpoint_path, map_location=DEVICE))
        self.steps_done = self.checkpoint.steps_done
        self.best_reward = self.checkpoint.get("best_reward", 0)
        self.episodes_done = self.checkpoint.episodes_done
        self.policy.load_state_dict(self.checkpoint.state_dict)
        print(
            "=> loaded {} checkpoint (steps_done {}/ episode {} / reward {})".
            format(prefix, self.steps_done, self.episodes_done,
                   self.best_reward),
            file=sys.stderr)

    def setup_config(self):
        raise NotImplementedError

    def setup_foldertree(self):
        """
        Create the folder tree of that specific model training
        """
        self.agent_folder = os.path.join(ROOT_DIR, 'models', self.agent_name)
        if not os.path.isdir(self.agent_folder):
            os.makedirs(self.agent_folder)

    def select_action(self, state, epsilon):
        raise NotImplementedError

    def play_one_episode(self, render=False, create_gif=False):
        env = gym.envs.make(self.env_name)
        frames = []
        state = env.reset()
        stats = Box(steps=0, reward=0)
        while 1:
            if render:
                env.render()
            if create_gif:
                frames.append(self.get_screen(env))
            action = self.select_action(state.to(DEVICE), epsilon=0.0)
            nstate, reward, done, info = env.step(action)
            nstate = self.stransformer.transform(nstate)
            stats.steps += 1
            stats.reward += reward
            if done:
                break
            state = nstate
        if create_gif:
            self.generate_giff(frames)
        return stats

    def get_screen(self, env):
        screen = env.render(mode='rgb_array')
        img = Image.fromarray(screen).resize(self.gif_size)
        return np.array(img).astype('uint8')

    def generate_giff(self, frames):
        expname = self.expname if hasattr(self, 'expname') else 'default'
        gifname = '{}_{}.gif'.format(expname, self.best_reward)
        fpath = os.path.join(self.agent_folder, gifname)
        frames = np.stack(frames)
        imageio.mimsave(fpath, frames, duration=1 / 30)
