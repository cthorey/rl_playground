import json
import os
import sys
from src.common import approximator, utils
import numpy as np

import torch
from box import Box
from PIL import Image

import imageio

ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    def __init__(self, agent_name, env_name, expname=None, seed=569):
        # agent_name
        self.agent_name = agent_name
        self.env_name = env_name
        self.seed = seed
        self.setup_foldertree()
        self.setup_config()

        # the network for the policy
        self.setup_approxmator()

        # reload previous
        if expname is not None:
            self.load_experiment(expname)

    def setup_approxmator(self):
        env = utils.make_env(self.env_name)
        obs_shape = env.observation_space.shape
        action_space = env.action_space
        self.policy = approximator.Policy(
            obs_shape=obs_shape, action_space=action_space)

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            print('Updating {}: {}'.format(key, value))
            if key not in self.__dict__:
                raise ValueError('Parameter {} does not exist'.format(key))
            setattr(self, key, value)

    def load_previous_config(self, expname=None):
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
        experiment = self.load_previous_config(expname)
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
                   self.best_reward))

    def setup_config(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Global step
        self.gamma = 0.999
        self.steps_done = 0
        self.num_episodes = 0
        self.episodes_done = 0
        self.best_reward = 0

        # chekpoints
        self.gif_size = (420, 320)
        self.checkpoint = None

    def setup_foldertree(self):
        """
        Create the folder tree of that specific model training
        """
        self.agent_folder = os.path.join(ROOT_DIR, 'models', self.agent_name)
        if not os.path.isdir(self.agent_folder):
            os.makedirs(self.agent_folder)

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
