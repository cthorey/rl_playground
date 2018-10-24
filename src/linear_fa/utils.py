import gym
import numpy as np

import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
import torch
import pickle
import os
ROOT_DIR = os.environ['ROOT_DIR']
ENV_NAME = "MountainCar-v0"


class StateTransformer(object):
    """
    The input of our Qvalue is the concatenation of the last four frames.
    """

    def __init__(self):
        scaler = sklearn.preprocessing.StandardScaler()
        featurizer = sklearn.pipeline.FeatureUnion(
            [("rbf1", RBFSampler(gamma=5.0, n_components=100)),
             ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
             ("rbf3", RBFSampler(gamma=1.0,
                                 n_components=100)), ("rbf4",
                                                      RBFSampler(
                                                          gamma=0.5,
                                                          n_components=100))])
        self.transforms = Pipeline([('scaler', scaler), ('feat', featurizer)])
        self.transforms.fit(self.get_random_observation())

    def get_random_observation(self):
        path = os.path.join(ROOT_DIR, 'models', 'linear_fa',
                            'observation.pickle')
        if not os.path.isfile(path):
            self.gen_random_observation()
        return pickle.load(open(path, 'rb'))

    def gen_random_observation(self):
        env = gym.envs.make(ENV_NAME)
        obs = np.array([env.observation_space.sample() for x in range(10000)])
        path = os.path.join(ROOT_DIR, 'models', 'linear_fa',
                            'observation.pickle')
        pickle.dump(obs, open(path, 'wb+'))

    def transform(self, state):
        state = np.expand_dims(state, 0)
        state = self.transforms.transform(state)
        state = torch.from_numpy(state).to('cpu').float()
        return state


class LinearApproximator(torch.nn.Module):
    """
    Simple linear approximator
    """

    def __init__(self):
        super(LinearApproximator, self).__init__()
        self.f1 = torch.nn.Linear(2, 400)
        self.head = torch.nn.Linear(400, 3)

    def forward(self, X):
        """
        Architecture of DQN
        """
        out = self.f1(X)
        return self.head(out)
