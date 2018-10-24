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
        stransformer_path = os.path.join(ROOT_DIR, 'models', 'linear_fa',
                                         'stransformer.pickle')
        if not os.path.isfile(stransformer_path):
            self.gen_transformer(stransformer_path)
        self.pipeline = pickle.load(open(stransformer_path, 'rb'))

    def gen_transformer(self, stransformer_path):
        scaler = sklearn.preprocessing.StandardScaler()
        featurizer = sklearn.pipeline.FeatureUnion(
            [("rbf1", RBFSampler(gamma=5.0, n_components=100)),
             ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
             ("rbf3", RBFSampler(gamma=1.0,
                                 n_components=100)), ("rbf4",
                                                      RBFSampler(
                                                          gamma=0.5,
                                                          n_components=100))])
        pipeline = Pipeline([('scaler', scaler), ('feat', featurizer)])

        env = gym.envs.make(ENV_NAME)
        obs = np.array([env.observation_space.sample() for x in range(10000)])
        pipeline.fit(obs)
        pickle.dump(pipeline, open(stransformer_path, 'wb+'))

    def transform(self, state):
        state = np.expand_dims(state, 0)
        state = self.pipeline.transform(state)
        state = torch.from_numpy(state).to('cpu').float()
        return state


class LinearApproximator(torch.nn.Module):
    """
    Simple linear approximator
    """

    def __init__(self):
        super(LinearApproximator, self).__init__()
        self.head = torch.nn.Linear(400, 3)

    def forward(self, X):
        """
        Architecture of DQN
        """
        return self.head(X)
