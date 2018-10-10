import itertools
import os
import random
import sys

import gym
import numpy as np

import matplotlib
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline


def get_pipeline(env):
    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    observation_examples = np.array(
        [env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # Used to convert a state to a featurized representation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = sklearn.pipeline.FeatureUnion(
        [("rbf1", RBFSampler(gamma=5.0, n_components=100)),
         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
         ("rbf3", RBFSampler(gamma=1.0,
                             n_components=100)), ("rbf4",
                                                  RBFSampler(
                                                      gamma=0.5,
                                                      n_components=100))])
    pipeline = Pipeline([('scaler', scaler), ('feat', featurizer)])
    pipeline.fit(observation_examples)
    return pipeline


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    """
    """

    def policy_fn(state):
        pgreedy = epsilon / float(nA)
        if np.random.rand() < pgreedy:
            action = random.choice(range(nA))
        else:
            action = np.argmax(estimator.predict(state))
        return action

    return policy_fn


def q_learning(env,
               estimator,
               num_episodes,
               discount_factor=1.0,
               epsilon=0.1,
               epsilon_decay=1.0):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    Args:
        env: OpenAI environment.
        estimator: Action-Value function estimator
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        epsilon_decay: Each episode, epsilon is decayed by this factor
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):
        if i_episode % 100 == 0:
            estimator.lr /= 2.
        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        # Print out which episode we're on, useful for debugging.
        # Also print reward for last episode
        last_reward = stats.episode_rewards[i_episode - 1]
        print(
            "\rEpisode {}/{} ({})".format(i_episode + 1, num_episodes,
                                          last_reward),
            end="")
        sys.stdout.flush()

        state = env.reset()
        t = 0
        while True:
            # get action
            env.render()
            action = policy(state)
            # env magic
            nstate, reward, done, _ = env.step(action)

            # update stats
            stats.episode_lengths[i_episode] += 1
            stats.episode_rewards[i_episode] += reward

            # predict Q(s,a)
            Qsa = estimator.predict(s=state, a=action)  #scalar

            # Compute the td target r + max(Q(s(t+1),a) over a
            nQsa = estimator.predict(nstate)  # Na,1
            target = reward + discount_factor * max(nQsa)

            # update the estimator toward lowering the MSE tdtarget/Qvalue
            estimator.update(state, action, target)

            # update state to nstate
            print(
                "\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, last_reward),
                end="")
            if done:
                break
            state = nstate
            t += 1

    return stats


class QEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, lr=0.1):
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.lr = lr

    def init_(self, env):
        self.pipeline = get_pipeline(env)
        state = env.reset()[np.newaxis, :]
        feat = self.pipeline.transform(state)
        nfeat = feat.shape[-1]
        self.w = np.random.randn(env.action_space.n, nfeat)

    def get_feature(self, s):
        s = s[np.newaxis, :]
        x = self.pipeline.transform(s)
        return x

    def predict(self, s, a=None):
        """
        Makes value function predictions.
        """
        x = self.get_feature(s)
        preds = np.dot(x, self.w.T).ravel()
        if a is not None:
            preds = preds[a]
        return preds

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        x = self.get_feature(s)  # 1XNf
        forward = np.dot(x, self.w.T).ravel()  # (1XNf) dot (NfXNa) --> 1XNa
        dw = (y - forward[a]) * x.T
        self.w[a, :] += self.lr * dw.ravel()
        return None
