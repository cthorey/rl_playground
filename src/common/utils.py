import gym
import numpy as np
from baselines.run import get_env_type, make_vec_env


class GymEnvWrapperDebug(object):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)

    def reset(self):
        obs = self.env.reset()
        return np.expand_dims(obs, axis=0)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def render(self):
        r = self.env.render('rgb_array')
        return r.transpose(2, 0, 1)

    def step(self, action):
        state, reward, done, info = self.env.step(action[0])
        state = np.expand_dims(state, axis=0)
        reward = np.array([reward])
        done = np.array([done])
        info = [info]
        return state, reward, done, info


def make_env(env_name, nenv=1, seed=132, debug=True):
    if debug:
        return GymEnvWrapperDebug(env_name)
    env_type, env_id = get_env_type(env_name)
    env = make_vec_env(env_id, env_type, nenv, seed, reward_scale=1)
    return env
