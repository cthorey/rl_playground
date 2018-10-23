from box import Box
import torch, torchvision
from src.dqn import agent, trainer, utils
import numpy as np
from src.experiment import RLExperiment
import fire
import time
import sys


def continue_experiment(exp_prefix, expname, nepisodes=500000):
    bob = agent.Agent(expname=expname)
    coach = trainer.PersonalTrainer(bob)
    experiment = RLExperiment(coach, exp_prefix=exp_prefix)
    experiment.continue_experiment(expname, nepisodes)


def run_experiment(exp_prefix, double_dqn=False, nepisodes=500000):
    bob = agent.Agent()
    coach = trainer.PersonalTrainer(bob)
    experiment = RLExperiment(coach, exp_prefix=exp_prefix)
    exp = Box(agent_config=Box(double_dqn=double_dqn))
    experiment.fit_experiment(exp, nepisodes=nepisodes)


if __name__ == '__main__':
    fire.Fire()
