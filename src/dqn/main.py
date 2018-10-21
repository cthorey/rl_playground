from box import Box
import torch, torchvision
from src.dqn import agent, trainer, utils
import numpy as np


def continue_training(exp_prefix, expname, nepisodes):
    bob = agent.Agent(expname)
    coach = trainer.PersonalTrainer(bob)
    experiment = RLExperiment(coach, exp_prefix=exp_prefix)
    experiment.continue_experiment(expname, nepisodes=nepisodes)
    exp = Box()
