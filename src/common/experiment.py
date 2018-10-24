"""
Base class to setup deep learning experiments
"""

import json
import os
import re
import sys
import time
from os.path import join as ojoin
import sys

import numpy as np
import pandas as pd
from box import Box

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)


class RLExperiment(object):
    """
    Framework to experiment with a model and some data.
    """

    def __init__(self, trainer, exp_prefix):
        self.exp_prefix = exp_prefix
        self.trainer = trainer
        self.experiment = dict(
            model_folder=self.trainer.agent_folder,
            model_name=self.trainer.agent_name)
        self.__dict__.update(self.experiment)
        self.exp_prefix = exp_prefix

    def next_exp_name(self):
        """
        Find the name of already existing exp and
        return a new one
        """
        exp = set(
            [f.split('_')[0].split('t')[-1] for f in self.list_experiments()])
        if len(exp) == 0:
            return '{}t0'.format(self.exp_prefix)
        exp = sorted(exp, key=lambda x: int(x), reverse=True)
        if len(exp) == 0:
            n = 1
        else:
            n = int(exp[0]) + 1
        return '{}t{}'.format(self.exp_prefix, str(n))

    def list_experiments(self, verbose=True):
        """
        List all experiment that have been runned already
        """
        exps = os.listdir(self.trainer.agent_folder)
        return list(
            filter(lambda sid: re.match('{}.*_experiment.json'.format(self.exp_prefix), sid), exps))

    def initial_setup(self, exp, expname, overwrite):
        """
        Initial setup
        """
        self.pprint('Initial setup')
        expname = self.next_exp_name()
        self.trainer.agent.expname = expname
        self.current_expname = expname
        exp.update(dict(expname=expname))
        exp.update(self.experiment)
        self.pprint('Starting experiment {}'.format(expname))
        return exp

    def dump_experiment(self, exp):
        """
        Dump all the information about one experiment before training.
        if the model is built on a basemodel, dump only the head. If not, dump the
        model itself.
        """
        exp.update({
            key: val
            for key, val in self.trainer.agent.__dict__.items()
            if type(val) in [tuple, list, str, int, float]
        })
        json_string = exp
        model_path = os.path.join(self.trainer.agent_folder,
                                  '{}_experiment.json'.format(exp['expname']))
        self.pprint('Dumping the experiment infos to {}'.format(model_path))
        json.dump(json_string, open(model_path, 'w+'))
        return json_string

    def continue_experiment(self, expname, nepisodes):
        self.trainer.agent.load_experiment(expname, prefix='current')
        self.trainer.optimizer.load_state_dict(
            self.trainer.agent.checkpoint.optimizer)
        self.trainer.train(num_episodes=nepisodes)

    def fit_experiment(self,
                       nepisodes,
                       exp=None,
                       expname=None,
                       overwrite=False,
                       overfit=False):
        """
        Fit the given experiment.

        exp: Dictionary containing the detail of the experiment you want to run
        expname : Name of the experiment. If None, it is going to look in the
        current folder and increment by 1 exp{}.
        overwrite: Overwrite or not the current exp if it exists.
        """

        # initial setup
        if exp is None:
            exp = Box()
        exp = self.initial_setup(exp, expname, overwrite)

        # setup the model
        if 'agent_config' in exp:
            self.trainer.update_config(**exp['agent_config'])

        # dump info
        self.dump_experiment(exp=exp)

        # setup the model
        self.trainer.train(num_episodes=nepisodes)

    def pprint(self, m):
        print('*' * 100, file=sys.stderr)
        print(m)
