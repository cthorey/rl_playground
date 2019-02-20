import os

import numpy as np
import torch
from box import Box
from src.common.base_trainer import PersonalTrainer
from src.common.rollout import Rollout
from tensorboardX import SummaryWriter
from tqdm import tqdm

ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonalTrainer(PersonalTrainer):
    def __init__(self,
                 agent,
                 optimizer='RMSprop',
                 lr=1e-5,
                 nenv=1,
                 nsteps=10,
                 max_grad_norm=0.5,
                 value_loss_coeff=0.5,
                 entropy_loss_coeff=0.01):
        super(PersonalTrainer, self).__init__(
            agent=agent,
            optimizer=optimizer,
            lr=lr,
            nenv=nenv,
            nsteps=nsteps,
        )
        self.rollout = Rollout(
            agent=agent, nenv=nenv, nsteps=nsteps, seed=agent.seed)
        self.max_grad_norm = max_grad_norm
        self.batch_size = nenv * nsteps
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff

    @property
    def cumulative_reward(self):
        r = []
        for rewards in self.rollout.cumulative_reward.values():
            r.append(np.mean(rewards))
        return np.mean(r)

    def get_tensor(self, name):
        tensor = getattr(self.rollout, name).to(DEVICE)
        tensor = tensor.reshape(self.batch_size, -1)
        return tensor

    def update_agent(self):
        """
        Perform one step of gradient ascent.
        Use the Returns Gt directly. MC way.
        """
        states = self.get_tensor('states').float()
        actions = self.get_tensor('actions').long()
        returns = self.get_tensor('returns').float()

        values, action_log_probs, dist_entropy = self.agent.policy.evaluate_actions(
            states, actions)
        advantages = returns - values
        value_loss = advantages.pow(2).mean()
        policy_loss = -(advantages.detach() * action_log_probs).mean()
        loss = policy_loss
        loss += self.value_loss_coeff * value_loss
        loss += -self.entropy_loss_coeff * dist_entropy

        if self.steps_done % 1000 == 0:
            self.writer.add_scalar('policy_loss', policy_loss, self.steps_done)
            self.writer.add_scalar('value_loss', value_loss, self.steps_done)
            self.writer.add_scalar('entropy_loss', -dist_entropy,
                                   self.steps_done)
            self.writer.add_scalar('total_loss', loss, self.steps_done)

        # additional loss to encourage exploration
        self.optimizer.zero_grad()
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(),
                                       self.max_grad_norm)

        self.optimizer.step()

    def write_cumulative_summary(self):
        r1 = Box()
        r2 = Box()
        for key, rewards in self.rollout.cumulative_reward.items():
            r = self.rollout.cumulative_reward[key][-1]
            r1['reward_env{}'.format(key)] = r
            r = np.mean(self.rollout.cumulative_reward[key][-100:])
            r2['reward100_env{}'.format(key)] = r
        self.writer.add_scalars('rewards', r1.to_dict(), self.steps_done)
        self.writer.add_scalars('rewards100', r2.to_dict(), self.steps_done)

    def train(self, num_episodes=50):
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.agent_folder, self.expname))
        self.rollout.reset()
        for i in tqdm(range(num_episodes)):
            self.rollout.run()
            self.update_agent()
            if self.cumulative_reward >= self.best_reward:
                self.save('best')
                self.agent.best_reward = self.cumulative_reward
            if self.steps_done % 1000 == 0:
                self.write_cumulative_summary()
            history_path = os.path.join(self.agent.agent_folder,
                                        'history.json'.format(self.expname))
            self.writer.export_scalars_to_json(history_path)
            self.save('current')

        self.writer.close()
