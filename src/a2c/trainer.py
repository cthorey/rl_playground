import os
import numpy as np
import torch
from src.common.base_trainer import BasePersonalTrainer
from src.a2c.utils import Rollout
import torch.nn.functional as F
ROOT_DIR = os.environ['ROOT_DIR']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter


class PersonalTrainer(BasePersonalTrainer):
    def __init__(self, agent):
        super(PersonalTrainer, self).__init__(agent=agent)
        self.rollout = Rollout(agent=agent)
        self.batch_size = self.rollout.nenv * self.agent.nsteps

    @property
    def cumulative_reward(self):
        return self.rollout.cumulative_reward

    def get_tensor(self, name):
        tensor = torch.from_numpy(getattr(self.rollout, name)).to(DEVICE)
        tensor = tensor.reshape(self.batch_size, -1)
        return tensor

    def update_agent(self):
        """
        Perform one step of gradient ascent.
        Use the Returns Gt directly. MC way.
        """
        states = self.get_tensor('states').float()
        actions = self.get_tensor('actions').long().squeeze(-1)
        advantages = self.get_tensor('advantages').float()
        returns = self.get_tensor('returns').float()

        logits, values = self.agent.policy(states)
        cross_entropy = F.cross_entropy(logits, actions)
        wcross_entropy = cross_entropy * advantages
        # policy loss
        policy_loss = wcross_entropy.sum()
        # value loss
        value_loss = F.mse_loss(values, returns).sum()
        # loss
        loss = policy_loss + self.agent.wloss.value * value_loss
        if self.steps_done % 1000 == 0:
            self.writer.add_scalar('policy_loss', policy_loss, self.steps_done)
            self.writer.add_scalar('value_loss', value_loss, self.steps_done)
            self.writer.add_scalar('total_loss', loss, self.steps_done)

        # additional loss to encourage exploration
        self.optimizer.zero_grad()
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(),
                                       self.agent.max_grad_norm)

        self.optimizer.step()

    def train(self, num_episodes=50):
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.agent_folder, self.expname))
        self.rollout.reset()
        rewards = []
        for i in tqdm(range(num_episodes)):
            self.rollout.run()
            self.update_agent()
            if max(self.cumulative_reward) >= self.best_reward:
                self.save('best')
                self.agent.best_reward = max(self.cumulative_reward)
            if self.steps_done % 1000 == 0:
                self.writer.add_scalar('explaine_variance',
                                       self.rollout.explained_variance,
                                       self.steps_done)
                self.writer.add_scalar('reward', self.cumulative_reward[-1],
                                       self.steps_done)
                self.writer.add_scalar('100_reward',
                                       np.mean(self.cumulative_reward[-100:]),
                                       self.steps_done)
            history_path = os.path.join(self.agent.agent_folder,
                                        'history.json'.format(self.expname))
            self.writer.export_scalars_to_json(history_path)
            self.save('current')

        self.writer.close()
