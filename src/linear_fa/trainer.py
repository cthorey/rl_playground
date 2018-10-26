from src.common.base_trainer import BasePersonalTrainer
from box import Box
import torch.nn.functional as F
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PersonalTrainer(BasePersonalTrainer):
    def train_one_episode(self, env):
        state = env.reset()
        state = self.agent.stransformer.transform(state)
        episode = Box(steps=0, reward=0)
        while True:
            epsilon = self.epsilon_decay(self.steps_done)
            if hasattr(self, 'writer'):
                self.writer.add_scalar('data/epsilon', epsilon,
                                       self.steps_done)
            action = self.agent.select_action(state.to(DEVICE), epsilon)
            # env magic
            nstate, reward, done, _ = env.step(action)

            # add to episode stats
            episode.steps += 1
            episode.reward += reward
            self.agent.steps_done += 1

            # predict Q(s,a)
            action_values = self.agent.policy(state)
            action_values = action_values[0, action]

            # Compute the td target r + max(Q(s(t+1),a) over a
            nstate = self.agent.stransformer.transform(nstate)
            naction_values = self.agent.policy(nstate).detach().max()
            td_target = reward + self.agent.gamma * naction_values * (
                1.0 - done)

            loss = F.smooth_l1_loss(action_values, td_target)
            if hasattr(self, 'writer'):
                self.writer.add_scalar('data/loss', loss, self.steps_done)

            # update the weight toward lowering the MSE tdtarget/Qvalue
            self.optimizer.zero_grad()
            loss.backward()
            # one step of gradient descent
            self.optimizer.step()
            if done:
                break
            state = nstate

        return episode
