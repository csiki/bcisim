import gym
import random
import torch
from neuronal_model import NeuronalModel
from device import Device


class BCIEnv(gym.Env):
    def __init__(self, neur: NeuronalModel, dev: Device, inputs: torch.Tensor, ep_len: int):
        self.neur = neur
        self.dev = dev
        self.inputs = inputs

        self.ep_len = ep_len
        self.state_i = 0

    def step(self, action: dict):
        nsteps, Ar, As = action['nsteps'], action['Ar'], action['Ar']
        self.dev.record(Ar)
        self.dev.stim(As)

        activity = self.neur.sim(self.inputs[self.state_i % self.inputs.shape[0]], nsteps)
        rec_activity = self.dev.read(activity)

        # TODO continue: https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

        # TODO compute reward
        reward = self._get_reward()

        self.state_i += 1
        return rec_activity, reward, self.state_i >= self.ep_len, {}

    def render(self, mode='human', close=False):
        pass

    def reset(self):
        self.neur.reset()
        self.dev.reset()
        self.state_i = 0

    def seed(self, seed):
        random.seed(seed)

    def _get_reward(self):
        pass

    @classmethod
    def placement(cls, Gn, Gd, T, k):  # in 3D
        pass
