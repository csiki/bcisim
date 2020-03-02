import gym
import random


class BCIEnv(gym.Env):
    def __init__(self, neur, dev):
        self.neur = neur
        self.dev = dev

    def step(self, action):
        pass

    def render(self, mode='human', close=False):
        pass

    def reset(self):
        pass

    def seed(self, seed):
        random.seed(seed)

    @classmethod
    def placement(cls, Gn, Gd, T, k):
        pass