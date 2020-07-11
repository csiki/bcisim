import gym
import random
import torch
from neuronal_model import NeuronalModel
from device import Device
from typing import Union, Tuple, List


class BCIEnv(gym.Env):
    def __init__(self, neur: NeuronalModel, dev: Device, inputs: torch.Tensor, nsim_steps: int, ep_len: int,
                 goal_output_state: Union[List[torch.Tensor], None] = None):
        self.neur = neur
        self.dev = dev
        self.inputs = inputs
        self.nsim_steps = nsim_steps
        self.goal_output_state = goal_output_state

        self.ep_len = ep_len
        self.state_i = 0

        self.action_space = self.dev.action_space
        self.observation_space = self.dev.observation_space

    def step(self, action):
        Ar, As = action[:self.dev.nrec], action[self.dev.nrec:]
        self.dev.record(Ar)
        self.dev.stim(As)

        activity = self.neur.sim(self.inputs[self.state_i % self.inputs.shape[0]], self.nsim_steps)
        rec_activity = self.dev.read(activity)  # (recordings, output) tuple of the simulation

        reward = self._get_reward(rec_activity)

        self.state_i += 1
        return rec_activity, reward, self.state_i >= self.ep_len, {}

    def render(self, mode='human', close=False):
        pass  # TODO

    def reset(self):
        self.neur.reset()
        self.dev.reset()
        self.state_i = 0

    def seed(self, seed):
        random.seed(seed)

    def _get_reward(self, rec_activity: Tuple[torch.Tensor, torch.Tensor]):
        # TODO add option for state prediction
        # simple inverse of the distance between the desired and actual output state of the neural model
        if self.goal_output_state is None:
            raise ValueError('For the base BCIEnv implementation, a goal state is required!')
        output = rec_activity[1]
        goal_output = self.goal_output_state[self.state_i % len(self.goal_output_state)]
        return -torch.norm(output - goal_output)  # -L2 distance

    @classmethod
    def rnd_placement(cls, Gn, Gd):
        pass  # TODO

    @classmethod
    def placement(cls, Gn, Gd, T, k):  # in 3D
        pass  # TODO
