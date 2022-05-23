import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        timestep_lst = []
        action_lst = []
        next_timestep_lst = []

        for transition in mini_batch:
            timestep, action, next_timestep = transition
            timestep_lst.append(timestep) # numpy array
            action_lst.append([action])
            next_timestep_lst.append(next_timestep)


        return obs_lst, delta_lst, action_lst, reward_lst, next_obs_list

    def size(self):
        return len(self.buffer)