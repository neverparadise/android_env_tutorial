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
            obs, action, reward, next_obs = transition
            obs_lst.append(obs) # numpy array
            action_lst.append([action])
            reward_lst.append([reward])
            next_obs_list.append(next_obs)


        return obs_lst, delta_lst, action_lst, reward_lst, next_obs_list

    def size(self):
        return len(self.buffer)