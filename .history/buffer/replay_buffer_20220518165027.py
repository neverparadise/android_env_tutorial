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
        obs_lst = []
        delta_lst = []
        action_lst = []
        reward_lst =[]
        next_obs_list = []

        for transition in mini_batch:
            obs, action, reward, next_obs = transition
            obs_lst.append(obs['pixels']) # numpy array
            delta_lst.append(obs['timedelta'])
            action_lst.append([action])
            reward_lst.append([reward])
            next_obs_list.append(next_obs['pixels'])

        s_lst = torch.tensor(obs_lst)
        delta_lst = torch.tensor(delta_lst)
        a_lst = torch.tensor(action_lst, dtype=torch.int64)
        r_lst = torch.tensor(reward_lst)
        ns_lst = torch.tensor(next_obs_list)

        return s_lst, delta_lst, a_lst, r_lst, ns_lst

    def size(self):
        return len(self.buffer)