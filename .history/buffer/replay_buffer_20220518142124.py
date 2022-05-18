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
        state_list = []
        action_list = []
        reward_list =[]
        next_state_list = []
        done_mask_list = []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            state_list.append(s)
            action_list.append([a])
            reward_list.append([r])
            next_state_list.append(s_prime)
            done_mask_list.append([done_mask])

        s_lst = torch.tensor(np.array(state_list))
        a_lst = torch.tensor(np.array(action_list), dtype=torch.int64)
        r_lst = torch.tensor(np.array(reward_list))
        ns_lst = torch.tensor(np.array(next_state_list))
        d_lst = torch.tensor(np.array(done_mask_list))

        return s_lst, a_lst, r_lst, ns_lst, d_lst

    def size(self):
        return len(self.buffer)