import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

#Hyperparameters
entropy_coef = 1e-2
critic_coef = 1
learning_rate = 0.0003
gamma         = 0.9
lmbda         = 0.9
eps_clip      = 0.2
K_epoch       = 10
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []
        self.fc1   = nn.Linear(3,64)
        self.fc2   = nn.Linear(64,256)
        self.fc_v  = nn.Linear(256,1)
        self.fc_pi = nn.Linear(256,1)
        self.fc_sigma = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2 * F.tanh(self.fc_pi(x))
        sigma = F.softplus(self.fc_sigma(x)) +1e-3
        return mu,sigma

    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
