import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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