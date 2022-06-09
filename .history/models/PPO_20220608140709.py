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

