import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PPO(nn.Module):
    def __init__(self, num_actions):
        super(PPO, self).__init__()
        self.num_actions = num_actions
        
        self.conv_layers = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=8, stride=4),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Flatten()
        )

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_size = conv2d_size_out(64, 8, 4)
        conv_size = conv2d_size_out(conv_size, 4, 2)
        conv_size = conv2d_size_out(conv_size, 3, 1)
        linear_input_size = conv_size * conv_size * 64 # 4 x 4 x 64 = 1024
        self.fc = nn.Linear(linear_input_size+1, 512)
        self.fc_pi = nn.Linear(512, self.num_actions)
        self.fc_v = nn.Linear(512, 1)
    
    def forward(self, obs, softmax_dim=1):
        # make_batch 코드에서 obs를 잘 만들어야 한다. 
        # pixels (Batch, C, H, W)
        # angles (Batch, angle)
        pixels, compassAngle = obs
        conv_feature = self.conv_layers(pixels) # (Batch, Linear_size)
        concat_feature = torch.cat((conv_feature, compassAngle), dim=1)
        feature = F.relu(self.fc(concat_feature))
        prob = self.fc_pi(feature)
        log_prob = F.softmax(prob, dim=softmax_dim)
        value = self.fc_v(feature)
        return log_prob, value

