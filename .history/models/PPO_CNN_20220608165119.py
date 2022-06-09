import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class PPO(nn.Module):
    def __init__(self, C, H, W, num_actions):
        super(PPO, self).__init__()
        self.num_actions = num_actions # for action_type
        
        self.conv_layers = nn.Sequential(
        nn.Conv2d(C, 32, kernel_size=8, stride=4),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Flatten())

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convh = conv2d_size_out(H, 8, 4)
        convh = conv2d_size_out(convh, 4, 2)
        convh = conv2d_size_out(convh, 3, 1)

        convw = conv2d_size_out(W, 8, 4)
        convw = conv2d_size_out(convw, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)


        linear_input_size = convh * convw * 64 # 4 x 4 x 64 = 1024
        print(linear_input_size)
        self.fc = nn.Linear(linear_input_size+1, 512)
        self.fc_mu = nn.Linear(512, 2)
        self.fc_sigma = nn.Linear(512, 2)

        self.fc_v = nn.Linear(512, 1)
    
    def forward(self, obs):
        pixels, timedeltas = obs
        conv_feature = self.conv_layers(pixels) # (Batch, Linear_size)
        concat_feature = torch.cat((conv_feature, timedeltas), dim=1)
        feature = F.relu(self.fc(concat_feature))
        return feature

    def pi(self, obs):
        feature = self.forward(obs)
        mu_x = torch.tanh(self.fc_mu(feature))
        sigma = F.softplus(self.fc_sigma(feature)) +1e-3
        return mu, sigma

    def v(self, obs):
        feature = self.forward(obs)
        value = self.fc_v(feature)
        return value

class Buffer:
    def __init__(self, T_horizon):
        self.T_horizon = T_horizon
        self.data = deque(maxlen=T_horizon)

    def put_data(self, transition):
        # obs : pixels, angle
        # trans (obs, a, r, next_obs, prob[a].item(), done)
        self.data.append(transition)
        
    def make_batch(self):
        pixels_lst, angles_lst = [], []
        a_lst, r_lst, prob_a_lst, done_lst = [], [], [], []
        n_pixels_lst, n_angles_lst = [], []

        for transition in self.data:
            obs, a, r, next_obs, prob_a, done = transition
            pixels_lst.append(obs[0])
            angles_lst.append([obs[1]])
            a_lst.append([a])
            r_lst.append([r])
            n_pixels_lst.append(next_obs[0])
            n_angles_lst.append([next_obs[1]])
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        pixels = torch.cat(pixels_lst).to(device)
        angles = torch.tensor(angles_lst).to(device)
        a = torch.tensor(a_lst, dtype=torch.int64).to(device)
        r = torch.tensor(r_lst, dtype=torch.float).to(device)
        n_pixels = torch.cat(n_pixels_lst).to(device)
        n_angles = torch.tensor(n_angles_lst).to(device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(device)
        prob_a = torch.tensor(prob_a_lst).to(device)
        self.data = deque(maxlen=self.T_horizon)
        obs = (pixels, angles)
        next_obs = (n_pixels, n_angles)
        return obs, a, r, next_obs, done_mask, prob_a

        
def train_ppo(model, buffer, optimizer, K_epoch, lmbda, gamma, eps_clip):
    obs, a, r, next_obs, done_mask, prob_a = buffer.make_batch()
    for i in range(K_epoch):
        next_log_prob, next_value = model(next_obs)
        td_target = r + gamma * next_value * done_mask
        delta = td_target - next_value
        delta = delta.detach().cpu().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

        pi, value = model(obs) # [batch, 4]
        pi_a = pi.gather(1,a) 
        ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

        m = Categorical(pi)
        entropy = m.entropy().mean()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(value , td_target.detach()) - 0.5 * entropy

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    
    return loss.mean().item()

        