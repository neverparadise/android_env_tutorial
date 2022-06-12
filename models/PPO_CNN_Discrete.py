import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.distributions import Normal, Categorical
import dm_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DONE = dm_env.StepType.LAST

class PPO(nn.Module):
    def __init__(self, C, H, W, num_actions):
        super(PPO, self).__init__()
        self.num_actions = num_actions # for action_type
        
        self.conv_layers = nn.Sequential(
        nn.Conv2d(C, 32, kernel_size=8, stride=3),
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

        convh = conv2d_size_out(H, 8, 3)
        convh = conv2d_size_out(convh, 4, 2)
        convh = conv2d_size_out(convh, 3, 1)

        convw = conv2d_size_out(W, 8, 3)
        convw = conv2d_size_out(convw, 4, 2)
        convw = conv2d_size_out(convw, 3, 1)


        linear_input_size = convh * convw * 64 # 4 x 4 x 64 = 1024
        print(linear_input_size)
        self.fc = nn.Linear(linear_input_size+1, 512)
        self.fc_pi = nn.Linear(512, num_actions)

        self.fc_v = nn.Linear(512, 1)
    
    def forward(self, obs):
        pixels, timedeltas = obs
        conv_feature = self.conv_layers(pixels) # (Batch, Linear_size)
        concat_feature = torch.cat((conv_feature, timedeltas), dim=1)
        feature = F.relu(self.fc(concat_feature))
        return feature

    def pi(self, obs, softmax_dim=1):
        feature = self.forward(obs)
        prob = self.fc_pi(feature)
        prob = F.softmax(prob, dim=softmax_dim)
        return prob

    def v(self, obs):
        feature = self.forward(obs)
        value = self.fc_v(feature)
        return value

class MLP(nn.Module):
    def __init__(self, flatten_dim, hidden=256, num_actions=3):
        super().__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(flatten_dim, hidden*2)
        self.fc2 = nn.Linear(hidden*2, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden+1, hidden)

        self.fc_pi = nn.Linear(hidden, num_actions)
        self.fc_v = nn.Linear(hidden, 1)

    def forward(self, obs):
        pixels, timedeltas = obs
        x = pixels.contiguous().view(pixels.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        concat_x = torch.cat((x, timedeltas), dim=1)
        feature = F.relu(self.fc4(concat_x))
        return feature

    def pi(self, obs, softmax_dim=1):
        feature = self.forward(obs)
        prob = self.fc_pi(feature)
        prob = F.softmax(prob, dim=softmax_dim)
        return prob

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
        pixels_lst, timedelta_lst = [], []
        a_lst, r_lst, prob_a_lst, done_lst = [], [], [], []
        n_pixels_lst, n_timedelta_lst = [], []

        # 코드 짤 때, 단일 스칼라 값들, prob, action, done, reward shape를 위해 []로 감싸줘야함.
        for transition in self.data:
            obs, a, r, next_obs, done, prob_a = transition
            pixels_lst.append(obs[0])
            timedelta_lst.append([obs[1]])
            a_lst.append([a])
            r_lst.append([r])
            n_pixels_lst.append(next_obs[0])
            n_timedelta_lst.append([next_obs[1]])
            prob_a_lst.append([prob_a])
            done_mask = 0 if DONE else 1
            done_lst.append([done_mask])
        
        # 텐서로 변환할 것이 텐서들이 존재하는 리스트 cat 또는 stack
        # 텐서로 변환할 것이 그냥 값이 존재하는 리스트 -> torch.tensor

        pixels = torch.cat(pixels_lst).to(device)
        timedeltas = torch.tensor(timedelta_lst).to(device)
        actions = torch.tensor(a_lst).to(device, dtype=torch.int64)
        r = torch.tensor(r_lst, dtype=torch.float).to(device)
        n_pixels = torch.cat(n_pixels_lst).to(device)
        n_timedeltas = torch.tensor(n_timedelta_lst).to(device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(device)
        prob_a = torch.tensor(prob_a_lst).to(device)
        self.data = deque(maxlen=self.T_horizon)
        obs = (pixels, timedeltas)
        next_obs = (n_pixels, n_timedeltas)
        return obs, actions, r, next_obs, done_mask, prob_a


def train_ppo(model, buffer, optimizer, K_epoch, lmbda, gamma, eps_clip, entropy_coef):
    critic_coef = 1
    obs, a, r, next_obs, done_mask, old_prob = buffer.make_batch()

    for i in range(K_epoch):
        next_value = model.v(next_obs)
        current_value = model.v(obs)
        td_target = r + gamma * next_value * done_mask
        delta = td_target.detach() - current_value
        delta = delta.detach().cpu().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float).cuda()

        pi = model.pi(obs) 
        pi_a = pi.gather(1, a)
        ratio = torch.exp(torch.log(pi_a) - torch.log(old_prob).detach())  # a/b == exp(log(a)-log(b))

        dist = Categorical(pi)
        entropy = dist.entropy() * entropy_coef
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
        actor_loss = (-torch.min(surr1, surr2) - entropy).mean()
        critic_loss = critic_coef * F.smooth_l1_loss(current_value, td_target.detach())
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.mean().item()

        