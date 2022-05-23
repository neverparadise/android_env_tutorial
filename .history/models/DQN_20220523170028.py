import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import dm_env

DONE = dm_env.StepType.LAST
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ClassicCNN(nn.Module):
    def __init__(self, C, H, W, K, S, num_actions):
        self.num_actions = num_actions
        super().__init__()
        def conv2d_size_out(size, kernel_size=K, stride=S):
          return (size - (kernel_size - 1) - 1) // stride + 1
      
      # 원래 브레이크아웃 환경은 state shape가 (3, 210, 160)이어서 가로 세로에 대해서 다르게 사이즈를 계산해야함
      # 전처리, 프레임스택을 통해 (4, 84, 84)로 바꿨으므로 너비, 높이가 동일해서 H 변수를 사용하지 않음
        convh = conv2d_size_out(H, K, S)
        convh = conv2d_size_out(convh, K, S)
        convh = conv2d_size_out(convh, K, S)
        
        convw = conv2d_size_out(W, K, S)
        convw = conv2d_size_out(convw, K, S)
        convw = conv2d_size_out(convw, K, S)
        
        self.channels = [C, 16, 32, 64]
        self.layers = nn.ModuleList()
        for i in range(3):
          conv_layer = nn.Sequential(
          nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=K, stride=S, bias=True),
          nn.BatchNorm2d(self.channels[i+1]),
          nn.LeakyReLU())
          self.layers.append(conv_layer)
      
        # Last layers
        linear_input_size = convh*convw*self.channels[-1] + 1
        print(linear_input_size) 
        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, num_actions),
        )
        self.conv_module = nn.Sequential(*self.layers)
        

    def forward(self, pixels, timedeltas):
      x = self.conv_module(pixels) 
      x = x.view(x.size(0), -1)
      concat_x = torch.cat((x, timedeltas), dim=1)
      out = self.fc(concat_x)

      return out
  
    def sample_action(self, obs_dict, eps):
      pixels = pixel_converter(obs_dict)        
      time_delta = torch.tensor([obs_dict['timedelta']]).to(device)
      time_delta = time_delta.unsqueeze(1)
      
      out = self.forward(pixels, time_delta)
      coin = random.random()
      if coin < eps:
          return random.randint(0, self.num_actions-1)
      else:
          return torch.argmax(out).item()
      
def pixel_converter(observation):
    if len(observation['pixels'].shape) < 4:
        pixel_tensor = observation['pixels']
        pixel_tensor = pixel_tensor.transpose(2, 0, 1)
        pixel_tensor = torch.tensor(pixel_tensor).to(device).float()
        pixel_tensor = pixel_tensor.unsqueeze(0)
    else:
        pixel_tensor = pixel_tensor.transpose(2, 0, 1)
        pixel_tensor = torch.tensor(pixel_tensor).to(device).float()
    return pixel_tensor

def make_batch(memory, batch_size):
    time_steps, actions_dicts, next_timesteps = memory.sample(batch_size)
    obs_dicts = {'pixels': [], 'timedelta': []}
    next_obs_dicts = {'pixels': [], 'timedelta': []}
    actions = []
    rewards = []
    dones = []
    
    # actions, 
    for action_dict in actions_dicts:
      actions.append(torch.tensor([action_dict['action_id']]).to(device))
    
    # observations, next_observations, rewards, dones
    for ts, next_ts in zip(time_steps, next_timesteps):
      print('before pixel shape : {}'.format(ts.observation['pixels'].shape)) # (120, 80, 1)
      obs_pixel = pixel_converter(ts.observation)
      print('after pixel shape : {}'.format(obs_pixel.shape)) # (1, 1, 120, 80)
      
      obs_timedelta = ts.observation['timedelta']
      next_obs_pixel = pixel_converter(next_ts.observation)
      next_obs_timedelta = next_ts.observation['timedelta']
      
      obs_dicts['pixels'].append(obs_pixel)
      next_obs_dicts['pixels'].append(next_obs_pixel)
      obs_dicts['timedelta'].append(torch.tensor([obs_timedelta]).to(device))
      next_obs_dicts['timedelta'].append(torch.tensor([next_obs_timedelta]).to(device))
      
      rewards.append(torch.tensor([ts.reward]).to(device))
      if ts.step_type != DONE:
        dones.append(torch.tensor([1]).to(device))
      else:
        dones.append(torch.tensor([0]).to(device))
        
    return obs_dicts, actions, next_obs_dicts, rewards, dones

def train_dqn(behavior_net, target_net, memory, optimizer, gamma, batch_size):
    obs_dicts, actions, next_obs_dicts, rewards, dones = make_batch(memory, batch_size)
    cur_pixels = torch.cat(obs_dicts['pixels'], dim=0).to(device).float()
    cur_timedeltas = torch.stack(obs_dicts['timedelta']).to(device)
    actions = torch.stack(actions).to(device).int()
    reward = torch.stack(rewards).to(device).float()
    next_pixels = torch.cat(next_obs_dicts['pixels'], dim=0).to(device).float()
    next_timedeltas = torch.stack(next_obs_dicts['timedelta']).to(device)
    dones = torch.stack(dones).to(device).float()
    
    q_out = behavior_net.forward(cur_pixels, cur_timedeltas)
    q_a = q_out.gather(1, actions)
    max_target_q = target_net(next_pixels, next_timedeltas).max(1)[0].unsqueeze(1).detach()
    target = reward + gamma * max_target_q * dones
    loss = F.smooth_l1_loss(q_a, target).to(device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

