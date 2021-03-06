import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import dm_env

DONE = dm_env.StepType.LAST
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, flatten_dim, hidden, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(flatten_dim, hidden)
        self.fc2 = nn.Linear(hidden+1, hidden)
        self.fc3 = nn.Linear(hidden, num_actions)

    def forward(self, pixels, timedeltas):
        x = pixels.contiguous().view(pixels.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        concat_x = torch.cat((x, timedeltas), dim=1)
        x = F.leaky_relu(self.fc2(concat_x))
        out = F.leaky_relu(self.fc3(x))
        return out

    def sample_action(self, obs_dict, eps):
        pixels = pixel_converter(obs_dict)
        time_delta = torch.tensor([obs_dict['timedelta']]).to(device, dtype=torch.float)
        time_delta = time_scaler(time_delta)
        time_delta = time_delta.unsqueeze(1)

        out = self.forward(pixels, time_delta)
        coin = random.random()
        if coin < eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return torch.argmax(out).item()


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
      coin = random.random()
      if coin < eps:
          return random.randint(0, self.num_actions-1)
      else:
          pixels = pixel_converter(obs_dict)        
          time_delta = torch.tensor([obs_dict['timedelta']]).to(device,  dtype=torch.float)
          time_delta = time_scaler(time_delta)
          time_delta = time_delta.unsqueeze(1)
          
          out = self.forward(pixels, time_delta)
          return torch.argmax(out).item()

def time_scaler(timedelta):
    return timedelta * 1e-6
      
def pixel_converter(observation):
    if len(observation['pixels'].shape) < 4:
        pixel_tensor = observation['pixels']
        pixel_tensor = pixel_tensor.transpose(2, 0, 1)
        pixel_tensor = torch.tensor(pixel_tensor).to(device).float()
        pixel_tensor = pixel_tensor.unsqueeze(0)
    else:
        pixel_tensor = observation['pixels']
        pixel_tensor = pixel_tensor.transpose(0, 3, 1, 2)
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
      obs_pixel = pixel_converter(ts.observation)
      obs_timedelta = time_scaler(ts.observation['timedelta'])
      next_obs_pixel = pixel_converter(next_ts.observation)
      next_obs_timedelta = time_scaler(next_ts.observation['timedelta'])
      
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
    cur_pixels = torch.cat(obs_dicts['pixels'], dim=0).to(device, dtype=torch.float)
    cur_timedeltas = torch.stack(obs_dicts['timedelta']).to(device, dtype=torch.float)
    actions = torch.stack(actions).to(device, dtype=torch.long)
    reward = torch.stack(rewards).to(device, dtype=torch.float)
    next_pixels = torch.cat(next_obs_dicts['pixels'], dim=0).to(device, dtype=torch.float)
    next_timedeltas = torch.stack(next_obs_dicts['timedelta']).to(device, dtype=torch.float)
    dones = torch.stack(dones).to(device, dtype=torch.float)


    q_out = behavior_net.forward(cur_pixels, cur_timedeltas)
    q_a = q_out.gather(1, actions)
    max_target_q = target_net(next_pixels, next_timedeltas).max(1)[0].unsqueeze(1).detach()
    target = reward + gamma * max_target_q * dones
    loss = F.smooth_l1_loss(q_a, target).to(device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    soft_update(behavior_net, target_net, 1e-3)
    
    # print(f'cur_pixels : {cur_pixels}')
    # print(f'cur_timedeltas : {cur_timedeltas}')
    # print(f'q_out : {q_out}')
    # print(f'max_target_q : {max_target_q}')
    # print(f'target : {target}')
    # print(f'q_a : {q_a}')
    return loss.item()

def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

