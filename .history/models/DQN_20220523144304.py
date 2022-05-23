from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
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
        

    def forward(self, obs_dict):
      pixels = obs_dict['pixels']
      time_deltas = obs_dict['time_delta']

      x = self.conv_module(pixels) 
      x = x.view(x.size(0), -1)
      concat_x = torch.cat((x, time_deltas), dim=1)
      out = self.fc(concat_x)

      return out
  
    def sample_action(self, obs_dict, eps):
        out = self.forward(obs_dict)
        coin = random.random()
        if coin < eps:
            return random.randint(0, self.num_actions-1)
        else:
            return torch.argmax(out).item()

def converter(observation):
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
    time_steps, actions, next_timesteps = memory.sample(batch_size)
    

def train_dqn(behavior_net, target_net, memory, optimizer, gamma, batch_size):
    obs_tensor, delta, action, reward, next_obs_tensor = memory.sample(batch_size)    
    obs_tensor = obs_tensor.to(device).float()
    delta = delta.to(device).float()
    action = action.to(device)
    reward = reward.to(device).float()
    next_obs_tensor = next_obs_tensor.to(device).float()
    
            s_lst = torch.tensor(np.array(obs_lst))
        delta_lst = torch.tensor(np.array(delta_lst))
        a_lst = torch.tensor(action_lst, dtype=torch.int64)
        r_lst = torch.tensor(np.array(reward_lst))
        ns_lst = torch.tensor(np.array(next_obs_list))
        
      


    q_out = behavior_net(obs_tensor)
    q_a = q_out.gather(1, action)
    # action 축 기준으로 max 취한 후, 0번째 인덱스를 가져오면 values를 가져온다.
    # 그 후 shape를 맞추기 위해 unsqueeze()를 한다.
    max_target_q = target_net(next_obs_tensor).max(1)[0].unsqueeze(1).detach()
    target = reward + gamma * max_target_q 
    loss = F.smooth_l1_loss(q_a, target).to(device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

class DeepResidualCNN(nn.Module):
  def __init__(self, in_channels, out_channels, output_dim, layer_num):
    super().__init__()
    # TODO
    kernel_size = 3
    stride = 1
    self.leakyrelu = nn.LeakyReLU()
    # First layer
    self.first_block = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                            padding=1, bias=True),
          nn.BatchNorm2d(out_channels),
          nn.LeakyReLU())
    
    # Middle layers
    self.conv_layers = nn.ModuleList()
    self.channels = [out_channels for i in range(layer_num)]
    for i in range(layer_num):  
      conv_block = nn.Sequential(
                                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=1, bias=True),
                                nn.LeakyReLU())
                                
      self.conv_layers.append(conv_block)
    
    linear_input_size = 4*3*out_channels
    
    # Last layer
    self.fc = nn.Sequential(
          nn.Flatten(),
          nn.Linear(linear_input_size, output_dim),
          nn.Softmax()
      )

  def forward(self, x):
    if len(x.shape) < 4:
      x = x.unsqueeze(0)
    x = self.first_block(x)
    shortcut = x
    for conv_block in self.conv_layers:
      x = conv_block(x)
      x += shortcut
      shortcut = x
    x = self.fc(x)
    return x
