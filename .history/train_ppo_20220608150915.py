from typing import Dict
from absl import app
from absl import flags
from absl import logging

import android_env
from dm_env import specs
import numpy as np
import os

import numpy as np
import random
import torch
import torch.optim as optim
from models.DQN import ClassicCNN, train_dqn, MLP
from buffer.replay_buffer import ReplayBuffer
from utils import save_model, load_model, make_continuous_env
from torch.utils.tensorboard import SummaryWriter
import os
import dm_env

original_env = android_env.load(
      emulator_path='~/Android/Sdk/emulator/emulator',
      android_sdk_root='~/Android/Sdk',
      android_avd_home='~/.android/avd',
      avd_name='my_avd',
      adb_path='~/Android/Sdk/platform-tools/adb',
      #task_path=f'{os.curdir}/tasks/mdp/mdp_0000.textproto',
      task_path=f'{os.curdir}/tasks/mdp/mdp_0003.textproto',
      run_headless=False)


env = make_continuous_env(original_env, zoom_factors=(1/24, 1/18))
action_spec = env.action_spec() 
obs_spec = env.observation_spec()
print(env.action_spec())
print(env.action_spec()['action_id'].num_values)
print(env.observation_spec())
print(env.observation_spec()['pixels'])
print(env.observation_spec()['pixels'].shape)
print(env.action_spec().items())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIRST = dm_env.StepType.FIRST
DONE = dm_env.StepType.LAST
BATCH_SIZE = 64   # 32S
# LEARNING_RATE = 0.000625   # 0.01
LEARNING_RATE = 3e-4
TARGET_UPDATE = 100  # 5
SAVE_PATH = "/home/slowlab/android_env_tutorial/weights/ppo/"
MODEL_NAME = 'PPO_CNN_GRAY8060_FLOAT_mdp'
ENV_NAME = 'mdp_0003'
SAVE_PERIOD = 2500
START_SIZE = 200


def main():
    summary_path = "/home/slowlab/android_env_tutorial/experiments/dqn/train/{}".format(ENV_NAME+MODEL_NAME)
    if not os.path.isdir(summary_path):
        os.mkdir(summary_path)
    writer = SummaryWriter(summary_path)
    
    startEpsilon = 0.95
    endEpsilon = 0.05
    total_episodes = 10000
    #total_steps = 100000
    epsilon = startEpsilon
    stepDrop = (startEpsilon - endEpsilon)  * 3 / total_episodes
    n_actions = env.action_spec()['action_id'].num_values
    state_dim = env.observation_spec()['pixels'].shape
    H, W, C = state_dim[0], state_dim[1], state_dim[2]
    print(H, W, C)
    flatten_dim = C*H*W
    behavior_policy = ClassicCNN(C, H, W, 3, 2, n_actions).to(device).float()    # C, H, W, K, S, num_actions
    target_policy = ClassicCNN(C, H, W, 3, 2, n_actions).to(device).float()
    # behavior_policy = MLP(flatten_dim, 256, n_actions).to(device).float()
    # target_policy = MLP(flatten_dim, 256, n_actions).to(device).float()
    target_policy.load_state_dict(behavior_policy.state_dict())
    optimizer = optim.Adam(behavior_policy.parameters(), lr=LEARNING_RATE, weight_decay=0.9)
    memory = ReplayBuffer(MEMORY_SIZE)

    # return Timestep object (step_type, reward, time_delta, obs)
    
    for episode in range(total_episodes):
        if(epsilon > endEpsilon):
            epsilon -= stepDrop
        total_rewards = 0
        timestep = env.reset()
        if episode == 0:
            print(timestep.observation['pixels'])
            print(timestep.observation['pixels'].shape)
            print(np.min(timestep.observation['pixels']))
            print(np.max(timestep.observation['pixels']))
            
        loss = 0.0
        #step_type, reward, discount, obs = timestep
        
        while not timestep.last():
            action_index = behavior_policy.sample_action(timestep.observation, epsilon)
            action = {}
            action['action_id'] = action_index
            #action['touch_position'] = action_index
        
            next_timestep = env.step(action=action)
            total_rewards += next_timestep.reward
            transition = (timestep, action, next_timestep)
            memory.put(transition) # break는 반드시 메모리에 넣고 끝낸다.
            
            if next_timestep.step_type == DONE:
                #print(f'total_rewards of episode {episode}: {total_rewards}')
                #print(f"# of transitions in memory: {memory.size()}")
                writer.add_scalar("total_rewards", total_rewards, episode)
                writer.add_scalar("memory size", memory.size(), episode)
                writer.add_scalar("epsilon", epsilon, episode)
                writer.add_scalar("loss", loss, episode)
                break
        
            timestep = next_timestep
            if memory.size() > START_SIZE:
                loss = train_dqn(behavior_policy, target_policy, memory, optimizer, GAMMA, BATCH_SIZE)
    
        if episode % TARGET_UPDATE == 0:
            target_policy.load_state_dict(behavior_policy.state_dict())
        save_model(episode, SAVE_PERIOD, SAVE_PATH,target_policy, MODEL_NAME)
        

    writer.close()
main()

        
