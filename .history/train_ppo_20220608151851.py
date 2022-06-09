from typing import Dict
import os
from absl import app
from absl import flags
from absl import logging
import android_env
import dm_env
from dm_env import specs
import numpy as np
import random
import torch
import torch.optim as optim
from models.PPO_CNN import PPO, Buffer, train_dqn
from utils import save_model, load_model, make_continuous_env
from torch.utils.tensorboard import SummaryWriter

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
LEARNING_RATE = 3e-4
TARGET_UPDATE = 100  # 5
MODEL_TYPE = 'ppo'
MODEL_NAME = 'PPO_CNN_GRAY8060_FloatAction_'
SAVE_PATH = "/home/slowlab/android_env_tutorial/weights/{}}/".format(MODEL_TYPE)
ENV_NAME = 'mdp_0003'
SUMMARY_PATH = f"/home/slowlab/android_env_tutorial/experiments/{MODEL_TYPE}}/train/{ENV_NAME+MODEL_NAME}"
SAVE_PERIOD = 1000


n_actions = env.action_spec()['action_id'].num_values
state_dim = env.observation_spec()['pixels'].shape
H, W, C = state_dim[0], state_dim[1], state_dim[2]

def main():
    if not os.path.isdir(SUMMARY_PATH):
        os.mkdir(SUMMARY_PATH)
    writer = SummaryWriter(SUMMARY_PATH)
    
    total_episodes = 10000
    print(H, W, C)
    behavior_policy = PPO(C, H, n_actions).to(device).float()    # C, H, W, K, S, num_actions
    optimizer = optim.Adam(behavior_policy.parameters(), lr=LEARNING_RATE, weight_decay=0.9)

    # return Timestep object (step_type, reward, time_delta, obs)
    for episode in range(total_episodes):
        total_rewards = 0
        timestep = env.reset()
        loss = 0.0        
        if episode == 0:
            print(timestep.observation['pixels'])
            print(timestep.observation['pixels'].shape)
            print(np.min(timestep.observation['pixels']))
            print(np.max(timestep.observation['pixels']))
            
        while not timestep.last():
            action_index = behavior_policy.sample_action(timestep.observation, epsilon)
            action = {}
            action['action_id'] = 0
            action['touch_position'] = action_index
        
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

        
