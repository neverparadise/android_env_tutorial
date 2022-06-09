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
from models.PPO_CNN import PPO, Buffer, train_ppo
from utils import save_model, load_model, obs_converter, make_continuous_env, make_continuous_action
from torch.utils.tensorboard import SummaryWriter

original_env = android_env.load(
      emulator_path='~/Android/Sdk/emulator/emulator',
      android_sdk_root='~/Android/Sdk',
      android_avd_home='~/.android/avd',
      avd_name='my_avd1',
      adb_path='~/Android/Sdk/platform-tools/adb',
      #task_path=f'{os.curdir}/tasks/mdp/mdp_0000.textproto',
      task_path=f'{os.curdir}/tasks/mdp/mdp_0003.textproto',
      run_headless=False)


env = make_continuous_env(original_env, zoom_factors=(1/24, 1/18))
action_spec = env.action_spec() 
obs_spec = env.observation_spec()
print(env.action_spec())
print(env.action_spec()['action_type'].num_values)
print(env.observation_spec())
print(env.observation_spec()['pixels'])
print(env.observation_spec()['pixels'].shape)
print(env.action_spec().items())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIRST = dm_env.StepType.FIRST
DONE = dm_env.StepType.LAST

# Hyperparameters
BATCH_SIZE = 64   # 32S
LEARNING_RATE = 3e-4
GAMMA = 0.9
LAMBDA = 0.9
ENROPY_COEF = 1e-2
EPS_CLIP = 0.2
K_EPOCH = 10
T_HORIZON = 20

# MODEL, ENV, INFO
MODEL_TYPE = 'ppo'
MODEL_NAME = 'PPO_CNN_GRAY8060_FloatAction_'
SAVE_PATH = "/home/slowlab/android_env_tutorial/weights/{}/".format(MODEL_TYPE)
ENV_NAME = 'mdp_0003'
SUMMARY_PATH = f"/home/slowlab/android_env_tutorial/experiments/{MODEL_TYPE}/train/{ENV_NAME+MODEL_NAME}"
SAVE_PERIOD = 1000


n_actions = env.action_spec()['action_type'].num_values
state_dim = env.observation_spec()['pixels'].shape
H, W, C = state_dim[0], state_dim[1], state_dim[2]

def main():
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    torch.manual_seed(31415)
    if not os.path.isdir(SUMMARY_PATH):
        os.mkdir(SUMMARY_PATH)

    writer = SummaryWriter(SUMMARY_PATH)
    
    total_episodes = 10000
    print(H, W, C)
    model = PPO(C, H, W, n_actions).to(device).float()    # C, H, W, K, S, num_actions
    buffer = Buffer(T_horizon=T_HORIZON)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.9)

    # return Timestep object (step_type, reward, time_deltWa, obs)
    for episode in range(total_episodes):
        score = 0
        timestep = env.reset()
        loss = 0.0        
        if episode == 0:
            print(timestep.observation['pixels'])
            print(timestep.observation['pixels'].shape)
            print(np.min(timestep.observation['pixels']))
            print(np.max(timestep.observation['pixels']))
            
        while not timestep.last():
            steps = 0
            # * 1. convert  obs dict to obs tensors
            obs = obs_converter(timestep.observation)

            # * 2. sample action from policy
            mu, sigma = model.pi(obs)
            action, log_prob = make_continuous_action(mu, sigma)
            
            if steps % 10 == 0:
                print(action['touch_position'])

            # * 3. take a action for state transition
            next_timestep = env.step(action)
            next_obs = obs_converter(next_timestep.observation)
            reward = next_timestep.reward
            done = next_timestep.step_type

            # * 4. calcuate score and put the transition in buffer
            score += next_timestep.reward
            transition = (obs, action, reward, next_obs, log_prob, done)
            buffer.put_data(transition)

            timestep = next_timestep
            
            # * 5. if done, terminate episode
            if next_timestep.step_type == DONE:
                print(f'total_rewards of episode {episode}: {score}')
                writer.add_scalar("total_rewards", score, episode)
                #writer.add_scalar("loss", loss, episode)
                break

            steps+=1
        
        save_model(episode, SAVE_PERIOD, SAVE_PATH, model, MODEL_NAME)
        
    writer.close()
main()

        
