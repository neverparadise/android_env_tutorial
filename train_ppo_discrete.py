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
from models.PPO_CNN_Discrete import PPO, Buffer, train_ppo, MLP
from utils import save_model, load_model, obs_converter, make_discrete_env, make_discrete_action
from torch.utils.tensorboard import SummaryWriter

original_env = android_env.load(
      emulator_path='~/Android/Sdk/emulator/emulator',
      android_sdk_root='~/Android/Sdk',
      android_avd_home='~/.android/avd',
      avd_name='my_avd',
      adb_path='~/Android/Sdk/platform-tools/adb',
      #task_path=f'{os.curdir}/tasks/mdp/mdp_0000.textproto',
      task_path=f'{os.curdir}/tasks/mdp/mdp_0003.textproto',
      run_headless=True)


env = make_discrete_env(original_env, touch_only=True, redundant_actions=False,\
      grid_shape=(8, 6), zoom_factors=(1/48, 1/36))
# env, touch_only, redundant_actions, grid_shape, zoom_factors, grayscale=True
action_spec = env.action_spec() 
obs_spec = env.observation_spec()
print(env.action_spec())
print(env.action_spec()['action_id'].num_values)
print(env.observation_spec())
# print(env.observation_spec()['pixels'])
print(env.observation_spec()['pixels'].shape)
print(env.action_spec().items())

n_actions = env.action_spec()['action_id'].num_values
state_dim = env.observation_spec()['pixels'].shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FIRST = dm_env.StepType.FIRST
DONE = dm_env.StepType.LAST

# Hyperparameters
BATCH_SIZE = 32   
LEARNING_RATE = 3e-4
GAMMA = 0.99
LAMBDA = 0.9
ENROPY_COEF = 1e-1
EPS_CLIP = 0.2
K_EPOCH = 10
T_HORIZON = 5

# MODEL, ENV, INFO
NET_TYPE = 'CNN'
ALG_TYPE = 'ppo'
MODEL_NAME = f'PPO_{NET_TYPE}_GRAY4030_Discrete{n_actions}Action_'
SAVE_PATH = "/home/slowlab/android_env_tutorial/weights/{}/".format(ALG_TYPE)
ENV_NAME = 'mdp_0003'
SUMMARY_PATH = f"/home/slowlab/android_env_tutorial/experiments/{ALG_TYPE}/train/{ENV_NAME+MODEL_NAME}"
SAVE_PERIOD = 1000


H, W, C = state_dim[0], state_dim[1], state_dim[2]
flatten_dim = H * W * C
def main():
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    torch.manual_seed(3407)
    if not os.path.isdir(SUMMARY_PATH):
        os.mkdir(SUMMARY_PATH)

    writer = SummaryWriter(SUMMARY_PATH)
    
    total_episodes = 1000000
    print(H, W, C)
    #model = MLP(flatten_dim, hidden=256, num_actions=n_actions).to(device).float() 
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
        steps = 0
        while not timestep.last():
            for t in range(T_HORIZON):
                steps += 1
                # * 1. convert  obs dict to obs tensors
                obs = obs_converter(timestep.observation)
                # * 2. sample action from policy
                prob = model.pi(obs)
                prob = prob.squeeze(0)
                action, action_dict = make_discrete_action(prob)
                # * 3. take a action for state transition
                next_timestep = env.step(action_dict)
                next_obs = obs_converter(next_timestep.observation)
                reward = next_timestep.reward
                done = next_timestep.step_type

                # * 4. calcuate score and put the transition in buffer
                score += next_timestep.reward
                transition = (obs, action, reward, next_obs, done, prob[action])
                buffer.put_data(transition)

                timestep = next_timestep
                steps+=1

                # * 5. if done, terminate episode
                if next_timestep.step_type == DONE:
                    break

            # * 6. train modelnd save
            loss = train_ppo(model, buffer, optimizer, K_EPOCH, LAMBDA, GAMMA, EPS_CLIP, ENROPY_COEF)
            save_model(episode, SAVE_PERIOD, SAVE_PATH, model, MODEL_NAME)
        
            if next_timestep.step_type == DONE:
                print(f'total_rewards of episode {episode}: {score}, steps: {steps}')
                writer.add_scalar("total_rewards", score, episode)
                writer.add_scalar("steps", steps, episode)
                writer.add_scalar("loss", loss, episode)
                break
        

    writer.close()

main()
