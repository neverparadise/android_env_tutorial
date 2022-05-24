# coding=utf-8
# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example script demonstrating usage of AndroidEnv."""

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
from models.DQN import ClassicCNN, train_dqn
from buffer.replay_buffer import ReplayBuffer
from utils import *
from torch.utils.tensorboard import SummaryWriter
import os



FLAGS = flags.FLAGS

# Simulator args.
flags.DEFINE_string('avd_name', 'my_avd', 'Name of AVD to use.')
flags.DEFINE_string('android_avd_home', '~/.android/avd', 'Path to AVD.')
flags.DEFINE_string('android_sdk_root', '~/Android/Sdk', 'Path to SDK.')
flags.DEFINE_string('emulator_path',
                    '~/Android/Sdk/emulator/emulator', 'Path to emulator.')
flags.DEFINE_string('adb_path',
                    '~/Android/Sdk/platform-tools/adb', 'Path to ADB.')
flags.DEFINE_bool('run_headless', False,
                  'Whether to display the emulator window.')

# Environment args.
#flags.DEFINE_string('task_path', f'{os.curdir}/tasks/catch/catch_the_ball_default.textproto', 'Path to task textproto file.')
flags.DEFINE_string('task_path', f'{os.curdir}/tasks/2048/classic_2048.textproto', 'Path to task textproto file.')

# Experiment args.
flags.DEFINE_integer('num_steps', 400000, 'Number of steps to take.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GAMMA=0.9
MEMORY_SIZE = 100000
BATCH_SIZE = 128   # 32
LEARNING_RATE = 0.000625   # 0.01
TARGET_UPDATE = 2000  # 5
SAVE_PATH = "/home/slowlab/android_env_tutorial/weights/dqn/"
MODEL_NAME = 'DQN'
SAVE_PERIOD = 10000
START_SIZE = 1000

startEpsilon = 0.95
endEpsilon = 0.05
total_steps = 300000
epsilon = startEpsilon
stepDrop = (startEpsilon - endEpsilon)  * 300 / total_steps
n_actions = env.action_spec()['action_id'].num_values
state_dim = env.observation_spec()['pixels'].shape
H, W, C = state_dim[0], state_dim[1], state_dim[2]
behavior_policy = ClassicCNN(C, H, W, 3, 2, n_actions).to(device).float()    # C, H, W, K, S, num_actions
target_policy = ClassicCNN(C, H, W, 3, 2, n_actions).to(device).float()
target_policy.load_state_dict(behavior_policy.state_dict())
optimizer = optim.Adam(behavior_policy.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEMORY_SIZE)

writer = SummaryWriter("/home/slowlab/android_env_tutorial/experiments/dqn/train")

def converter(obs):
    if len(obs['pixels'].shape) < 4:
        obs_pixels = obs['pixels']
        obs_pixels = obs_pixels.transpose(2, 0, 1)
        obs_tensor = torch.tensor(obs_pixels).to(device).float()
        obs_tensor = obs_tensor.unsqueeze(0)
    else:
        obs_pixels = obs_pixels.transpose(2, 0, 1)
        obs_tensor = torch.tensor(obs_pixels).to(device).float()
        obs_tensor = obs_tensor.unsqueeze(0)
    return obs_tensor
    
from android_env.wrappers.discrete_action_wrapper import DiscreteActionWrapper
from android_env.wrappers.image_rescale_wrapper import ImageRescaleWrapper
from android_env.wrappers.float_pixels_wrapper import FloatPixelsWrapper
from android_env.wrappers.tap_action_wrapper import TapActionWrapper

def make_env(env):
    env = ImageRescaleWrapper(env, zoom_factors=(0.0625, 0.0745),  grayscale=True)
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    
    # env = TapActionWrapper(env, touch_only=True)
    # print('-'*128)
    # print(env.action_spec())
    # print()
    # print(env.observation_spec())  
    env = DiscreteActionWrapper(env, (6, 9), redundant_actions=False) # action touch grid: 54 blocks
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    

    return env

def main(_):

  with android_env.load(
      emulator_path=FLAGS.emulator_path,
      android_sdk_root=FLAGS.android_sdk_root,
      android_avd_home=FLAGS.android_avd_home,
      avd_name=FLAGS.avd_name,
      adb_path=FLAGS.adb_path,
      task_path=FLAGS.task_path,
      run_headless=FLAGS.run_headless) as env:

    total_rewards = 0
    step_type, reward, discount, obs = env.reset() # return Timestep object (step_type, reward, time_delta, obs)

    for step in range(10):
        action = behavior_policy.sample_action(converter(obs), epsilon)
        print(action)
        step_type, reward, discount, next_obs = env.step(action=action)
        writer.add_scalar("reward", reward, step)
        transition = (obs, action, reward, next_obs)
        memory.put(transition)
        obs = next_obs
        
        if memory.size() > START_SIZE:
            loss = train_dqn(behavior_policy, target_policy, memory, optimizer,GAMMA, BATCH_SIZE)

    if step % TARGET_UPDATE == 0:
        target_policy.load_state_dict(behavior_policy.state_dict())
        
    save_model(step, SAVE_PERIOD, SAVE_PATH,target_policy, MODEL_NAME)


if __name__ == '__main__':
  logging.set_verbosity('info')
  logging.set_stderrthreshold('info')
  flags.mark_flags_as_required(['avd_name', 'task_path'])
  app.run(main)
