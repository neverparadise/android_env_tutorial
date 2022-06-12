from typing import Dict
from absl import app
from absl import flags
from absl import logging

import android_env
from dm_env import specs
import numpy as np
import os


from android_env.wrappers.discrete_action_wrapper import DiscreteActionWrapper
from android_env.wrappers.image_rescale_wrapper import ImageRescaleWrapper
from android_env.wrappers.float_pixels_wrapper import FloatPixelsWrapper
from android_env.wrappers.gym_wrapper import GymInterfaceWrapper

def make_env2(env):
    env = FloatPixelsWrapper(env)
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())   
    env = GymInterfaceWrapper(env)
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())   
    env = ImageRescaleWrapper(env, zoom_factors=(0.042, 0.112),  grayscale=True)
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())   
    env = DiscreteActionWrapper(env, (6, 9))
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())   
    return env


def main():
    original_env = android_env.load(
      emulator_path='~/Android/Sdk/emulator/emulator',
      android_sdk_root='~/Android/Sdk',
      android_avd_home='~/.android/avd',
      avd_name='my_avd1',
      adb_path='~/Android/Sdk/platform-tools/adb',
      #task_path=f'{os.curdir}/tasks/mdp/mdp_0000.textproto',
      task_path=f'{os.curdir}/tasks/mdp/mdp_0003.textproto',
      run_headless=True)
    with android_env.load(
      emulator_path='~/Android/Sdk/emulator/emulator',
      android_sdk_root='~/Android/Sdk',
      android_avd_home='~/.android/avd',
      avd_name='my_avd1',
      adb_path='~/Android/Sdk/platform-tools/adb',
      #task_path=f'{os.curdir}/tasks/mdp/mdp_0000.textproto',
      task_path=f'{os.curdir}/tasks/mdp/mdp_0003.textproto',
      run_headless=True) as env:
        
        # task의 observation과 action에 대한 정보를 보겠습니다.
        env = make_env2(env)
        print(env.action_spec())
        print(env.observation_spec())
        action_spec = env.action_spec() 
        obs_spec = env.observation_spec()


main()