from typing import Dict
from absl import app
from absl import flags
from absl import logging

import android_env
from dm_env import specs
import numpy as np
import os

with android_env.load(
      emulator_path='~/Android/Sdk/emulator/emulator',
      android_sdk_root='~/Android/Sdk',
      android_avd_home='~/.android/avd',
      avd_name='my_avd',
      adb_path='~/Android/Sdk/platform-tools/adb',
      task_path=f'{os.curdir}/tasks/2048/classic_2048.textproto',
      run_headless=False) as env

