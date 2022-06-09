import torch
import gym
import random


from android_env.wrappers.discrete_action_wrapper import DiscreteActionWrapper
from android_env.wrappers.image_rescale_wrapper import ImageRescaleWrapper
from android_env.wrappers.float_pixels_wrapper import FloatPixelsWrapper
from android_env.wrappers.tap_action_wrapper import TapActionWrapper


def save_model(episode, save_period, save_path, model, model_name):
    if episode % save_period == 0:
        save_path_name = save_path+model_name+str(episode)+'.pt'
        torch.save(model.state_dict(), save_path_name)


def load_model(model, path):
    model.load_state_dict(torch.load(path))

def make_continuous_env(env):
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    print()
    
    env = ImageRescaleWrapper(env, zoom_factors=(1/24, 1/18),  grayscale=True)
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    print()
    
    env = TapActionWrapper(env, touch_only=True)
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    print()

    env = FloatPixelsWrapper(env)
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    print()
    
    return env

