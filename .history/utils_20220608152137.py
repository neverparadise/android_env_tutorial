import torch
import gym
import random


from android_env.wrappers.discrete_action_wrapper import DiscreteActionWrapper
from android_env.wrappers.image_rescale_wrapper import ImageRescaleWrapper
from android_env.wrappers.float_pixels_wrapper import FloatPixelsWrapper
from android_env.wrappers.tap_action_wrapper import TapActionWrapper

def converter():
    pass

def make_action():
    


def save_model(episode, save_period, save_path, model, model_name):
    if episode % save_period == 0:
        save_path_name = save_path+model_name+str(episode)+'.pt'
        torch.save(model.state_dict(), save_path_name)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def make_discrete_env(env, touch_only, redundant_actions, grid_shape, zoom_factors, grayscale=True):
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    print()
    

    env = TapActionWrapper(env, touch_only=touch_only)
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    print()
    
    # action space가 여전히 큰가?
    # env = DiscreteActionWrapper(env, (6, 4), redundant_actions=False) # action touch grid: 54 blocks
    env = DiscreteActionWrapper(env, grid_shape, redundant_actions=redundant_actions) # action touch grid: 25 blocks

    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    print()
    
    # env = ImageRescaleWrapper(env, zoom_factors=(0.0625, 0.0745),  grayscale=True)
    #env = ImageRescaleWrapper(env, zoom_factors=(1/24, 1/18),  grayscale=True)
    env = ImageRescaleWrapper(env, zoom_factors=zoom_factors,  grayscale=grayscale)

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

def make_continuous_env(env, zoom_factors, grayscale=True, touch_only=True):
    print('-'*128)
    print(env.action_spec())
    print()
    print(env.observation_spec())  
    print()
    
    env = ImageRescaleWrapper(env, zoom_factors=zoom_factors,  grayscale=True)
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

