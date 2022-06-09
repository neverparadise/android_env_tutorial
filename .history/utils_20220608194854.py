import torch
import torch.nn.functional as F
from torch.distributions import Normal
import math
from android_env.wrappers.discrete_action_wrapper import DiscreteActionWrapper
from android_env.wrappers.image_rescale_wrapper import ImageRescaleWrapper
from android_env.wrappers.float_pixels_wrapper import FloatPixelsWrapper
from android_env.wrappers.tap_action_wrapper import TapActionWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def obs_converter(observation):
    pixel_tensor = pixel_converter(observation['pixels'])
    timedelta = time_scaler(observation['timedelta'])
    obs = (pixel_tensor, timedelta)
    return obs

def pixel_converter(pixels):
    if len(pixels.shape) < 4:
        pixel_tensor = pixels.transpose(2, 0, 1)
        pixel_tensor = torch.tensor(pixel_tensor).to(device).float()
        pixel_tensor = pixel_tensor.unsqueeze(0)
    else:
        pixel_tensor = pixels.transpose(0, 3, 1, 2)
        pixel_tensor = torch.tensor(pixel_tensor).to(device).float()
    return pixel_tensor

def time_scaler(timedelta):
    timedelta = timedelta * 1e-6
    timedelta = torch.tensor([timedelta]).to(device).float()
    timedelta = timedelta.unsqueeze(0)
    return timedelta

def make_continuous_action(mu, sigma):
    dist = Normal(mu,sigma)
    touch_position = torch.clamp(dist.sample(), min=0.0, max=1.0)
    log_prob = dist.log_prob(touch_position)
   #touch_position = touch_position.squeeze(0)
    #log_prob = log_prob.squeeze(0)
    action_dict = dict()
    action_dict['action_type'] = 0
    action_dict['touch_position'] = [touch_position[0][0].item(), touch_position[0][1].item()]
    return touch_position, action_dict, log_prob

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

