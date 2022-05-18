import torch
import gym
import random
from wrapper.framestack import FrameBuffer
from wrapper.preprocess import PreprocessAtari

def save_model(episode, save_period, save_path, model, model_name):
    if episode % save_period == 0:
        save_path_name = save_path+model_name+str(episode)+'.pt'
        torch.save(model.state_dict(), save_path_name)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
