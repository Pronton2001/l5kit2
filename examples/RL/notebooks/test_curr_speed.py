# from typing import Dict
# from l5kit.data.map_api import MapAPI
# # from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym
# from l5kit.environment.envs.l5_env2 import GymStepOutput, SimulationConfigGym, L5Env2
# from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym, L5Env
# # Dataset is assumed to be on the folder specified
# # in the L5KIT_DATA_FOLDER environment variable
# import gym
# import ray
# from ray.rllib.agents.ppo import PPOTrainer
# from ray.tune.logger import pretty_print
# import ray.rllib.algorithms.ppo as ppo
# from ray import tune
# from ray.rllib.models import ModelCatalog
# # from wrapper import L5EnvWrapper
# from testWrapper import L5EnvWrapper
# from customModel import TFGNNCNN, TorchGNCNN, TorchAttentionModel
# import os

# from matplotlib import pyplot as plt
# import numpy as np
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from tempfile import gettempdir

# from l5kit.configs import load_config_data
# from l5kit.data import ChunkedDataset, LocalDataManager
# from l5kit.dataset import EgoDatasetVectorized
# from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
# from l5kit.planning.vectorized.open_loop_model import VectorizedModel
# from l5kit.vectorization.vectorizer_builder import build_vectorizer

# # get environment config
# env_config_path = '/home/pronton/rl/l5kit/examples/RL/gym_vectorizer_config.yaml'
# dmg = LocalDataManager(None)
# cfg = load_config_data(env_config_path)

# train_eps_length = 32
# train_sim_cfg = SimulationConfigGym()
# train_sim_cfg.num_simulation_steps = train_eps_length + 1

# # env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
# env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
# env = L5Env(**env_kwargs)
# # env1_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
# # env1 = L5Env(**env_kwargs)
# env.reset()
# done = False
# while True:
#     # action = np.array(env.action_space.sample())
#     action = np.array(env.action_space.sample())
#     obs, r, done, info = env.step(action)
#     print(obs, r, action)
#     break
#     if done:
#         break
from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym, L5Env
from l5kit.environment.envs.l5_env2 import GymStepOutput, SimulationConfigGym, L5Env2
# env_config_path = '/home/pronton/rl/l5kit/examples/RL/gym_config.yaml'
env_config_path = '/home/pronton/rl/l5kit/examples/RL/gym_vectorizer_config.yaml'

import numpy as np

train_eps_length = 32
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}

# env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
env = L5Env2(**env_kwargs)
env.reset()
done = False
while True:
    # action = np.array(env.action_space.sample())
    action = np.array(env.action_space.sample())
    obs, r, done, info = env.step(action)
    # print(obs)

    break
    if done:
        break