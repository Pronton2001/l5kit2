import matplotlib.pyplot as plt
import numpy as np
import torch
from prettytable import PrettyTable

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset

from l5kit.dataset import EgoDatasetVectorized
from l5kit.vectorization.vectorizer_builder import build_vectorizer

from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator

from l5kit.visualization.visualizer.visualizer import visualize, visualize2, visualize3, visualize4
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI
from bokeh.models import Button


import os

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/home/pronton/rl/l5kit_dataset/"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("/home/pronton/rl/l5kit/examples/RL/gym_config.yaml")

####################################################
import gym

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv

from l5kit.configs import load_config_data
from l5kit.environment.feature_extractor import CustomFeatureExtractor
from l5kit.environment.callbacks import L5KitEvalCallback
from l5kit.environment.envs.l5_env import SimulationConfigGym

from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show

# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable

# get environment config
env_config_path = '/home/pronton/rl/l5kit/examples/RL/gg colabs/gym_config.yaml'
cfg = load_config_data(env_config_path)
# Train on episodes of length 32 time steps
train_eps_length = 32
train_envs = 4

# make train env
train_sim_cfg = SimulationConfigGym()
train_sim_cfg.num_simulation_steps = train_eps_length + 1
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
env = make_vec_env("L5-CLE-v0", env_kwargs=env_kwargs, n_envs=train_envs,
                   vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"})

# make train env
modelA = SAC.load('/home/pronton/rl/l5kit/examples/RL/gg colabs/logs/SAC_640000_steps.zip', env = env
        #          , custom_objects = {
        #     "learning_rate": 0.0,
        #     "lr_schedule": lambda _: 0.0,
        #     "clip_range": lambda _: 0.0,
        # }
        )

rollout_sim_cfg = SimulationConfigGym()
rollout_sim_cfg.num_simulation_steps = 3

rollout_env = gym.make("L5-CLE-v0", env_config_path=env_config_path, sim_cfg=rollout_sim_cfg, \
                       use_kinematic=True, train=False, return_info=True)

def rollout_episode(model, env, idx = 0):
    """Rollout a particular scene index and return the simulation output.

    :param model: the RL policy
    :param env: the gym environment
    :param idx: the scene index to be rolled out
    :return: the episode output of the rolled out scene
    """

    # Set the reset_scene_id to 'idx'
    env.set_reset_id(idx)
    
    # Rollout step-by-step
    obs = env.reset()
    done = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        if done:
            break

    # The episode outputs are present in the key "sim_outs"
    sim_out = info["sim_outs"][0]
    return sim_out

# Rollout one episode
# sim_out = rollout_episode(model, rollout_env)
# Rollout 5 episodes
sim_outs =[]
for i in range(1):
    sim_outs.append(rollout_episode(modelA, rollout_env))

print(sim_outs[0].ego_ins_outs)
