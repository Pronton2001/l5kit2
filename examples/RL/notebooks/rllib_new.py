from typing import Dict
from l5kit.data.map_api import MapAPI
# from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym
from l5kit.environment.envs.l5_env2 import GymStepOutput, SimulationConfigGym, L5Env2
from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym, L5Env
# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
import ray.rllib.algorithms.ppo as ppo
from ray import tune
from ray.rllib.models import ModelCatalog
# from wrapper import L5EnvWrapper
from testWrapper import L5EnvWrapper
from customModel import TFGNNCNN, TorchGNCNN
import os

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tempfile import gettempdir

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDatasetVectorized
from l5kit.planning.vectorized.closed_loop_model import VectorizedUnrollModel
from l5kit.planning.vectorized.open_loop_model import VectorizedModel
from l5kit.vectorization.vectorizer_builder import build_vectorizer

# get environment config
env_config_path = '/home/pronton/rl/l5kit/examples/RL/gym_vectorizer_config.yaml'
dmg = LocalDataManager(None)
cfg = load_config_data(env_config_path)
###############
# train_zarr = ChunkedDataset(dmg.require(cfg["train_data_loader"]["key"])).open()
# vectorizer = build_vectorizer(cfg, dmg)
# history_num_frames_ego = 1
# history_num_frames_agents = 3
# cfg["model_params"]["history_num_frames_ego"] = history_num_frames_ego
# cfg["model_params"]["history_num_frames_agents"] = history_num_frames_agents
# max_history_num_frames = max(history_num_frames_ego, history_num_frames_agents)
# num_agents = cfg["data_generation_params"]["other_agents_num"]

# cfg["model_params"]["history_num_frames_ego"] = history_num_frames_ego
# cfg["model_params"]["history_num_frames_agents"] = history_num_frames_agents

# zarr_dataset = train_zarr
# vect = build_vectorizer(cfg, dmg)
# dataset = EgoDatasetVectorized(cfg, zarr_dataset, vect)
# indexes = [0, 1, 10, -1]
# data = dataset[1]

# train_cfg = cfg["train_data_loader"]
# train_dataloader = DataLoader(dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
#                               num_workers=train_cfg["num_workers"])
#######################

train_sim_cfg = SimulationConfigGym()

env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': train_sim_cfg}
# tune.register_env("L5-CLE-V0", lambda config: L5Env(**env_kwargs))
env = L5Env2(**env_kwargs)
# print(env.observation_space)
# print(env._get_obs(0,True))
rollout_sim_cfg = SimulationConfigGym()
rollout_sim_cfg.num_simulation_steps = None
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': False, 'sim_cfg': rollout_sim_cfg,  'train': False, 'return_info': True}
rollout_env = L5Env2(**env_kwargs)
rollout_env = L5Env(**env_kwargs)

def rollout_episode(env, idx = 0):
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
        # action = np.array(env.action_space.sample())
        obs, _, done, info = env.step(np.array(env.action_space.sample()))
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
    sim_outs.append(rollout_episode(rollout_env, i))

#######################
from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle, simulation_out_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.io import output_notebook, show
# might change with different rasterizer
# map_API = rollout_env.dataset.rasterizer.sem_rast.mapAPI

# def visualize_outputs(sim_outs, map_API):
#     for sim_out in sim_outs: # for each scene
#         vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, map_API)

mapAPI = MapAPI.from_cfg(dmg, cfg)
for sim_out in sim_outs: # for each scene
    vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
    vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, mapAPI)
    show(visualize(sim_out.scene_id, vis_in))
# output_notebook()
# visualize_outputs(sim_outs, mapAPI)
exit()
tune.register_env("L5-CLE-V2", lambda config: L5Env2(**env_kwargs))
ray.init(num_cpus=1, ignore_reinit_error=True, log_to_driver=False)
ModelCatalog.register_custom_model(
        "VectorizedModel", VectorizedModel
    )
# Create the Trainer.
algo = ppo.PPO(
        env="L5-CLE-V2",
        config={
            'num_worker': 1,
            'disable_env_checking':True,
            "framework": "torch",
            "log_level":"INFO",
            # "model": {
            #     "custom_model": "VectorizedModel",
            #     # Extra kwargs to be passed to your model's c'tor.
            #     # "custom_model_config": ,
            # },
            '_disable_preprocessor_api': True,
        },
    )

for i in range(2):
    result = algo.train()
    print(pretty_print(result))