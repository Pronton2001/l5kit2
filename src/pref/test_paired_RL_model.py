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
modelB = PPO.load('/home/pronton/rl/l5kit/examples/RL/gg colabs/logs/PPO_5410000_steps.zip', env = env
        #          , custom_objects = {
        #     "learning_rate": 0.0,
        #     "lr_schedule": lambda _: 0.0,
        #     "clip_range": lambda _: 0.0,
        # }
        )
rollout_sim_cfg = SimulationConfigGym()
rollout_sim_cfg.num_simulation_steps = 50
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
    sim_outs.append(rollout_episode(modelA, rollout_env, 20))
    sim_outs.append(rollout_episode(modelB, rollout_env, 20))

####################################################

mapAPI = MapAPI.from_cfg(dm, cfg)
from bokeh.layouts import column, LayoutDOM, row, gridplot
from bokeh.io import curdoc

############################################ 2 scene
doc = curdoc()
# for sim_out in sim_outs[:1]: # for each scene
sim_out = sim_outs[0]
vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, mapAPI)
v1 = visualize4(sim_out.scene_id, vis_in, doc, 'left')

sim_out = sim_outs[1]
vis_in = episode_out_to_visualizer_scene_gym_cle(sim_out, mapAPI)
v2 = visualize4(sim_out.scene_id, vis_in, doc, 'right')

############################################ button

# define the callback function
def button_callback(button):
    button_name = button.label
    # button_name = event.source.label
    wait_function(button_name)

# define the wait function
def wait_function(button_name):
    '''TODO: this function store pref.json
    pref.json:
    t1: [(s0,a0), (s1,a1),...] , t2: [(s0,a0),(s1,a1),...] pref
    '''
    print(f"The '{button_name}' button was clicked")

# Define the buttons
left_button = Button(label="Left", button_type="success")
right_button = Button(label="Right", button_type="success")
cannot_tell_button = Button(label="Can't tell", button_type="warning")
same_button = Button(label="Same", button_type="danger")


# Attach the callbacks to the buttons
left_button.on_click(lambda: button_callback(left_button))
right_button.on_click(lambda: button_callback(right_button))
cannot_tell_button.on_click(lambda: button_callback(cannot_tell_button))
same_button.on_click(lambda: button_callback(same_button))

pref = row(left_button, column(same_button, cannot_tell_button), right_button)

doc.add_root(column(row(v1,v2), pref))
# doc2.add_root(v2)

# show(fs)
# print(v1)
# doc.add_root(v1) # open the document in a browser

# show(row(fs))
# cols = []

# for i,sim_out in enumerate(sim_outs): # for each scene
#     vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
#     f, button = visualize3(sim_out.scene_id, vis_in)
#     cols.append(column(f,button))
# grid = gridplot(cols, ncols=4, plot_width=250, plot_height=250)

# show(grid)