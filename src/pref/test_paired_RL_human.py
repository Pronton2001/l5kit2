import json
from multiprocessing import Process
import threading
import time
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

from l5kit.visualization.visualizer.visualizer import visualize, visualize2, visualize3, visualize4, visualize5
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI
from bokeh.models import Button


import os
from pref_db import PrefDB


# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/media/pronton/linux_files/a100code/l5kit/l5kit_dataset/"
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

from l5kit.visualization.visualizer.zarr_utils import episode_out_to_visualizer_scene_gym_cle, zarr_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize
from bokeh.layouts import column, LayoutDOM, row, gridplot
from bokeh.io import curdoc

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
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg, 'train': False, 'return_info': True,}
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
rollout_sim_cfg.num_simulation_steps = 50 + 1
rollout_env = gym.make("L5-CLE-v0", env_config_path=env_config_path, sim_cfg=rollout_sim_cfg, \
                       use_kinematic=True, train=False, return_info=True)

traj1 = []
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
        traj1.append([obs['image'], action])
        # print(np.array(traj1).shape)
        obs, _, done, info = env.step(action)
        if done:
            break

    # The episode outputs are present in the key "sim_outs"
    sim_out = info["sim_outs"][0]
    return sim_out

sim_outs =[]
dataset_path = dm.require(cfg["val_data_loader"]["key"])
zarr_dataset = ChunkedDataset(dataset_path) #TODO: should load 1 time only
zarr_dataset.open()

# define the callback function
def button_callback(button):
    button_name = button.label
    print(f"The '{button_name}' button was clicked")
    if button_name == 'Left':
        pref = [1.0, 0.0]
    elif button_name == 'Right':
        pref = [0.0, 1.0]
    elif button_name == 'Same':
        pref = [0.5, 0.5]
    else:
        pref = None

    wait_function(pref)

pref_db = PrefDB(maxlen=5)
PREFLOGDIR = 'src/pref/preferences/'
idx = 0
# define the wait function
def wait_function(pref):
    global pref_db, idx
    '''TODO: this function store pref.json (disk storage)
    pref.json:
    t1: [(s0,a0), (s1,a1),...] , t2: [(s0,a0),(s1,a1),...] pref
    '''
    t1, t2 = traj1, traj1 #TODO: just for test, after test, change t2 to traj2
    pref_db.append(t1, t2, pref)
    if len(pref_db) >= pref_db.maxlen:
        pref_db.save(PREFLOGDIR + str(idx + 1) + '.pkl.gz')
        print('saved')

        for i in range(len(pref_db)): # del all
            pref_db.del_first()
    # layout.children.remove(button)
    doc_demo.clear()
    print(doc_demo.session_callbacks)
    for cb in doc_demo.session_callbacks:
        doc_demo.remove_periodic_callback(cb)
    print(doc_demo.session_callbacks)

    idx = idx + 1
    PrefInterface(idx)

def save_prefs(pref_db_train, pref_db_val, log_dir = 'src/pref/preferences'):
    train_path = os.path.join(log_dir, 'train.pkl.gz')
    pref_db_train.save(train_path)
    print("Saved training preferences to '{}'".format(train_path))
    val_path = os.path.join(log_dir, 'val.pkl.gz')
    pref_db_val.save(val_path)
    print("Saved validation preferences to '{}'".format(val_path))

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
pref_buttons = row(left_button, column(same_button, cannot_tell_button), right_button)


mapAPI = MapAPI.from_cfg(dm, cfg)

doc_demo = curdoc()
doc_buttons = curdoc()
# doc4 = curdoc()
layout = None

# button = Button(label="Play", button_type="success")
def PrefInterface(scene_idx):
    # global v1, v2, layout, doc_demo, doc_buttons
    # doc_demo = curdoc()
    # doc2 = curdoc()

    start_time = time.time()
    sac_out = rollout_episode(modelA, rollout_env, scene_idx)
    vis_in = episode_out_to_visualizer_scene_gym_cle(sac_out, mapAPI)
    v1 = visualize4(scene_idx, vis_in, doc_demo, 'left')
    # v1 = visualize3(scene_idx, vis_in, button)
    print(time.time() - start_time)
    start_time = time.time()
    human_out = zarr_to_visualizer_scene(zarr_dataset.get_scene_dataset(scene_idx), mapAPI)[:50-2]
    v2 = visualize4(scene_idx, human_out, doc_demo, 'right')
    # v2 = visualize3(scene_idx, human_out, button)
    print(time.time() - start_time)
    # layout1 = v1
    doc_demo.add_root(row(v1, v2))
    # layout2 = v2
    # doc2.add_root(column(v2))
    doc_buttons.add_root(pref_buttons)
    # layout = row(doc1.roots + doc2.roots) # a trick to show 2 diff doc horizontally
    # curdoc().add_root(layout) 

PrefInterface(11)
# doc1.clear()
# doc2.clear()
# PrefInterface(11)
