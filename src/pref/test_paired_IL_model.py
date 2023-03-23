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
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator

from l5kit.visualization.visualizer.zarr_utils import simulation_out_to_visualizer_scene
from l5kit.visualization.visualizer.visualizer import visualize, visualize2, visualize3, visualize4, visualize5
from bokeh.io import output_notebook, show
from l5kit.data import MapAPI
from bokeh.models import Button

from collections import defaultdict
import os
from stable_baselines3 import SAC
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/media/pronton/linux_files/a100code/l5kit/l5kit_dataset/"
dm = LocalDataManager(None)
# get config
cfg = load_config_data("/home/pronton/rl/l5kit/examples/urban_driver/config.yaml")
model_path = "/home/pronton/rl/l5kit/examples/urban_driver/BPTT.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path).to(device)
# model = SAC.load("/home/pronton/rl/l5kit/examples/RL/gg colabs/logs/SAC_640000_steps.zip")
model = model.eval()
torch.set_grad_enabled(False)
# ===== INIT DATASET
eval_cfg = cfg["val_data_loader"]
eval_zarr = ChunkedDataset(dm.require(eval_cfg["key"])).open()
vectorizer = build_vectorizer(cfg, dm)
eval_dataset = EgoDatasetVectorized(cfg, eval_zarr, vectorizer)
print(eval_dataset)
num_scenes_to_unroll = 2
num_simulation_steps = 10
# ==== DEFINE CLOSED-LOOP SIMULATION
sim_cfg = SimulationConfig(use_ego_gt=False, use_agents_gt=True, disable_new_agents=True,
                           distance_th_far=500, distance_th_close=50, num_simulation_steps=num_simulation_steps,
                           start_frame_index=0, show_info=True)

sim_loop = ClosedLoopSimulator(sim_cfg, eval_dataset, device, model_ego=model, model_agents=None)
# ==== UNROLL
scenes_to_unroll = list(range(0, len(eval_zarr.scenes), len(eval_zarr.scenes)//num_scenes_to_unroll))
sim_outs = sim_loop.unroll(scenes_to_unroll)
mapAPI = MapAPI.from_cfg(dm, cfg)
from bokeh.layouts import column, LayoutDOM, row, gridplot
from bokeh.io import curdoc

button = Button(label="Play", button_type="success")
############################################ 2 scene
doc = curdoc()
# for sim_out in sim_outs[:1]: # for each scene
sim_out = sim_outs[0]
vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
# v1 = visualize4(sim_out.scene_id, vis_in, doc, 'left')
v1 = visualize3(sim_out.scene_id, vis_in, button)

sim_out = sim_outs[1]
vis_in = simulation_out_to_visualizer_scene(sim_out, mapAPI)
# v2 = visualize3(sim_out.scene_id, vis_in, doc, 'right')
v2 = visualize3(sim_out.scene_id, vis_in, button)

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

doc.add_root(column(button,row(v1,v2), pref))
# show(column(row(v1,v2), pref))
# show(column(row(v1,v2)))
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