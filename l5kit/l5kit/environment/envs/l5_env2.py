import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, NamedTuple, Optional

import gym
from l5kit.dataset.ego import EgoDatasetVectorized
import numpy as np
import torch
from gym import spaces
from gym.utils import seeding
from torch.utils.data.dataloader import default_collate

from l5kit.configs import load_config_data
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.dataset.utils import move_to_device, move_to_numpy
from l5kit.environment.kinematic_model import KinematicModel, UnicycleModel
from l5kit.environment.reward import L2DisplacementYawReward, Reward
from l5kit.environment.utils import (calculate_non_kinematic_rescale_params, KinematicActionRescaleParams,
                                     NonKinematicActionRescaleParams)
from l5kit.vectorization.vectorizer_builder import build_vectorizer
from l5kit.simulation.dataset import SimulationConfig, SimulationDataset
from l5kit.simulation.unroll import (ClosedLoopSimulator, ClosedLoopSimulatorModes, SimulationOutputCLE,
                                     UnrollInputOutput)
import logging
logging.basicConfig(filename='/workspace/source/src/log/info.log', level=logging.DEBUG, filemode='w')


#: Maximum acceleration magnitude for kinematic model
MAX_ACC = 6
#: Maximum steer magnitude for kinematic model
MAX_STEER = math.radians(45)


@dataclass
class SimulationConfigGym(SimulationConfig):
    """Defines the default parameters used for the simulation of ego and agents around it in L5Kit Gym.
    Note: num_simulation_steps should be eps_length + 1
    This is because we (may) require to extract the initial speed of the vehicle for the kinematic model
    The speed at start_frame_index is always 0 (not indicative of the true current speed).
    We therefore simulate the episode from (start_frame_index + 1, start_frame_index + eps_length + 1)

    :param use_ego_gt: whether to use GT annotations for ego instead of model's outputs
    :param use_agents_gt: whether to use GT annotations for agents instead of model's outputs
    :param disable_new_agents: whether to disable agents that are not returned at start_frame_index
    :param distance_th_far: if a tracked agent is closed than this value to ego, it will be controlled
    :param distance_th_close: if a new agent is closer than this value to ego, it will be controlled
    :param start_frame_index: the start index of the simulation
    :param num_simulation_steps: the number of step to simulate
    """
    use_ego_gt: bool = False
    use_agents_gt: bool = True
    disable_new_agents: bool = False
    distance_th_far: float = 30.0
    distance_th_close: float = 15.0
    start_frame_index: int = 0
    num_simulation_steps: int = 33


class EpisodeOutputGym(SimulationOutputCLE):
    """This object holds information regarding the simulation output at the end of an episode
    in the gym-compatible L5Kit environment. The output can be used to
    calculate quantitative metrics and provide qualitative visualization.

    :param scene_id: the scene indices
    :param sim_dataset: the simulation dataset
    :param ego_ins_outs: all inputs and outputs for ego (each frame of each scene has only one)
    :param agents_ins_outs: all inputs and outputs for agents (multiple per frame in a scene)
    """

    def __init__(self, scene_id: int, sim_dataset: SimulationDataset,
                 ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]],
                 agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]]):
        """Constructor method
        """
        super(EpisodeOutputGym, self).__init__(scene_id, sim_dataset, ego_ins_outs, agents_ins_outs)

        # Required for Bokeh Visualizer
        simulated_dataset = sim_dataset.scene_dataset_batch[scene_id]
        self.tls_frames = simulated_dataset.dataset.tl_faces
        self.agents_th = simulated_dataset.cfg["raster_params"]["filter_agents_threshold"]


class GymStepOutput(NamedTuple):
    """The output dict of gym env.step

    :param obs: the next observation on performing environment step
    :param reward: the reward of the current step
    :param done: flag to indicate end of episode
    :param info: additional information
    """
    obs: Dict[str, np.ndarray]
    reward: float
    done: bool
    info: Dict[str, Any]


class L5Env2(gym.Env):
    """Custom Environment of L5 Kit that can be registered in OpenAI Gym.

    :param env_config_path: path to the L5Kit environment configuration file
    :param dmg: local data manager object
    :param simulation_cfg: configuration of the L5Kit closed loop simulator
    :param train: flag to determine whether to use train or validation dataset
    :param reward: calculates the reward for the gym environment
    :param cle: flag to enable close loop environment updates
    :param rescale_action: flag to rescale the model action back to the un-normalized action space
    :param use_kinematic: flag to use the kinematic model
    :param kin_model: the kinematic model
    :param return_info: flag to return info when a episode ends
    :param randomize_start: flag to randomize the start frame of episode
    :param simnet_model_path: path to simnet model that controls agents
    """

    def __init__(self, env_config_path: Optional[str] = None, dmg: Optional[LocalDataManager] = None,
                 sim_cfg: Optional[SimulationConfig] = None, train: bool = True,
                 reward: Optional[Reward] = None, cle: bool = True, rescale_action: bool = True,
                 use_kinematic: bool = False, kin_model: Optional[KinematicModel] = None,
                 reset_scene_id: Optional[int] = None, return_info: bool = False,
                 randomize_start: bool = True, simnet_model_path: Optional[str] = None) -> None:
        """Constructor method
        """
        super(L5Env2, self).__init__()

        # Required to register environment
        if env_config_path is None:
            return

        # env config
        dm = dmg if dmg is not None else LocalDataManager(None)
        cfg = load_config_data(env_config_path)
        self.step_time = cfg["model_params"]["step_time"]

        # vectorization
        vectorizer = build_vectorizer(cfg, dm)
        # raster_size = cfg["raster_params"]["raster_size"][0]
        # n_channels = vectorizer.num_channels()

        # load dataset of environment
        self.train = train
        self.overfit = cfg["gym_params"]["overfit"]
        self.randomize_start_frame = randomize_start
        if self.train or self.overfit:
            loader_key = cfg["train_data_loader"]["key"]
        else:
            loader_key = cfg["val_data_loader"]["key"]
        dataset_zarr = ChunkedDataset(dm.require(loader_key)).open()
        # self.dataset = EgoDataset(cfg, dataset_zarr, vectorizer)
        self.dataset = EgoDatasetVectorized(cfg, dataset_zarr, vectorizer)

        # Define action and observation space
        num_future_states = cfg['model_params']['future_num_frames']
        num_future_states = 1
        # Continuous Action Space: gym.spaces.Box (X, Y, Yaw * number of future states)
        # self.action_space = spaces.Box(low=np.array([-4.78996407367,-0.08430253761, -1]), high=np.array([5.95755327367, 0.08809601841, 1]), shape=(num_future_states*3,))
        # self.action_space = spaces.Box(low=np.array([-4.78996407367, -2, -2]), high=np.array([5.95755327367, 2,2]), shape=(num_future_states*3,))
        # self.action_space = spaces.Box(low =np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0,1.0]), dtype=np.float32, shape=(num_future_states*3,))
        # self.action_space = spaces.Box(low =np.array([-10.0, -10.0, -10.0]), high=np.array([10.0, 10.0,10.0]), dtype=np.float32, shape=(num_future_states*3,))
        self.action_space = spaces.Box(low =-np.inf, high=np.inf, dtype=np.float32, shape=(num_future_states*3,))

        # Observation Space: gym.spaces.Dict (image: [n_channels, raster_size, raster_size])
        # obs_shape = (n_channels, raster_size, raster_size)
        # self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)})

        # self.observation_space = spaces.Dict({'image': spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)})

        history_num_frames_ego = cfg["model_params"]["history_num_frames_ego"]
        history_num_frames_agents = cfg["model_params"]["history_num_frames_agents"]
        max_history_num_frames = max(history_num_frames_ego, history_num_frames_agents)
        num_agents = cfg["data_generation_params"]["other_agents_num"]
        MAX_LANES = cfg['data_generation_params']['lane_params']["max_num_lanes"]
        MAX_POINTS_LANES = cfg['data_generation_params']['lane_params']["max_points_per_lane"]
        MAX_POINTS_CW = cfg['data_generation_params']['lane_params']["max_points_per_crosswalk"]
        MAX_CROSSWALKS = cfg['data_generation_params']['lane_params']["max_num_crosswalks"]
        self.observation_space = spaces.Dict({
            # "lanes": lanes,
            # "lanes_availabilities": lanes_availabilities.astype(np.bool),
            # "lanes_mid": lanes_mid,
            # "lanes_mid_availabilities": lanes_mid_availabilities.astype(np.bool),
            # "crosswalks": crosswalks,
            # "crosswalks_availabilities": crosswalks_availabilities.astype(np.bool),
            # "all_other_agents_history_positions": spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

            # "all_other_agents_history_yaws": all_other_agents_history_yaws,
            # "all_other_agents_history_extents": all_other_agents_history_extents,
            # "all_other_agents_history_availability": all_other_agents_history_availability.astype(np.bool),
            # "all_other_agents_future_positions": all_other_agents_future_positions,
            # "all_other_agents_future_yaws": all_other_agents_future_yaws,
            # "all_other_agents_future_extents": all_other_agents_future_extents,
            # "all_other_agents_future_availability": all_other_agents_future_availability.astype(np.bool),
            # "agent_trajectory_polyline": agent_trajectory_polyline,
            # "agent_polyline_availability": agent_polyline_availability.astype(np.bool),
            # "other_agents_polyline": other_agents_polyline,
            # "other_agents_polyline_availability": other_agents_polyline_availability.astype(np.bool),
# @@@@@@@@@@

            'type': spaces.Discrete(17),
            'all_other_agents_types': spaces.MultiDiscrete(nvec=[17]*num_agents),
            'agent_trajectory_polyline': spaces.Box(low=-1e9, high=1e9, shape= (max_history_num_frames + 1, 3), dtype=np.float32),
            'agent_polyline_availability' :  spaces.MultiBinary(n= (max_history_num_frames + 1,)),

      
            'other_agents_polyline': spaces.Box(low=-1e9, high=1e9, shape= (num_agents, max_history_num_frames + 1, 3), dtype=np.float32),
            'other_agents_polyline_availability': spaces.MultiBinary(n= (num_agents, max_history_num_frames + 1,)),

            'lanes_mid':  spaces.Box(low=-1e9, high=1e9, shape= (MAX_LANES, MAX_POINTS_LANES, 3), dtype=np.float32),
            'lanes_mid_availabilities': spaces.MultiBinary(n= (MAX_LANES, MAX_POINTS_LANES)),
            'lanes': spaces.Box(low=-1e9, high=1e9, shape= (MAX_LANES * 2, MAX_POINTS_LANES, 3), dtype=np.float32),
            'crosswalks':spaces.Box(low=-1e9, high=1e9, shape= (MAX_CROSSWALKS, MAX_POINTS_CW, 3), dtype=np.float32),
            'crosswalks_availabilities': spaces.MultiBinary(n= (MAX_CROSSWALKS, MAX_POINTS_CW)), 

            # 'target_yaws': spaces.Box(low=-2*math.pi, high=2*math.pi, shape= (cfg["model_params"]["future_num_frames"], 1), dtype=np.float32), 
            # 'target_positions': spaces.Box(low=-5, high=5, shape= (cfg["model_params"]["future_num_frames"], 2), dtype=np.float32), 
            # 'target_availabilities': spaces.MultiBinary(n=(cfg["model_params"]["future_num_frames"],)),

            # 'all_other_agents_future_positions': spaces.Box(low=-1, high=1, shape= (num_agents, cfg["model_params"]["future_num_frames"], 2), dtype=np.float32), 
            # 'all_other_agents_future_availability': spaces.MultiBinary(n= (num_agents, cfg["model_params"]["future_num_frames"],)), 
            # 'all_other_agents_future_yaws':  spaces.Box(low=0, high=1, shape= (num_agents, cfg["model_params"]["future_num_frames"], 1), dtype=np.float32), 
            # 'history_positions':  spaces.Box(low=-1, high=1, shape=(max_history_num_frames + 1, 2), dtype=np.float32),
            # 'history_yaws': spaces.Box(low=-1, high=1, shape=(max_history_num_frames + 1, 1), dtype=np.float32),
            # 'history_availabilities': spaces.MultiBinary(n= (num_agents, max_history_num_frames + 1,)),
        })
        # self.observation_space =spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

        # Simulator Config within Gym
        self.sim_cfg = sim_cfg if sim_cfg is not None else SimulationConfigGym()
        simulation_model = None
        self.device = torch.device("cpu")
        if not self.sim_cfg.use_agents_gt:
            simulation_model = torch.jit.load(simnet_model_path).to(self.device)
            simulation_model = simulation_model.eval()
        self.simulator = ClosedLoopSimulator(self.sim_cfg, self.dataset, device=self.device,
                                             mode=ClosedLoopSimulatorModes.GYM,
                                             model_agents=simulation_model)

        self.reward = reward if reward is not None else L2DisplacementYawReward()

        self.max_scene_id = cfg["gym_params"]["max_scene_id"]
        if not self.train:
            self.max_scene_id = cfg["gym_params"]["max_val_scene_id"]
            self.randomize_start_frame = False
        if self.overfit:
            self.overfit_scene_id = cfg["gym_params"]["overfit_id"]

        self.cle = cle
        self.rescale_action = rescale_action
        self.use_kinematic = use_kinematic

        if self.use_kinematic:
            self.kin_model = kin_model if kin_model is not None else UnicycleModel()
            self.kin_rescale = self._get_kin_rescale_params()
            #NOTE: Set constants value may be dangerous, training set != testset
            # self.kin_rescale = KinematicActionRescaleParams(steer_scale=0.07853981633974483, acc_scale=0.6000000000000001)
        else:
            self.non_kin_rescale = self._get_non_kin_rescale_params()
            # self.non_kin_rescale = NonKinematicActionRescaleParams(x_mu=0.5837946, x_scale=5.373758673667908, y_mu=0.0018967404, y_scale=0.08619927801191807, yaw_mu=-0.0006447283, yaw_scale=0.04215553868561983)

        # If not None, reset_scene_id is the scene_id that will be rolled out when reset is called
        self.reset_scene_id = reset_scene_id
        if self.overfit:
            self.reset_scene_id = self.overfit_scene_id

        # flag to decide whether to return any info at end of episode
        # helps to limit the IPC
        self.return_info = return_info

        self.seed()

    def seed(self, seed: int = None) -> List[int]:
        """Generate the random seed.

        :param seed: the seed integer
        :return: the output random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        # TODO : add a torch seed for future
        return [seed]

    def set_reset_id(self, reset_id: int = None) -> None:
        """Set the reset_id to unroll from specific scene_id.
        Useful during deterministic evaluation.

        :param reset_id: the scene_id to unroll
        """
        # thailekhanhphuongyeuhuynhminhtri        
        self.reset_scene_id = reset_id

    def reset(self) -> Dict[str, np.ndarray]:
        """ Resets the environment and outputs first frame of a new scene sample.

        :return: the observation of first frame of sampled scene index
        """
        # Define in / outs for new episode scene
        self.agents_ins_outs: DefaultDict[int, List[List[UnrollInputOutput]]] = defaultdict(list)
        self.ego_ins_outs: DefaultDict[int, List[UnrollInputOutput]] = defaultdict(list)

        # Select Scene ID
        self.scene_index = self.np_random.randint(0, self.max_scene_id)
        if self.reset_scene_id is not None:
            self.scene_index = min(self.reset_scene_id, self.max_scene_id - 1)
            self.reset_scene_id += 1

        # Select Frame ID (within bounds of the scene)
        if self.randomize_start_frame:
            scene_length = len(self.dataset.get_scene_indices(self.scene_index))
            self.eps_length = self.sim_cfg.num_simulation_steps or scene_length
            end_frame = scene_length - self.eps_length
            self.sim_cfg.start_frame_index = self.np_random.randint(0, end_frame + 1)

        # Prepare episode scene
        self.sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, [self.scene_index], self.sim_cfg)

        # Reset CLE evaluator
        self.reward.reset()

        # Output first observation
        self.frame_index = 1  # Frame_index 1 has access to the true ego speed
        ego_input = self.sim_dataset.rasterise_frame_batch(self.frame_index)
        self.ego_input_dict = {k: np.expand_dims(v, axis=0) for k, v in ego_input[0].items()}

        # raise ValueError(self.ego_input_dict.keys())
        # Reset Kinematic model
        if self.use_kinematic:
            init_kin_state = np.array([0.0, 0.0, 0.0, self.step_time * ego_input[0]['speed']])
            self.kin_model.reset(init_kin_state)

        # Only output the image attribute
        # raise ValueError(ego_input[0]['lanes'])
        # obs = {'image': ego_input[0]["image"]}
        obs = { 
                'type':ego_input[0]["type"],
                'all_other_agents_types': ego_input[0]["all_other_agents_types"],

                'agent_polyline_availability' :  ego_input[0]["agent_polyline_availability"],
                'agent_trajectory_polyline': ego_input[0]["agent_trajectory_polyline"],

                'other_agents_polyline': ego_input[0]["other_agents_polyline"],
                'other_agents_polyline_availability': ego_input[0]["other_agents_polyline_availability"],

                'crosswalks':ego_input[0]["crosswalks"],
                'crosswalks_availabilities':ego_input[0]["crosswalks_availabilities"],

                'lanes':ego_input[0]["lanes"],
                'lanes_mid': ego_input[0]["lanes_mid"],
                'lanes_mid_availabilities':ego_input[0]["lanes_mid_availabilities"],

                # 'target_positions': ego_input[0]["target_positions"],
                # 'target_yaws': ego_input[0]["target_yaws"],
                # 'target_availabilities': ego_input[0]["target_availabilities"],

                # 'all_other_agents_future_positions': ego_input[0]["all_other_agents_future_positions"],
                # 'all_other_agents_future_availability':ego_input[0]["all_other_agents_future_availability"],
                # 'all_other_agents_future_yaws':ego_input[0]["all_other_agents_future_yaws"],
                }
        # obs = ego_input[0]["image"]
        # raise ValueError(obs)
        return obs

    def step(self, action: np.ndarray) -> GymStepOutput:
        """Inputs the action, updates the environment state and outputs the next frame.

        :param action: the action to perform on current state/frame
        :return: the namedTuple comprising the (next observation, reward, done, info)
            based on the current action
        """
        frame_index = self.frame_index
        next_frame_index = frame_index + 1
        episode_over = next_frame_index == (len(self.sim_dataset) - 1)

        # AGENTS
        if not self.sim_cfg.use_agents_gt:
            agents_input = self.sim_dataset.rasterise_agents_frame_batch(frame_index)
            if len(agents_input):  # agents may not be available
                agents_input_dict = default_collate(list(agents_input.values()))
                with torch.no_grad():
                    agents_output_dict = self.simulator.model_agents(move_to_device(agents_input_dict, self.device))

                # for update we need everything as numpy
                agents_input_dict = move_to_numpy(agents_input_dict)
                agents_output_dict = move_to_numpy(agents_output_dict)

                if self.cle:
                    self.simulator.update_agents(self.sim_dataset, next_frame_index,
                                                 agents_input_dict, agents_output_dict)

                # update input and output buffers
                agents_frame_in_out = self.simulator.get_agents_in_out(agents_input_dict, agents_output_dict,
                                                                       self.simulator.keys_to_exclude)
                self.agents_ins_outs[self.scene_index].append(agents_frame_in_out.get(self.scene_index, []))

        # EGO
        if not self.sim_cfg.use_ego_gt:
            # print('l5env2 action:', action)
            # newAction = action.copy() / 10.0
            newAction = action
            logging.debug(f'l5env2 original action:{newAction}')
            newAction = self._rescale_action(newAction)
            logging.debug(f'l5env2 rescaled action:{newAction}')
            ego_output = self._convert_action_to_ego_output(newAction)
            # print('l5env2 output dict:', ego_output)
            self.ego_output_dict = ego_output
            logging.debug(f'l5env2 ego dict:{ego_output}')

            if self.cle:
                # In closed loop training, the raster is updated according to predicted ego positions.
                self.simulator.update_ego(self.sim_dataset, next_frame_index, self.ego_input_dict, self.ego_output_dict)

            ego_frame_in_out = self.simulator.get_ego_in_out(self.ego_input_dict, self.ego_output_dict,
                                                             self.simulator.keys_to_exclude)
            self.ego_ins_outs[self.scene_index].append(ego_frame_in_out[self.scene_index])

        # generate simulated_outputs
        simulated_outputs = SimulationOutputCLE(self.scene_index, self.sim_dataset, self.ego_ins_outs,
                                                self.agents_ins_outs)

        # reward calculation
        reward = self.reward.get_reward(self.frame_index, [simulated_outputs])

        # done is True when episode ends
        done = episode_over

        # Optionally we can pass additional info
        # We are using "info" to output rewards and simulated outputs (during evaluation)
        info: Dict[str, Any]
        info = {'reward_tot': reward["total"], 'reward_dist': reward["distance"], 'reward_yaw': reward["yaw"]}
        if done and self.return_info:
            info = {"sim_outs": self.get_episode_outputs(), "reward_tot": reward["total"],
                    "reward_dist": reward["distance"], "reward_yaw": reward["yaw"]}

        # Get next obs
        self.frame_index += 1
        obs = self._get_obs(self.frame_index, episode_over)

        # return obs, reward, done, info
        return GymStepOutput(obs, reward["total"], done, info)

    def get_episode_outputs(self) -> List[EpisodeOutputGym]:
        """Generate and return the outputs at the end of the episode.

        :return: List of episode outputs
        """
        episode_outputs = [EpisodeOutputGym(self.scene_index, self.sim_dataset, self.ego_ins_outs,
                                            self.agents_ins_outs)]
        return episode_outputs

    def render(self) -> None:
        """Render a frame during the simulation
        """
        raise NotImplementedError

    def _get_obs(self, frame_index: int, episode_over: bool) -> Dict[str, np.ndarray]:
        """Get the observation corresponding to a given frame index in the scene.

        :param frame_index: the index of the frame which provides the observation
        :param episode_over: flag to determine if the episode is over
        :return: the observation corresponding to the frame index
        """
        if episode_over:
            frame_index = 0  # Dummy final obs (when episode_over)

        ego_input = self.sim_dataset.rasterise_frame_batch(frame_index)
        self.ego_input_dict = {k: np.expand_dims(v, axis=0) for k, v in ego_input[0].items()}
        obs = { 
                'type':ego_input[0]["type"],
                'all_other_agents_types': ego_input[0]["all_other_agents_types"],

                'agent_polyline_availability' :  ego_input[0]["agent_polyline_availability"],
                'agent_trajectory_polyline': ego_input[0]["agent_trajectory_polyline"],

                'other_agents_polyline': ego_input[0]["other_agents_polyline"],
                'other_agents_polyline_availability': ego_input[0]["other_agents_polyline_availability"],

                'crosswalks':ego_input[0]["crosswalks"],
                'crosswalks_availabilities':ego_input[0]["crosswalks_availabilities"],

                'lanes':ego_input[0]["lanes"],
                'lanes_mid': ego_input[0]["lanes_mid"],
                'lanes_mid_availabilities':ego_input[0]["lanes_mid_availabilities"],

                # 'type': torch.as_tensor(ego_input[0]["type"]).to(self.device),
                # 'all_other_agents_types': torch.as_tensor(ego_input[0]["all_other_agents_types"]).to(self.device),

                # 'agent_polyline_availability' :  torch.as_tensor(ego_input[0]["agent_polyline_availability"]).to(self.device),
                # 'agent_trajectory_polyline': torch.as_tensor(ego_input[0]["agent_trajectory_polyline"]).to(self.device),

                # 'other_agents_polyline': torch.as_tensor(ego_input[0]["other_agents_polyline"]).to(self.device),
                # 'other_agents_polyline_availability': torch.as_tensor(ego_input[0]["other_agents_polyline_availability"]).to(self.device),

                # 'crosswalks':torch.as_tensor(ego_input[0]["crosswalks"]).to(self.device),
                # 'crosswalks_availabilities':torch.as_tensor(ego_input[0]["crosswalks_availabilities"]).to(self.device),

                # 'lanes':torch.as_tensor(ego_input[0]["lanes"]).to(self.device),
                # 'lanes_mid': torch.as_tensor(ego_input[0]["lanes_mid"]).to(self.device),
                # 'lanes_mid_availabilities':torch.as_tensor(ego_input[0]["lanes_mid_availabilities"]).to(self.device),

                # 'target_availabilities': ego_input[0]["target_availabilities"],
                # 'target_positions': ego_input[0]["target_positions"],
                # 'target_yaws': ego_input[0]["target_yaws"],

                # 'all_other_agents_future_positions': ego_input[0]["all_other_agents_future_positions"],
                # 'all_other_agents_future_availability':ego_input[0]["all_other_agents_future_availability"],
                # 'all_other_agents_future_yaws':ego_input[0]["all_other_agents_future_yaws"],
                }
        # raise ValueError(obs)
        # obs = {"image": ego_input[0]["image"]}

        # obs = ego_input[0]["image"]
        return obs

    def _rescale_action(self, action: np.ndarray) -> np.ndarray:
        """Rescale the input action back to the un-normalized action space. PPO and related algorithms work well
        with normalized action spaces. The environment receives a normalized action and we un-normalize it back to
        the original action space for environment updates.

        :param action: the normalized action
        :return: the unnormalized action
        """
        newAction = action.copy()
        if self.rescale_action:
            if self.use_kinematic:
                newAction[0] = self.kin_rescale.steer_scale * action[0]
                newAction[1] = self.kin_rescale.acc_scale * action[1]
            else:
                newAction[0] = self.non_kin_rescale.x_mu + self.non_kin_rescale.x_scale * action[0]
                newAction[1] = self.non_kin_rescale.y_mu + self.non_kin_rescale.y_scale * action[1]
                newAction[2] = self.non_kin_rescale.yaw_mu + self.non_kin_rescale.yaw_scale * action[2]
            return newAction
        return newAction

    def _get_kin_rescale_params(self) -> KinematicActionRescaleParams:
        """Determine the action un-normalization parameters for the kinematic model
        from the current dataset in the L5Kit environment.

        :return: Tuple of the action un-normalization parameters for kinematic model
        """
        global MAX_ACC, MAX_STEER
        return KinematicActionRescaleParams(MAX_STEER * self.step_time, MAX_ACC * self.step_time)

    def _get_non_kin_rescale_params(self, max_num_scenes: int = 10) -> NonKinematicActionRescaleParams:
        """Determine the action un-normalization parameters for the non-kinematic model
        from the current dataset in the L5Kit environment.

        :param max_num_scenes: maximum number of scenes to consider to determine parameters
        :return: Tuple of the action un-normalization parameters for non-kinematic model
        """
        scene_ids = list(range(self.max_scene_id)) if not self.overfit else [self.overfit_scene_id]
        if len(scene_ids) > max_num_scenes:  # If too many scenes, CPU crashes
            scene_ids = scene_ids[:max_num_scenes]
        sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_ids, self.sim_cfg)
        return calculate_non_kinematic_rescale_params(sim_dataset)

    def _convert_action_to_ego_output(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert the input action into ego output format.

        :param action: the input action provided by policy
        :return: action in ego output format, a numpy dict with keys 'positions' and 'yaws'
        """
        if self.use_kinematic:
            data_dict = self.kin_model.update(action[:2])
        else:
            # [batch_size=1, num_steps, (X, Y, yaw)]
            data = action.reshape(1, 1, 3)
            pred_positions = data[:, :, :2]
            # [batch_size, num_steps, 1->(yaw)]
            pred_yaws = data[:, :, 2:3]
            data_dict = {"positions": pred_positions, "yaws": pred_yaws}
        return data_dict