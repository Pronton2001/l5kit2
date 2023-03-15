from typing import Dict
import gym
from l5kit.configs.config import load_config_data
from l5kit.environment.envs.l5_env import GymStepOutput
from l5kit.vectorization.vectorizer_builder import build_vectorizer
import numpy as np
from l5kit.data import ChunkedDataset, LocalDataManager


class L5EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # env config
        dm =  LocalDataManager(None)
        cfg = load_config_data(env_config_path)
        self.step_time = cfg["model_params"]["step_time"]

        # rasterisation
        rasterizer = build_vectorizer(cfg, dm)
        raster_size = cfg["raster_params"]["raster_size"][0]
        n_channels = rasterizer.num_channels()


        self.env = env
        self.n_channels = 7
        self.raster_size = 84
        obs_shape = (self.raster_size, self.raster_size, self.n_channels)# change to vectorization
        self.observation_space =gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def step(self, action:  np.ndarray) -> GymStepOutput:
        # return GymStepOutput(obs, reward["total"], done, info)
        output =  self.env.step(action)
        onlyImageState = output.obs['image'].reshape(self.raster_size, self.raster_size, self.n_channels)
        return GymStepOutput(onlyImageState, output.reward, output.done, output.info)

    def reset(self) -> Dict[str, np.ndarray]:
        return self.env.reset()['image'].reshape(self.raster_size, self.raster_size, self.n_channels)