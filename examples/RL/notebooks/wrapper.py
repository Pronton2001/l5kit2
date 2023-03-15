from typing import Dict
import gym
from l5kit.environment.envs.l5_env import GymStepOutput
import numpy as np


class L5EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.n_channels = 7
        self.raster_size = 84
        obs_shape = (self.raster_size, self.raster_size, self.n_channels)
        self.observation_space =gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def step(self, action:  np.ndarray) -> GymStepOutput:
        # return GymStepOutput(obs, reward["total"], done, info)
        output =  self.env.step(action)
        onlyImageState = output.obs['image'].reshape(self.raster_size, self.raster_size, self.n_channels)
        return GymStepOutput(onlyImageState, output.reward, output.done, output.info)

    def reset(self) -> Dict[str, np.ndarray]:
        return self.env.reset()['image'].reshape(self.raster_size, self.raster_size, self.n_channels)