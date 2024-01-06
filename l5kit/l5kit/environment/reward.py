from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from l5kit.cle.metric_set import L5MetricSet
from l5kit.environment.gym_metric_set import L2DisplacementYawMetricSet
from l5kit.environment.gym_metric_set import CLEMetricSetForReward
from l5kit.simulation.unroll import SimulationOutputCLE
import logging
from src.constant import SRC_PATH
logging.basicConfig(filename=SRC_PATH + 'src/log/info.log', level=logging.DEBUG, filemode='w')


class Reward(ABC):
    """Base class interface for gym environment reward."""
    #: The prefix that will identify this reward class
    reward_prefix: str

    @abstractmethod
    def reset(self) -> None:
        """Reset the reward state when new episode starts.
        """
        raise NotImplementedError

    @abstractmethod
    def get_reward(self, frame_index: int, simulated_outputs: List[SimulationOutputCLE]) -> Dict[str, float]:
        """Return the reward at a particular time-step during the episode.

        :param frame_index: the frame index for which reward is to be calculated
        :param simulated_outputs: the object contain the ego target and prediction attributes
        :return: reward at a particular frame index (time-step) during the episode containing total reward
            and individual components that make up the reward.
        """
        raise NotImplementedError


class L2DisplacementYawReward(Reward):
    """This class is responsible for calculating a reward based on
    (1) L2 displacement error on the (x, y) coordinates
    (2) Closest angle error on the yaw coordinate
    during close loop simulation within the gym-compatible L5Kit environment.

    :param reward_prefix: the prefix that will identify this reward class
    :param metric_set: the set of metrics to compute
    :param enable_clip: flag to determine whether to clip reward
    :param rew_clip_thresh: the threshold to clip the reward
    :param use_yaw: flag to penalize the yaw prediction
    :param yaw_weight: weight of the yaw error
    """

    def __init__(self, reward_prefix: str = "L2DisplacementYaw", metric_set: Optional[L5MetricSet] = None,
                 enable_clip: bool = True, rew_clip_thresh: float = 15.0,
                 use_yaw: Optional[bool] = True, yaw_weight: Optional[float] = 1.0) -> None:
        """Constructor method
        """
        self.reward_prefix = reward_prefix
        # Metric Set
        self.metric_set = metric_set if metric_set is not None else L2DisplacementYawMetricSet()

        # Verify that error metrics necessary for reward calculation are present in the metric set
        if 'yaw_error_closest_angle' not in self.metric_set.evaluation_plan.metrics_dict():
            raise RuntimeError('\'yaw_error_closest_angle\' missing in metric set')
        if 'displacement_error_l2' not in self.metric_set.evaluation_plan.metrics_dict():
            raise RuntimeError('\'displacement_error_l2\' missing in metric set')

        self.use_yaw = use_yaw
        self.yaw_weight = yaw_weight

        self.enable_clip = enable_clip
        self.rew_clip_thresh = rew_clip_thresh

    def reset(self) -> None:
        """Reset the closed loop evaluator when a new episode starts.
        """
        self.metric_set.reset()

    @staticmethod
    def slice_simulated_output(index: int, simulated_outputs: List[SimulationOutputCLE]) -> List[SimulationOutputCLE]:
        """ Slice the simulated output at a particular frame index.
        This prevent calculating metric over all frames.

        :param index: the frame index at which the simulation outputs is to be sliced
        :param simulated_outputs: the object contain the ego target and prediction attributes
        :return: the sliced simulation output
        """
        # Only the simulated and recorded ego states are used for metric calculation
        simulated_outputs[0].recorded_ego_states = simulated_outputs[0].recorded_ego_states[index:index + 1]
        simulated_outputs[0].simulated_ego_states = simulated_outputs[0].simulated_ego_states[index:index + 1]
        return simulated_outputs

    def get_reward(self, frame_index: int, simulated_outputs: List[SimulationOutputCLE]) -> Dict[str, float]:
        """Get the reward for the given step in close loop training.

        :param frame_index: the frame index for which reward is to be calculated
        :param simulated_outputs: the object contain the ego target and prediction attributes
        :return: the dictionary containing total reward and individual components that make up the reward
        """
        scene_id = simulated_outputs[0].scene_id

        # Get the simulated output value at frame index + 1
        simulated_outputs = self.slice_simulated_output(frame_index + 1, simulated_outputs)

        # Evaluate metrics on the sliced simulated output
        self.metric_set.evaluate(simulated_outputs)
        scene_metrics = self.metric_set.evaluator.scene_metric_results[scene_id]
        dist_error = scene_metrics['displacement_error_l2']
        yaw_error = self.yaw_weight * scene_metrics['yaw_error_closest_angle']

        # clip the distance error (in x, y) only, not the yaw error (yaw error is bounded).
        dist_reward = float(-dist_error.item())
        if self.enable_clip:
            dist_reward = max(-self.rew_clip_thresh, -dist_error.item())

        # use yaw
        yaw_reward = 0.0
        if self.use_yaw:
            yaw_reward -= yaw_error.item()

        # Total reward
        total_reward = dist_reward + yaw_reward

        reward_dict = {"total": total_reward, "distance": dist_reward, "yaw": yaw_reward}
        return reward_dict

class CLEMetricReward(Reward):
    """This class is responsible for calculating a reward based on
    (1) L2 displacement error on the (x, y) coordinates
    (2) Closest angle error on the yaw coordinate
    during close loop simulation within the gym-compatible L5Kit environment.

    :param reward_prefix: the prefix that will identify this reward class
    :param metric_set: the set of metrics to compute
    :param enable_clip: flag to determine whether to clip reward
    :param rew_clip_thresh: the threshold to clip the reward
    :param use_yaw: flag to penalize the yaw prediction
    :param yaw_weight: weight of the yaw error
    """

    def __init__(self, reward_prefix: str = "CLE", metric_set: Optional[L5MetricSet] = None,
                 enable_clip: bool = True, rew_clip_thresh: float = 20.0, yaw_weight = 1.0, dist_weight = 1.0, cf_weight = 5, cr_weight = 5, cs_weight = 5) -> None:
        """Constructor method
        """
        self.reward_prefix = reward_prefix
        # Metric Set
        self.cle_metric_set = CLEMetricSetForReward()

        # Verify that error metrics necessary for reward calculation are present in the metric set
        # if 'yaw_error_closest_angle' not in self.cle_metric_set.evaluation_plan.metrics_dict():
        #     raise RuntimeError('\'yaw_error_closest_angle\' missing in metric set')
        # if 'displacement_error_l2' not in self.cle_metric_set.evaluation_plan.metrics_dict():
        #     raise RuntimeError('\'displacement_error_l2\' missing in metric set')

        self.yaw_weight = yaw_weight
        self.dist_weight = dist_weight
        self.cf_weight = cf_weight
        self.cr_weight = cr_weight
        self.cs_weight = cs_weight

        self.enable_clip = enable_clip
        self.rew_clip_thresh = rew_clip_thresh

    def reset(self) -> None:
        """Reset the closed loop evaluator when a new episode starts.
        """
        self.cle_metric_set.reset()

    @staticmethod
    def slice_simulated_output(index: int, simulated_outputs: List[SimulationOutputCLE]) -> List[SimulationOutputCLE]:
        """ Slice the simulated output at a particular frame index.
        This prevent calculating metric over all frames.

        :param index: the frame index at which the simulation outputs is to be sliced
        :param simulated_outputs: the object contain the ego target and prediction attributes
        :return: the sliced simulation output
        """
        # Only the simulated and recorded ego states are used for metric calculation
        simulated_outputs[0].recorded_ego_states = simulated_outputs[0].recorded_ego_states[index:index + 1]
        simulated_outputs[0].simulated_ego_states = simulated_outputs[0].simulated_ego_states[index:index + 1]
        return simulated_outputs

    @staticmethod
    def slice_simulated_output_woRecord(index: int, simulated_outputs: List[SimulationOutputCLE]) -> List[SimulationOutputCLE]:
        """ Slice the simulated output at a particular frame index.
        This prevent calculating metric over all frames.

        :param index: the frame index at which the simulation outputs is to be sliced
        :param simulated_outputs: the object contain the ego target and prediction attributes
        :return: the sliced simulation output
        """
        # Only the simulated and recorded ego states are used for metric calculation
        simulated_outputs[0].recorded_ego_states = simulated_outputs[0].recorded_ego_states
        simulated_outputs[0].simulated_ego_states = simulated_outputs[0].simulated_ego_states[index:index + 1]
        return simulated_outputs
    

    def get_reward(self, frame_index: int, simulated_outputs: List[SimulationOutputCLE]) -> Dict[str, float]:
        """Get the reward for the given step in close loop training.

        :param frame_index: the frame index for which reward is to be calculated
        :param simulated_outputs: the object contain the ego target and prediction attributes
        :return: the dictionary containing total reward and individual components that make up the reward
        """
        scene_id = simulated_outputs[0].scene_id

        # Get the simulated output value at frame index + 1
        simulated_outputs = self.slice_simulated_output(frame_index + 1, simulated_outputs)

        # Evaluate metrics on the sliced simulated output
        self.cle_metric_set.evaluate(simulated_outputs)
        scene_metrics = self.cle_metric_set.evaluator.scene_metric_results[scene_id]
        scene_metrics_yaw = self.cle_metric_set.evaluator.scene_metric_results[scene_id]
        dist_error = scene_metrics['displacement_error_l2']
        yaw_error = scene_metrics_yaw['yaw_error_closest_angle']
        # d2r_error = scene_metrics['distance_to_reference_trajectory']
        cf_error = scene_metrics['collision_front']
        cr_error = scene_metrics['collision_rear']
        cs_error = scene_metrics['collision_side']
        # p_ego = scene_metrics['simulated_minus_recorded_ego_speed']
        # logging.debug(f'{drt_error, cf_error, cr_error}, ')
        # logging.debug(f'len {len(drt_error)}')


        # clip the distance error (in x, y) only, not the yaw error (yaw error is bounded).

        dist_reward    = float(-dist_error.item()) * self.dist_weight
        # d2r_reward     = float(-d2r_error.item()) * self.d2r_weight
        yaw_reward     = float(-yaw_error.item()) * self.yaw_weight
        cf_reward      = float(-cf_error.item()) * self.cf_weight
        cr_reward      = float(-cr_error.item()) * self.cr_weight
        cs_reward      = float(-cs_error.item()) * self.cs_weight

        if self.enable_clip:
            dist_reward = max(-self.rew_clip_thresh, dist_reward)
            # d2r_reward  = max(-self.rew_clip_thresh, d2r_reward)
            # if len(drt_error) == 0:
            #     drt_reward = 0
            # else: 

        # Total reward
        total_reward = dist_reward + yaw_reward + cf_reward + cr_reward + cs_reward# dist_reward and d2r_reward is the same???

        reward_dict = {"total": total_reward, "distance": dist_reward, "yaw": yaw_reward, 'cf': cf_reward, 'cr': cr_reward, 'cs': cs_reward}
        # logging.debug(f'reward dict {reward_dict}')
        # print(f'reward dict {reward_dict}')
        return reward_dict