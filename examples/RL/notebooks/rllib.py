from typing import Dict
from l5kit.environment.envs.l5_env import GymStepOutput, SimulationConfigGym, L5Env
# Dataset is assumed to be on the folder specified
# in the L5KIT_DATA_FOLDER environment variable
import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.models import ModelCatalog
from wrapper import L5EnvWrapper

# get environment config
env_config_path = '../gym_config.yaml'
from l5kit.configs import load_config_data
cfg = load_config_data(env_config_path)



train_sim_cfg = SimulationConfigGym()
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}
from ray import tune

# Register , how your env should be constructed (always with 5, or you can take values from the `config` EnvContext object):
env_kwargs = {'env_config_path': env_config_path, 'use_kinematic': True, 'sim_cfg': train_sim_cfg}

tune.register_env("L5-CLE-V0", lambda config: L5Env(**env_kwargs))
tune.register_env("L5-CLE-V1", lambda config: L5EnvWrapper(L5Env(**env_kwargs)))

from stable_baselines3.common.callbacks import CheckpointCallback
from l5kit.environment.callbacks import L5KitEvalCallback

# callback_list = []

# Save Model Periodically
save_freq = 100000
save_path = './logs/'
output = 'PPO'
train_envs = 4
eval_envs = 1
# checkpoint_callback = CheckpointCallback(save_freq=(save_freq // train_envs), save_path=save_path, \
#                                          name_prefix=output)
# callback_list.append(checkpoint_callback)

# # Eval Model Periodically
# eval_freq = 100000
# n_eval_episodes = 1
# val_eval_callback = L5KitEvalCallback("L5-CLE-V1", eval_freq=(eval_freq // train_envs), \
#                                       n_eval_episodes=n_eval_episodes, n_eval_envs=eval_envs)
# callback_list.append(val_eval_callback)

ray.init(num_cpus=3, ignore_reinit_error=True, log_to_driver=False)
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 2
config["framework"] = 'tf2'
config['_disable_preprocessor_api'] = True,
config["model"]["dim"] = 112
config["model"]["conv_filters"] = [[64, 7, 3], [32, 11, 3], [256, 11, 3]]
# config["log_level"] = "WARN",
config["num_envs_per_worker"] = train_envs
# config["callbacks"] = callback_list
# config["model"]["conv_activation"] = 'relu'
# config["model"]["post_fcnet_hiddens"] =  [256]
# config["model"]["post_fcnet_activation"] = "relu",
algo = ppo.PPO(config=config, env="L5-CLE-V1")

for i in range(2):
    result = algo.train()
    print(pretty_print(result))
