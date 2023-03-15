import gym
import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
from ray import tune
import psutil
from ray.tune.logger import UnifiedLogger

print(psutil.cpu_count())
ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
# algo = ppo.PPO(config=config, env="CartPole-v0")

# Can optionally call algo.restore(path) to load a checkpoint.

env_class = 'CartPole-v0'
# checkpoint_path = '/home/pronton/rl/l5kit/examples/RL/notebooks/check_points/checkpoint_000020'

algo = ppo.PPO(config=config, env=env_class)
# algo.restore(checkpoint_path)

# analysis = tune.run("PPO",
#     config={
#         "env": "CartPole-v0",
#         "num_gpus": 0,
#         "num_workers": 4,
#     },
#     stop={
#         "episode_reward_mean": 200,
#         "time_total_s": 600
#     },
#     checkpoint_freq=1,
#     checkpoint_at_end=True,
#     local_dir='./logs'
# )
import ray
from ray import air, tune

config_param_space = {
    "env": "CartPole-v0",
    "num_gpus": 0,
    "num_workers": 1,
    "lr": tune.grid_search([0.01, 0.001, 0.0001]),
}
log_dir = '/home/pronton/rl/l5kit/examples/RL/notebooks/logs/PPO'

# tune.Tuner(
#     "PPO",
#     run_config=air.RunConfig(
#         stop={"episode_reward_mean": 200},
#         local_dir=log_dir,
#         checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True)
#         ),
#     param_space=config_param_space).fit()
tuner = tune.Tuner.restore(
    path=log_dir
)
tuner.fit()



# checkpoint_path =checkpoint_dir
# model = ppo.PPOTrainer(env="NameOfYourEnv", config=config)
# model.restore(checkpoint_path)
# env = <create your env> 

# obs  = env.reset()

# for i in range(1, 1000):
#    # Perform one iteration of training the policy with PPO
#    result = algo.train()
#    print(pretty_print(result))

#    if i % 10 == 0:
#        checkpoint = algo.save(checkpoint_dir= './check_points/')
#        print("checkpoint saved at", checkpoint)
# instantiate env class
env =gym.make(env_class)

# run until episode ends
episode_reward = 0
done = False
obs = env.reset()
while not done:
    action = algo.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    env.render()

print(episode_reward)

# Also, in case you have trained a model outside of ray/RLlib and have created
# an h5-file with weight values in it, e.g.
# my_keras_model_trained_outside_rllib.save_weights("model.h5")
# (see: https://keras.io/models/about-keras-models/)

# ... you can load the h5-weights into your Algorithm's Policy's ModelV2
# (tf or torch) by doing:
# algo.import_model("my_weights.h5")
# NOTE: In order for this to work, your (custom) model needs to implement
# the `import_from_h5` method.
# See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
# for detailed examples for tf- and torch policies/models.