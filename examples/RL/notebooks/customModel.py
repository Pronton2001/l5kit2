from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
import torch
from torch.nn import functional as F

from l5kit.planning.vectorized.open_loop_model import VectorizedModel, CustomVectorizedModel
class TorchGNCNN(TorchModelV2, nn.Module):
    """
    Simple Convolution agent that calculates the required linear output layer
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # raise ValueError(num_outputs)
        self._num_objects = obs_space.shape[2] # num_of_channels of input, size x size x channels
        self._num_actions = num_outputs
        self._feature_dim = model_config["custom_model_config"]['feature_dim']

        # linear_flatten = np.prod(obs_space.shape[:2])*64

        self.network = nn.Sequential(
            nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=1568, out_features=self._feature_dim),
            # layer_init(nn.Conv2d(self._num_objects, 32, 3, padding=1)),
            # nn.ReLU(),
            # layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            # nn.ReLU(),
            # nn.Flatten(),
            # layer_init(nn.Linear(linear_flatten, 1024)),
            # nn.ReLU(),
            # layer_init(nn.Linear(1024, 512)),
            # nn.ReLU(),
        )

        self._actor_head = nn.Sequential(
            # layer_init(nn.Linear(512, 256), std=0.01),
            # nn.ReLU(),
            # layer_init(nn.Linear(256, self._num_actions), std=0.01)
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self._num_actions),
        )

        self._critic_head = nn.Sequential(
            # layer_init(nn.Linear(512, 1), std=0.01)
            nn.Linear(self._feature_dim, 1),
        )

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
        network_output = self.network(obs_transformed)
        value = self._critic_head(network_output)
        self._value = value.reshape(-1)
        logits = self._actor_head(network_output)
        return logits, state

    def value_function(self):
        return self._value
# class TorchAttentionSACModel(TorchModelV2, nn.Module):

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers.children():
            nn.init.zeros_(layer.bias)
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TorchAttentionModel2(TorchModelV2, nn.Module):
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # raise ValueError(num_outputs)

        cfg = model_config["custom_model_config"]['cfg']
        d_model = 256
        self._num_actions = num_outputs
        self._actor_head = MLP(d_model, d_model * 4, output_dim= num_outputs, num_layers=3)
        self._critic_head = MLP(d_model, d_model * 4, output_dim= 1, num_layers=1)

        self.feature_extractor= CustomVectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

    def forward(self, input_dict, state, seq_lens):
        obs_transformed = input_dict['obs']
        out, attns = self.feature_extractor(obs_transformed)
        logits = self._actor_head(out).view(-1, self._num_actions)
        self._value = self._critic_head(out).view(-1)
        return logits, state

    def value_function(self):
        return self._value

class TorchAttentionModel(TorchModelV2, nn.Module):
    """
    Attention Model agent
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # raise ValueError(num_outputs)

        cfg = model_config["custom_model_config"]['cfg']
        weights_scaling = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        self._num_predicted_frames = cfg["model_params"]["future_num_frames"]
        self._num_predicted_params = len(weights_scaling)
        weights_scaling_critic = [1.0]
        _num_predicted_params_critic = len(weights_scaling_critic)

        self._actor_head = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=self._num_predicted_params * self._num_predicted_frames, # N (X,Y,Yaw) 72
            weights_scaling=weights_scaling, # 6
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

        self._critic_head = VectorizedModel(
            history_num_frames_ego=cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=cfg["model_params"]["history_num_frames_agents"],
            num_targets=_num_predicted_params_critic, # just 1 (X,Y,Yaw)
            weights_scaling=weights_scaling_critic,
            criterion=nn.L1Loss(reduction="none"),
            global_head_dropout=cfg["model_params"]["global_head_dropout"],
            disable_other_agents=cfg["model_params"]["disable_other_agents"],
            disable_map=cfg["model_params"]["disable_map"],
            disable_lane_boundaries=cfg["model_params"]["disable_lane_boundaries"])

    def forward(self, input_dict, state, seq_lens):
        # obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [b, size, size, channels]
        # raise ValueError(input_dict)
        obs_transformed = input_dict['obs']
        # raise ValueError(input_dict['obs'])
        # network_output = self.network(obs_transformed)
        logits = self._actor_head(obs_transformed)
        # raise ValueError(str(logits['positions'].shape))
        # logits = torch.cat((logits['positions'], logits['yaws']),axis=-1)
        logits = logits.view(-1, int(self._num_predicted_frames * self._num_predicted_params))
        # raise ValueError(logits.shape)
        value = self._critic_head(obs_transformed)
        self._value = value.view(-1)
        # raise ValueError(logits.shape)
        # raise ValueError('positions: ' + str(logits['positions'].shape) + 'yaw:' + str(logits['yaws'].shape))
        return logits, state

    def value_function(self):
        return self._value

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from tensorflow import keras


tf1, tf, tfv = try_import_tf()
layers = tf.keras.layers

class TFGNNCNN(TFModelV2):
    """Custom model for policy gradient algorithms."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(TFGNNCNN, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self._num_objects = obs_space.shape[2] # num_of_channels of input, size x size x channels
        self._num_actions = num_outputs
        self._feature_dim = model_config["custom_model_config"]['feature_dim']
        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        layer_1 = layers.Conv2D(64,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False, kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_2 = layers.GroupNormalization(64)(layer_1)
        layer_3 = layers.ReLU()(layer_2)
        layer_4 = layers.MaxPool2D(pool_size=(2,2), strides=2)(layer_3) # x
        layer_5 = layers.Conv2D(32,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False, kernel_initializer=normc_initializer(1.0))(layer_4)
        layer_6 = layers.GroupNormalization(32)(layer_5),
        # raise ValueError(layer_6.__repr__())
        layer_7 = layers.ReLU()(layer_6)
        layer_8 = layers.MaxPool2D(pool_size=(2,2), strides=2)(layer_7)
        layer_9 = layers.Flatten()(layer_8)
        layer_10 = layers.Dense(self._feature_dim)(layer_9)
            # nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # nn.GroupNorm(4, 64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # nn.GroupNorm(2, 32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Flatten(),
            # nn.Linear(in_features=1568, out_features=self._feature_dim),

        actor_out_1 = layers.Dense( 256, kernel_initializer=normc_initializer(0.01),)(layer_10)
        actor_out_2= layers.ReLU()(actor_out_1)
        actor_out_3 = layers.Dense( num_outputs, kernel_initializer=normc_initializer(0.01),name='critic_out')(actor_out_2)
        critic_out = tf.keras.layers.Dense( 1, name="value_out", activation=None, kernel_initializer=normc_initializer(0.01),)(layer_1)

        self.base_model = tf.keras.Model(self.inputs, [actor_out_3, critic_out])

        # self.network = tf.keras.Sequential(
        #     [
        #         tf.keras.Input(shape=obs_space.shape),
        #         layers.Conv2D(64,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False),
        #         layers.GroupNormalization(64),
        #         layers.ReLU(),
        #         layers.MaxPool2D(pool_size=(2,2), strides=2),
        #         layers.Conv2D(32,kernel_size= (7,7), strides=(2,2), padding='same', use_bias=False),
        #         layers.GroupNormalization(32),
        #         layers.ReLU(),
        #         layers.MaxPool2D(pool_size=(2,2), strides=2),
        #         layers.Flatten(),
        #         layers.Dense(self._feature_dim)
        #     ]
        # )
        # self._actor_head = tf.keras.Sequential(
        #     [
        #         layers.Dense(256),
        #         layers.ReLU(),
        #         layers.Dense(self._num_actions)
        #     ]
        # )

        # self._critic_head = tf.keras.Sequential([
        #         layers.Dense(1),
        # ]
        # )
            # nn.Conv2d(self._num_objects, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # nn.GroupNorm(4, 64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            # nn.GroupNorm(2, 32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Flatten(),
            # nn.Linear(in_features=1568, out_features=self._feature_dim),
        # self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        # layer_1 = tf.keras.layers.Dense( 128,name="my_layer1",activation=tf.nn.relu,kernel_initializer=normc_initializer(1.0),)(self.inputs)
        # layer_out = tf.keras.layers.Dense(
        #     num_outputs,
        #     name="my_out",
        #     activation=None,
        #     kernel_initializer=normc_initializer(0.01),
        # )(layer_1)
        # value_out = tf.keras.layers.Dense(
        #     1,
        #     name="value_out",
        #     activation=None,
        #     kernel_initializer=normc_initializer(0.01),
        # )(layer_1)
        # self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def forward(self, input_dict, state, seq_lens):
        # obs_transformed = input_dict['obs'].permute(0, 3, 1, 2) # 32 x 112 x 112 x 7 [B, size, size, channels]
        # raise ValueError(input_dict["obs"].shape)
        actor_out, self._value_out = self.base_model(input_dict["obs"])
        return actor_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])