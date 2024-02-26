import math
from collections import deque, namedtuple
from typing import Any, Iterator, SupportsFloat
from itertools import cycle

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import TimeLimit
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchmetrics import MeanSquaredError
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from utils.config import FrozenLakeConfig


ACTIONS = {"LEFT": 0,
           "DOWN": 1,
           "RIGHT": 2,
           "UP": 3}

# MAPS = {
#     '2x2': ['SF', 'HG'],
#     '4x4': ['SFFF', 'FFFF', 'HHFF', 'HHFG'],
#     '6x6': ['SFFFFF', 'FFFFFF', 'FFFFFF', 'HHHFFF', 'HHHFFF', 'HHHFFG'],
#     '8x8': ['SFFFFFFF', 'FFFFFFFF', 'HHHHHHFF', 'FFFFFFFF', 'FFFFFFFF', 'FFHHHHHH', 'FFFFFFFF', 'FFFFFFFG'],
#     '16x16': ['SFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFF', 'HHHHHHHHHHHHFFFF', 'HHHHHHHHHHHHFFFF', 'FFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFF', 'FFFFHHHHHHHHHHHH', 'FFFFHHHHHHHHHHHH', 'FFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFGF', 'FFFFFFFFFFFFFFFF'],
#     '24x24': ['SFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'HHHHHHHHHHHHHHHHHHFFFFFF', 'HHHHHHHHHHHHHHHHHHFFFFFF', 'HHHHHHHHHHHHHHHHHHFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFHHHHHHHHHHHHHHHHHH', 'FFFFFFHHHHHHHHHHHHHHHHHH', 'FFFFFFHHHHHHHHHHHHHHHHHH', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFGFF', 'FFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFF'],
#     '32x32': ['SFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'HHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFF', 'HHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFF', 'HHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFF', 'HHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHH', 'FFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHH', 'FFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHH', 'FFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHH', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFGFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF', 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'],
#     '48x48': ['SFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFGFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'],
#     '64x64': ['SFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF','HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFGFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF','FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF']
# }

MAPS = {
    '8x8':['SFHHHHFG',
           'FFHHHHFF',
           'FFHHHHFF',
           'FFHHHHFF',
           'FFHHHHFF',
           'FFHHHHFF',
           'FFFFFFFF',
           'FFFFFFFF'],
    '24x24': ['SFFFFFHHHHHHHHHHHHFFFFFG',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFHHHHHHHHHHHHFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFF'],
    '32x32': ['SFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFG',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFHHHHHHHHHHHHHHHHFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'],
    '48x48': ['SFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFG',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'],
    '64x64': ['SFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFG',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF',
              'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF']
}


def bfs(map_input):
    n_rows = len(map_input)
    n_cols = len(map_input[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, down, left, right

    distance = [[-1 for _ in range(n_cols)] for _ in range(n_rows)]
    queue = [(0, 0, 0)]

    while queue:
        row, col, dist = queue.pop(0)

        if distance[row][col] != -1:
            continue

        # Ensure row and column indices are within bounds before accessing
        if 0 <= row < n_rows and 0 <= col < n_cols:
            distance[row][col] = dist

            for d_row, d_col in directions:
                new_row, new_col = row + d_row, col + d_col
                if (
                    0 <= new_row < n_rows
                    and 0 <= new_col < n_cols
                    and distance[new_row][new_col] == -1
                    # Assuming "#" represents obstacles
                    and map_input[new_row][new_col] != "H"
                ):
                    queue.append((new_row, new_col, dist+1))
    return distance


REWARDS = {key: np.array(bfs(value)) for key, value in MAPS.items()}
REWARDS = {key: value/np.max(value) for key, value in REWARDS.items()}


class FullObservation(gym.ObservationWrapper):

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=1, shape=(3, self.nrow, self.ncol))

    def observation(self, observation: any) -> any:
        row, col = self._get_player_coordinates(observation)
        player = np.zeros(self.desc.shape)
        player[row, col] = 1
        frozen = np.array(self.desc == b"F")
        start = np.array(self.desc == b"S")
        goal = np.array(self.desc == b"G")
        frozen = np.any((frozen, start, goal), axis=0)
        observation = np.stack((player, goal, frozen))
        return observation

    def _get_player_coordinates(self, observation: any) -> tuple[int, int]:
        row, col = divmod(observation, self.nrow)
        return row, col


def get_observation_from_player_position(desc, row, col):
    player = np.zeros(desc.shape)
    player[row, col] = 1
    frozen = np.array(desc == b"F")
    start = np.array(desc == b"S")
    goal = np.array(desc == b"G")
    frozen = np.any((frozen, start, goal), axis=0)
    observation = np.stack((player, goal, frozen))
    return observation


class RewardShaping(gym.Wrapper):
    def __init__(self, env: Env, rewards):
        super().__init__(env)
        self.rewards = rewards

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if (terminated or truncated) and reward == 0:
            reward = -100
        else:
            reward = np.max(np.multiply(obs[0], self.rewards)) * 10
            reward = math.floor(reward)
            reward = reward -10
            if reward == 0:
                reward = 1000
        return obs, reward, terminated, truncated, info


def print_policy(desc, net):
    print('----------------------------------------')
    with torch.no_grad():
        for row in range(desc.shape[0]):
            for col in range(desc.shape[1]):
                if desc[row][col] == b"H":
                    print("H", end="")
                else:
                    observation = get_observation_from_player_position(
                        desc, row, col)
                    state = torch.tensor(np.expand_dims(observation, axis=0)).cuda('cuda')
                    q_values = net(state)
                    _, action = torch.max(q_values, dim=1)
                    action = int(action.item())
                    match action:
                        case 0:
                            print("<", end="")
                        case 1:
                            print("v", end="")
                        case 2:
                            print(">", end="")
                        case 3:
                            print("^", end="")
            print()
    print('----------------------------------------')

# Adapted from https://github.com/revidee/pytorch-pyramid-pooling
class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="max"):
        """
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        """
        Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [
            int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_pad1 = int(math.floor(
                (w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(
                math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor(
                (h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(
                math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
                h_pad1 + h_pad2 == (h_kernel *
                                    levels[i] - previous_conv_size[0])

            padded_input = nn.functional.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                             mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(
                    h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(
                    h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError(
                    "Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                spp = x.reshape(num_sample, -1)
            else:
                spp = torch.cat((spp, x.reshape(num_sample, -1)), 1)

        return spp


class SpatialPyramidPooling(PyramidPooling):
    def __init__(self, levels, mode="max"):
        """
                Spatial Pyramid Pooling Module, which divides the input Tensor horizontally and horizontally
                (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
                Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
                In other words: It divides the Input Tensor in level*level rectangles width of roughly (previous_conv.size(3) / level)
                and height of roughly (previous_conv.size(2) / level) and pools its value. (pads input to fit)
                :param levels defines the different divisions to be made in the width dimension
                :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
                :returns (forward) a tensor vector with shape [batch x 1 x n],
                                                    where n: sum(filter_amount*level*level) for each level in levels
                                                    which is the concentration of multi-level pooling
                """
        super(SpatialPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """
                Calculates the output shape given a filter_amount: sum(filter_amount*level*level) for each level in levels
                Can be used to x.view(-1, spp.get_output_size(filter_amount)) for the fully-connected layers
                :param filters: the amount of filter of output fed into the spatial pyramid pooling
                :return: sum(filter_amount*level*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out


class DQN(nn.Module):

    def __init__(self, n_channels: int, n_actions: int):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(DQN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 4, kernel_size=3, padding=0),
            nn.ReLU(),
            SpatialPyramidPooling([8]),
        )
        self.decoder = nn.Linear(256, n_actions)
        # self.net = nn.Sequential(
        #     nn.Conv2d(n_channels, 4, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     SpatialPyramidPooling([8]),
        #     # nn.Flatten(),
        #     nn.Linear(256, n_actions),
        # )

    def forward(self, x):
        encoder_output = self.encoder(x.float())
        self.last_encoder = encoder_output
        return self.decoder(encoder_output)


# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "next_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, next_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[tuple]:
        states, actions, rewards, dones, next_states = self.buffer.sample(
            self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], next_states[i]


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state, _ = self.env.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state, _ = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(np.expand_dims(self.state, axis=0))

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
            # print(f'Q-Values: {q_values}')
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
    ) -> tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """
        action = self.get_action(net, epsilon, device)

        # do step in the environment
        next_state, reward, terminated, truncated, _ = self.env.step(action)

        if np.array_equal(next_state, self.state):
            reward = -20
        done = terminated or truncated

        exp = Experience(self.state, action, reward, done, next_state)

        self.replay_buffer.append(exp)

        self.state = next_state
        if done:
            self.reset()
        return reward, done

    def play_route(self, net, real_action, device, save_exp=False):
        # action = self.get_action(net, 0, device)
        next_state, reward, terminated, truncated, _ = self.env.step(real_action)
        if np.array_equal(next_state, self.state):
            reward = -20
        done = terminated or truncated
        if save_exp:
            exp = Experience(self.state, real_action, reward, done, next_state)
            self.replay_buffer.append(exp)

        self.state = next_state
        if done:
            self.reset()
        return reward, done


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self,
        batch_size: int = 256,
        lr: float = 1e-2,
        gamma: float = 0.95,
        sync_rate: int = 5,
        replay_size: int = 4096,
        warm_start_steps: int = 4096,
        epsilon_start: float = 1.0,
        epsilon_decay: float = 0.9997,
        test_episodes: int = 100
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_steps: max episode reward in the environment
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
        """
        super().__init__()
        self.save_hyperparameters()

        self.env = RewardShaping(TimeLimit(FullObservation(
            gym.make('FrozenLake-v1', desc=MAPS['24x24'], is_slippery=False, max_episode_steps=150, render_mode="ansi")),max_episode_steps=150), REWARDS['24x24'])
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.mse = MeanSquaredError()
        self.episode = 0.0
        self.episodic_return = 0.0
        self.discounted_episodic_return = 0.0
        self.episodic_length = 0.0
        self.populate(self.hparams.warm_start_steps)
        self.epsilon = self.hparams.epsilon_start
        self.route = [ACTIONS['DOWN']] * 18
        self.route.extend([ACTIONS['RIGHT']] * 23)
        self.route.extend([ACTIONS['UP']] * 18)
        self.route = iter(cycle(self.route))

    def populate(self, steps) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)
        self.agent.reset()

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(
            1, actions.long().unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return self.mse(state_action_values, expected_state_action_values)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)

        if self.global_step % 2000 == 0:
            print_policy(self.env.desc, self.net)

        # if self.global_step == 100000:
        #     self.env = TimeLimit(RewardShaping(FullObservation(
        #         gym.make('FrozenLake-v1', desc=MAPS['32x32'], is_slippery=False)), REWARDS['32x32']), 100)
        #     self.agent.env = self.env
        #     self.agent.reset()
        #     self.populate(self.hparams.warm_start_steps)
        # elif self.global_step == 200000:
        #     self.env = TimeLimit(RewardShaping(FullObservation(
        #         gym.make('FrozenLake-v1', desc=MAPS['48x48'], is_slippery=False)), REWARDS['48x48']), 100)
        #     self.agent.env = self.env
        #     self.agent.reset()
        #     self.populate(self.hparams.warm_start_steps)

        # step through environment with agent
        # self.env.render()
        
        # make the agent go through the best route a few times before following policy
        # if self.global_step <= 20000:
        #     reward, done = self.agent.play_route(self.net, next(self.route), device, save_exp=True)
        # else:
        reward, done = self.agent.play_step(self.net, self.epsilon, device)
        self.episodic_return += reward
        self.discounted_episodic_return += (self.hparams.gamma **
                                            self.episodic_length) * reward
        self.episodic_length += 1.0

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if done:
            self.log_dict(
                {
                    "train/episodic_return": self.episodic_return,
                    "train/discounted_episodic_return": self.discounted_episodic_return,
                    "train/episodic_length": self.episodic_length,
                }
            )
            self.episodic_return = 0
            self.discounted_episodic_return = 0
            self.episode += 1
            self.episodic_length = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "train/reward": reward,
                "train/loss": loss,
                "train/episode": self.episode,
                "train/epsilon": self.epsilon,
            }
        )
        self.epsilon *= self.hparams.epsilon_decay

        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx) -> Tensor:
        total_reward = 0
        total_discounted_reward = 0
        episodic_length = 0
        current_episode = 0
        device = self.get_device(batch)

        print_policy(self.env.desc, self.net)

        self.env = RewardShaping(TimeLimit(FullObservation(
                gym.make('FrozenLake-v1', desc=MAPS['64x64'], is_slippery=False, max_episode_steps=300)), max_episode_steps=300), REWARDS['64x64'])
        self.agent.env = self.env
        self.agent.reset()

        print_policy(self.env.desc, self.net)
        goal_reached = 0
        while current_episode < self.hparams.test_episodes:
            reward, done = self.agent.play_step(self.net, 0, device)
            if episodic_length >= 200:
                done = True
            total_reward += reward
            total_discounted_reward += (
                self.hparams.gamma ** self.episodic_length) * reward
            episodic_length += 1.0

            if done:
                current_episode += 1
                episodic_length = 0
                if reward == 100:
                    goal_reached += 1

        metrics = {
            "test/goal_reached": goal_reached,
            "test/total_reward": total_reward,
            "test/total_discounted_reward": total_discounted_reward,
            "test/average_reward": (total_reward/self.hparams.test_episodes),
            "test/average_discounted_reward": (total_discounted_reward/self.hparams.test_episodes),
        }
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self) -> list[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return optimizer

    def _dataloader(self):
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.batch_size)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        return self._dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


@hydra.main(version_base=None, config_path='../conf', config_name='config.yaml')
def main(cfg: FrozenLakeConfig):
    pl_logger = WandbLogger(project=cfg.wandb.project, entity=cfg.wandb.entity)
    pl_logger.experiment.config.update(OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True))

    model = DQNLightning(
        batch_size=cfg.dqn.batch_size,
        lr=cfg.dqn.learning_rate,
        gamma=cfg.dqn.gamma,
        sync_rate=cfg.dqn.sync_rate,
        replay_size=cfg.dqn.replay_buffer_size,
        warm_start_steps=cfg.dqn.replay_buffer_size,
        epsilon_start=cfg.dqn.epsilon,
        epsilon_decay=cfg.dqn.epsilon_decay,
    )

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=-1,
        max_steps=cfg.dqn.max_steps,
        log_every_n_steps=1,
        logger=pl_logger,
        enable_progress_bar=False
    )

    trainer.fit(model)

    trainer.test(model)

    # encoders = []

    # # print("8x8")
    # encoder_route = []
    # env = TimeLimit(FullObservation(gym.make('FrozenLake-v1', desc=MAPS['8x8'], is_slippery=False)), 100)
    # model.agent.env = env
    # model.agent.reset()
    # route = ([ACTIONS['RIGHT']] * 6) + ([ACTIONS['DOWN']] * 3) + ([ACTIONS['LEFT']] * 5) + ([ACTIONS['DOWN']] * 4) + ([ACTIONS['RIGHT']] * 6)
    # for action in route:
    #     model.agent.play_route(model.net, action, "cpu")
    #     # print(model.agent.state[0])
    #     encoder_route.append(model.net.last_encoder.detach().numpy())
    # encoder_route = np.concatenate(encoder_route, axis=0)
    # encoders.append(encoder_route.flatten())
    # print(encoders[0].shape)

    # # print("16x16")
    # encoder_route = []
    # env = TimeLimit(FullObservation(gym.make('FrozenLake-v1', desc=MAPS['16x16'], is_slippery=False)), 100)
    # model.agent.env = env
    # model.agent.reset()
    # route = ([ACTIONS['RIGHT']] * 12) + ([ACTIONS['DOWN']] * 6) + ([ACTIONS['LEFT']] * 10) + ([ACTIONS['DOWN']] * 8) + ([ACTIONS['RIGHT']] * 12)
    # for action in route:
    #     model.agent.play_route(model.net, action, "cpu")
    #     # print(model.agent.state[0])
    #     encoder_route.append(model.net.last_encoder.detach().numpy())
    # encoder_route = np.concatenate(encoder_route, axis=0)
    # new_encoder_route = []
    # for index, _ in enumerate(encoder_route):
    #     if index % 2 != 0:
    #         continue
    #     new_encoder_route.append([np.mean([encoder_route[index], encoder_route[index+1]], axis=0)])
    # encoder_route=np.concatenate(new_encoder_route)
    # encoders.append(encoder_route.flatten())

    # # print("24x24")
    # encoder_route = []
    # env = TimeLimit(FullObservation(gym.make('FrozenLake-v1', desc=MAPS['24x24'], is_slippery=False)), 100)
    # model.agent.env = env
    # model.agent.reset()
    # route = ([ACTIONS['RIGHT']] * 18) + ([ACTIONS['DOWN']] * 9) + ([ACTIONS['LEFT']] * 15) + ([ACTIONS['DOWN']] * 12) + ([ACTIONS['RIGHT']] * 18)
    # for action in route:
    #     model.agent.play_route(model.net, action, "cpu")
    #     # print(model.agent.state[0])
    #     encoder_route.append(model.net.last_encoder.detach().numpy())
    # encoder_route = np.concatenate(encoder_route, axis=0)
    # new_encoder_route = []
    # for index, _ in enumerate(encoder_route):
    #     if index % 3 != 0:
    #         continue
    #     new_encoder_route.append([np.mean([encoder_route[index], encoder_route[index+1], encoder_route[index+2]], axis=0)])
    # encoder_route=np.concatenate(new_encoder_route)
    # encoders.append(encoder_route.flatten())

    # # print("32x32")
    # encoder_route = []
    # env = TimeLimit(FullObservation(gym.make('FrozenLake-v1', desc=MAPS['32x32'], is_slippery=False)), 100)
    # model.agent.env = env
    # model.agent.reset()
    # route = ([ACTIONS['RIGHT']] * 24) + ([ACTIONS['DOWN']] * 12) + ([ACTIONS['LEFT']] * 20) + ([ACTIONS['DOWN']] * 16) + ([ACTIONS['RIGHT']] * 24)
    # for action in route:
    #     model.agent.play_route(model.net, action, "cpu")
    #     # print(model.agent.state[0])
    #     encoder_route.append(model.net.last_encoder.detach().numpy())
    # encoder_route = np.concatenate(encoder_route, axis=0)
    # new_encoder_route = []
    # for index, _ in enumerate(encoder_route):
    #     if index % 4 != 0:
    #         continue
    #     new_encoder_route.append([np.mean([encoder_route[index], encoder_route[index+1], encoder_route[index+2], encoder_route[index+3]], axis=0)])
    # encoder_route=np.concatenate(new_encoder_route)
    # encoders.append(encoder_route.flatten())

    # encoder_route = []
    # env = TimeLimit(FullObservation(gym.make('FrozenLake-v1', desc=MAPS['48x48'], is_slippery=False)), 100)
    # model.agent.env = env
    # model.agent.reset()
    # route = ([ACTIONS['RIGHT']] * 36) + ([ACTIONS['DOWN']] * 18) + ([ACTIONS['LEFT']] * 30) + ([ACTIONS['DOWN']] * 24) + ([ACTIONS['RIGHT']] * 36)
    # for action in route:
    #     model.agent.play_route(model.net, action, "cpu")
    #     # print(model.agent.state[0])
    #     encoder_route.append(model.net.last_encoder.detach().numpy())
    # encoder_route = np.concatenate(encoder_route, axis=0)
    # new_encoder_route = []
    # for index, _ in enumerate(encoder_route):
    #     if index % 6 != 0:
    #         continue
    #     new_encoder_route.append([np.mean([encoder_route[index], encoder_route[index+1], encoder_route[index+2], encoder_route[index+3], encoder_route[index+4], encoder_route[index+5]], axis=0)])
    # encoder_route=np.concatenate(new_encoder_route)
    # encoders.append(encoder_route.flatten())

    # encoder_route = []
    # env = TimeLimit(FullObservation(gym.make('FrozenLake-v1', desc=MAPS['64x64'], is_slippery=False)), 100)
    # model.agent.env = env
    # model.agent.reset()
    # route = ([ACTIONS['RIGHT']] * 48) + ([ACTIONS['DOWN']] * 24) + ([ACTIONS['LEFT']] * 40) + ([ACTIONS['DOWN']] * 32) + ([ACTIONS['RIGHT']] * 48)
    # for action in route:
    #     model.agent.play_route(model.net, action, "cpu")
    #     # print(model.agent.state[0])
    #     encoder_route.append(model.net.last_encoder.detach().numpy())
    # encoder_route = np.concatenate(encoder_route, axis=0)
    # new_encoder_route = []
    # for index, _ in enumerate(encoder_route):
    #     if index % 8 != 0:
    #         continue
    #     new_encoder_route.append([np.mean([encoder_route[index], encoder_route[index+1], encoder_route[index+2], encoder_route[index+3], encoder_route[index+4], encoder_route[index+5], encoder_route[index+6], encoder_route[index+7]], axis=0)])
    # encoder_route=np.concatenate(new_encoder_route)
    # encoders.append(encoder_route.flatten())

    # dists = np.zeros((6,6))
    # for i in range(6):
    #     for j in range(6):
    #         dists[i][j] = np.sqrt(np.sum(encoders[i]-encoders[j])**2)
    # print(dists)

    # for i in range(6):
    #     for j in range(6):
    #         dists[i][j] = np.dot(encoders[i],encoders[j])
    # print(dists)


if __name__ == '__main__':
    cs = ConfigStore.instance()
    cs.store(name='frozen_lake_config', node=FrozenLakeConfig)
    main()
