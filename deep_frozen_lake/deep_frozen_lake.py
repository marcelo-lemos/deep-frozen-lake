from collections import deque, namedtuple
from typing import Iterator

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
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


class DQN(nn.Module):

    def __init__(self, n_channels: int, n_actions: int):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())


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
        next_state, reward, done, _, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, next_state)

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

        self.env = FullObservation(
            gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False))
        # self.env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
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

    def populate(self, steps) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

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

        # step through environment with agent
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

        while current_episode < self.hparams.test_episodes:
            reward, done = self.agent.play_step(self.net, 0, device)
            total_reward += reward
            total_discounted_reward += (
                self.hparams.gamma ** self.episodic_length) * reward
            episodic_length += 1.0

            if done:
                current_episode += 1
                episodic_length = 0

        self.log_dict(
            {
                "test/total_reward": total_reward,
                "test/total_discounted_reward": total_discounted_reward,
                "test/average_reward": (total_reward/self.hparams.test_episodes),
                "test/average_discounted_reward": (total_discounted_reward/self.hparams.test_episodes),
            }
        )

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
            num_workers=1,
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
        accelerator="auto",
        max_epochs=-1,
        max_steps=cfg.dqn.max_steps,
        log_every_n_steps=1,
        logger=pl_logger,
    )

    trainer.fit(model)

    trainer.test(model)


if __name__ == '__main__':
    cs = ConfigStore.instance()
    cs.store(name='frozen_lake_config', node=FrozenLakeConfig)
    main()
