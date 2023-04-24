from dataclasses import dataclass


@dataclass
class DQNConfig:
    learning_rate: float
    gamma: float
    epsilon: float
    epsilon_decay: float
    replay_buffer_size: int
    sync_rate: int
    max_steps: int
    batch_size: int


@dataclass
class WandbConfig:
    entity: str
    project: str


@dataclass
class FrozenLakeConfig:
    dqn: DQNConfig
    wandb: WandbConfig
