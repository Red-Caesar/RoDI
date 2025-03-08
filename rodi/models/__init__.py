from dataclasses import dataclass
from typing import Optional
from rodi.models.td3 import TD3
from rodi.models.portfolio_env import StockEnv
from rodi.models.replay_buffer import ReplayBuffer
from rodi.models.base import BaseModel, BaseBuffer
import gymnasium as gym


@dataclass
class ModelProfile:
    model: BaseModel
    prefix_model_path: str
    env: gym.Env
    replay_buffer: Optional[BaseBuffer]


PROFILES = {
    "TD3": ModelProfile(
        model=TD3,
        prefix_model_path="td3",
        env=StockEnv,
        replay_buffer=ReplayBuffer,
    ),
}
