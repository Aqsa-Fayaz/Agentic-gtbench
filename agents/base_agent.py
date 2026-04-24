"""Abstract base class for all agent types."""

from abc import ABC, abstractmethod
from config.settings import settings


class BaseAgent(ABC):
    def __init__(self, agent_id: str, model: str = None, temperature: float = None):
        self.agent_id = agent_id
        self.model = model or settings.default_model
        self.temperature = temperature if temperature is not None else settings.default_temperature

    @abstractmethod
    def decide(self, game_state: dict, game_name: str, legal_moves: list) -> dict:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.agent_id}, model={self.model})"
