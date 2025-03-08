from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def train(self, replay_buffer, batch_size):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass


class BaseBuffer(ABC):
    @abstractmethod
    def add(self, state, action, next_state, reward, done):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass
