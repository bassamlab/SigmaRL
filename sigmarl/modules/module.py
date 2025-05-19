from abc import abstractmethod

import torch


class Module:
    @abstractmethod
    def get_observation_key(self) -> tuple:
        pass
