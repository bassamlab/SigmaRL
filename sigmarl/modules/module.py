from abc import abstractmethod

import torch

class Module:

    def __init__(self, env, mappo = True):
        # TODO find common ground of modules to init some default values here
        self.env = env
        self.policy = None
        self.critic = None
        self.GAE = None
        self.loss_module = None
        self.optim = None

    @abstractmethod
    def get_observation_key(self) -> tuple:
        pass




