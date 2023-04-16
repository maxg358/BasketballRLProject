import math
import warnings

import gym
from gym import spaces
import sys
sys.path.append('..')
from ..utils import *
from ..vis import *

class NBAGymEnv(gym.Env):
    def __init__(self):
        super(NBAGymEnv, self).__init__()

        # Define action and observation spaces
        self.action_space = None #TODO 
        self.state = None #TODO

        # Initialize the play
        
    def reset(self):
        pass

    def reward(self):
        pass

    def step(self, action):
        pass
       
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
    