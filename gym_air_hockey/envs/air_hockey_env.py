import gym
from gym import spaces
import numpy as np

from air_hockey import AirHockey

class AirHockeyEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, display_screen=True):
        self.game = AirHockey()
        
        self._action_set = np.copy(self.game.actions)
        self.action_space = spaces.Discrete(9)
        
        self.screen_width, self.screen_height = 128, 128
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        
        self.viewer = None


    def _step(self, action_index):
        state, reward = self.game.step(self._action_set[action_index])
        terminal = (reward != 0.0)
        return state, reward, terminal, {}

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        self.game.reset()
        state = self.game.state
        return state

    def _render(self, mode='human', close=False):
        pass


    def _seed(self, seed):
        pass














