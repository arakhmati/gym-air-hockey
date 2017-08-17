import gym
import cv2
import numpy as np
from gym_air_hockey import AirHockeyProcessor

from air_hockey import AirHockey

class AirHockeyEnv(gym.Env):
    def __init__(self, video_file='reinforcement.avi'):
        self.game = AirHockey(video_file=video_file)
        
        self._actions = np.array([
                            [-1, -1],
                            [-1,  0],
                            [-1,  1],
                            [ 0, -1],
                            [ 0,  0],
                            [ 0,  1],
                            [ 1, -1],
                            [ 1,  0],
                            [ 1,  1]], 
                            dtype=np.int8)
        self.action_space = gym.spaces.Discrete(len(self._actions))
        self.nb_actions = self.action_space.n
        
        self.screen_width, self.screen_height = 128, 128
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        self.reward_range = (-10000.0, 1000.0)
        
        self.viewer = None


    def _step(self, action_index):
        state, reward = self.game.step(self._actions[action_index])
        state = (cv2.resize(state, (128, 128)).astype(np.float32) - 128)/128
        
        
        terminal = (reward == self.reward_range[0])
        
        
        return state, reward, terminal, {}

    def _reset(self):
        self.game.reset()
        return self.game.state














