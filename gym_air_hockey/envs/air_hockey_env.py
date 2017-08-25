import gym
import gym.spaces

from air_hockey import AirHockey
from gym_air_hockey import DataProcessor

class AirHockeyEnv(gym.Env):
    def __init__(self, video_file='reinforcement.avi'):
        
        self.game = AirHockey(video_file=video_file)
        self.processor = DataProcessor()
        
        self.action_space = gym.spaces.Discrete(len(self.processor.actions))
        self.nb_actions = self.action_space.n
        
        self.screen_width = self.screen_height = self.processor.dim
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3))
        self.reward_range = (-10000.0, 1000.0)
        
        self.viewer = None


    def _step(self, action):
        state, reward = self.game.step(action)
        terminal = (reward == self.reward_range[0] or reward == self.reward_range[1])
        return state, reward, terminal, {}

    def _reset(self):
        self.game.reset()
        return self.game.state














