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
        game_info = self.game.step(action)
        self.g = game_info
        
        state = game_info.frame
        terminal = game_info.scored is not None
        
        reward = 0.0
        if game_info.puck_was_hit:
            reward += 1000.0
        else:
            reward -= 10000.0
            
        if game_info.scored == 'top':
            reward -= 1000000.0
        elif game_info.scored == 'bottom':
            reward += 10000000.0
        
        return state, reward, terminal, {}

    def _reset(self):
        self.game.reset()
        return self.game.cropped_frame














