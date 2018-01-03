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

        self.height = self.width = self.processor.dim
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(9, self.height, self.width))
        self.reward_range = (-10.0, 10.0)

        self.viewer = None

    def _step(self, action):
        game_info = self.game.step(action, dt=2)

        state = game_info.frame
        terminal = game_info.scored is not None

        reward = 0.0
        if game_info.puck_was_hit:
            reward = 1.0
        else:
            reward = -0.1

        if game_info.scored == 'top':
            reward -= 10.0
        elif game_info.scored == 'bottom':
            reward += 10.0

        return state, reward, terminal, {}

    def _reset(self):
        game_info = self.game.reset()
        return game_info.frame














