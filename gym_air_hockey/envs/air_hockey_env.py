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
        self.reward_range = (-1.0, 1.0)

        self.viewer = None

    def step(self, action, adversarial_action=None):
        game_info = self.game.step(action=action, adversarial_action=adversarial_action)

        state = game_info.frame
        terminal = game_info.scored is not None

        reward = 0.0
        if game_info.puck_was_hit:
            reward = 1.0
        else:
            if game_info.puck_is_at_the_bottom:
                if game_info.distance_decreased:
                    reward = 0.1
                else:
                    reward = -0.1

        if game_info.scored == 'top':
            reward = -1.0
        elif game_info.scored == 'bottom':
            reward = 1.0

        return state, reward, terminal, {}

    def _reset(self):
        game_info = self.game.reset()
        return game_info.frame














