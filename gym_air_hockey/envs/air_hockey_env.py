import gym
import gym.spaces
import numpy as np

from air_hockey import AirHockey
from gym_air_hockey import DataProcessor

class AirHockeyEnv(gym.Env):
    def __init__(self):

        self.game = AirHockey()
        self.processor = DataProcessor()

        self.height = self.width = self.processor.dim

        self.action_space = gym.spaces.Discrete(len(self.processor.actions))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(9, self.height, self.width))
        self.reward_range = (-1.0, 1.0)

        self.n_actions = self.action_space.n

        self.viewer = None

    def update(self, video_file=None, mode=None):
        if video_file is not None:
            self.game = AirHockey(video_file=video_file)
        if mode is not None:
            self.processor = DataProcessor(mode=mode)

    def step(self, robot_action=None, human_action=None, debug=False):

        robot_action = self.processor.process_action(robot_action)
        human_action = self.processor.process_action(human_action)

        game_info = self.game.step(robot_action=robot_action, human_action=human_action)
        terminal = game_info.scored is not None
        
        robot_action = self.processor.action_to_label(game_info.robot_action)
        human_action = self.processor.action_to_label(game_info.human_action)

        # Reward
        reward = 0.0
        if game_info.scored == 'top':
            if debug: print('GAME INFO: Goal ☹')
            reward = -1.0
        elif game_info.scored == 'bottom':
            if debug: print('GAME INFO: Goal ☺')
            reward = 1.0
        elif game_info.puck_was_hit:
            if debug: print('GAME INFO: puck was hit ☺')
            reward = 0.5
        elif game_info.hit_the_border:
            if debug: print('GAME INFO: hit the border ☹')
            reward = -0.5
        elif game_info.puck_is_at_the_bottom:
            if game_info.distance_decreased:
                if debug: print('GAME INFO: decreased distance ☺')
                reward = 0.5
            else:
                if debug: print('GAME INFO: increased distance ☹')
                reward = -0.5
        else:
            if robot_action == 4:
                if debug: print('GAME INFO: standing still ☺')
                reward = 0.5
            else:
                if debug: print('GAME INFO: not standing still ☹')
                reward = -0.5
    
        reward = np.clip(reward, -1.0, 1.0)
        state = self.processor.process_observation(game_info.frame)

        return state, reward, terminal, {'robot_action': robot_action, 'human_action': human_action}

    def reset(self):
        # Reset the environment
        self.game.reset()
        # Fill in the current_state
        stand_action = self.processor.process_action(4)
        for _ in range(3):
            game_info = self.game.step(robot_action=stand_action, human_action=stand_action)
            state = self.processor.process_observation(game_info.frame)
        return state














