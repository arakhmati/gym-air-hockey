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
        
        self.reset()
        
    def reset(self):
        
        episodes = ['normal'] * 16 + ['missing'] * 3 + ['random']
        self.episode = episodes[np.random.randint(len(episodes))]
        print('Episode type: {}'.format(self.episode))
        
        self.use_object = {'arm': True,
                           'puck': True,
                           'top_mallet': True}
        
        if self.episode == 'normal':
            pass
        elif self.episode == 'missing':
            random_num = np.random.randint(6)
            if random_num == 0:
                self.use_object['arm'] = False
                self.use_object['puck'] = False
                self.use_object['top_mallet'] = False
            elif random_num == 1:
                self.use_object['arm'] = False
                self.use_object['puck'] = False
                self.use_object['top_mallet'] = True
            elif random_num == 2:
                self.use_object['arm'] = False
                self.use_object['puck'] = True
                self.use_object['top_mallet'] = False
            elif random_num == 3:
                self.use_object['arm'] = False
                self.use_object['puck'] = True
                self.use_object['top_mallet'] = True
            elif random_num == 4:
                self.use_object['arm'] = True
                self.use_object['puck'] = False
                self.use_object['top_mallet'] = False
            elif random_num == 5:
                self.use_object['arm'] = True
                self.use_object['puck'] = False
                self.use_object['top_mallet'] = True
        
        if self.episode != 'random':
            # Reset the environment
            self.game.reset(use_object=self.use_object)
            # Fill in the current_state
            stand_action = self.processor.process_action(4)
            for _ in range(3):
                game_info = self.game.step(robot_action=stand_action, human_action=stand_action)
                state = self.processor.process_observation(game_info.frame)    
        else:
            state = np.random.randint(0, 255, (9, 128, 128), dtype=np.uint8)
            
        return state

    def update(self, video_file=None, mode=None):
        if video_file is not None:
            self.game = AirHockey(video_file=video_file)
        if mode is not None:
            self.processor = DataProcessor(mode=mode)

    def step(self, robot_action=None, human_action=None, debug=False):

        robot_action = self.processor.process_action(robot_action)
        human_action = self.processor.process_action(human_action)

        game_info = self.game.step(robot_action=robot_action, human_action=human_action, use_object=self.use_object)
        terminal = game_info.scored is not None
        
        robot_action = self.processor.action_to_label(game_info.robot_action)
        human_action = self.processor.action_to_label(game_info.human_action)

        # Reward
        reward = 0.0
        if self.episode == 'normal':
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
                if game_info.in_the_target:
                    if robot_action == 4 or robot_action == 9:
                        if debug: print('GAME INFO: standing still in the target ☺')
                        reward = 0.5
                    else:
                        if debug: print('GAME INFO: not standing still in the target ☹')
                        reward = -0.5
                else:
                    if debug: print('GAME INFO: robot is not in the target ☹')
                    reward = -0.5
                    
        elif self.episode == 'missing':
            if robot_action == 4:
                if debug: print('GAME INFO: standing while missing ☺')
                reward =  0.5
            else:
                if debug: print('GAME INFO: moving while missing ☹')
                reward = -0.5
                
        elif self.episode == 'random':
            if robot_action == 9:
                if debug: print('GAME INFO: standing while random ☺')
                reward =  0.5
            else:
                if debug: print('GAME INFO: moving while random ☹')
                reward = -0.5
    
        reward = np.clip(reward, -1.0, 1.0)
        
        if self.episode == 'random':
            terminal = False
            state = np.random.ranf((9, 128, 128))
            state -= np.mean(state)
            state /= np.std(state)
        else:
            state = self.processor.process_observation(game_info.frame)

        return state, reward, terminal, {'robot_action': robot_action, 'human_action': human_action}















