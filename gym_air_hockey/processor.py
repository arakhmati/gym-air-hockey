import cv2
import numpy as np

class DataProcessor(object):
    def __init__(self, dim=128):
        self.dim = dim
        self.actions = np.array([
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
        self.metrics = []
        self.metrics_names = []

    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info
    
    def _resize_observation(self, observation):
        return cv2.resize(observation, (self.dim, self.dim))

    @staticmethod
    def _normalize_observation(observation):
        observation = observation - 0 # normalize
        observation = observation / 256 # scale
        return observation

    def process_observation(self, observation):
        observation = self._resize_observation(observation)
        observation = observation.astype(np.float32)
        observation = self._normalize_observation(observation)
        return observation

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_action(self, action):
        if isinstance(action, np.ndarray):
            return action
        return self.actions[action]

    def process_state_batch(self, batch):
        _, height, width, depth = batch[0].shape
        batch = np.array(batch).reshape(len(batch), height, width, depth)
        return batch
    
    # Used to generate labels for the neural network
    def action_to_label(self, action):
        return action[0]*3 + action[1] + 4
