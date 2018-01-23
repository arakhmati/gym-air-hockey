from PIL import Image
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
                            [ 1,  1],
                            [ 0,  0]],
                            dtype=np.int8)
        self.metrics = []
        self.metrics_names = []

        self.frame = np.zeros((3 * 3, self.dim, self.dim), dtype=np.float32)


    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def _resize_observation(self, observation):
        observation = Image.fromarray(observation)
        observation = np.array(observation.resize((self.dim, self.dim)))
        observation = observation.transpose((2, 0 ,1))
        return observation

    @staticmethod
    def _normalize_observation(observation):
        observation = observation - 128 # normalize
        observation = observation / 128 # scale
        return observation

    def process_observation(self, observation):
        observation = self._resize_observation(observation)
        observation = observation.astype(np.float32)
        observation = self._normalize_observation(observation)
        self.frame[2*3:(2+1)*3] = np.copy(self.frame[1*3:(1+1)*3])
        self.frame[1*3:(1+1)*3] = np.copy(self.frame[0*3:(0+1)*3])
        self.frame[0*3:(0+1)*3] = np.copy(observation)
        return np.copy(self.frame)

    def process_reward(self, reward):
        return reward

    def process_info(self, info):
        return info

    def process_action(self, label):
        if label is None: return None
        return self.actions[label]

    def process_state_batch(self, batch):
        _, depth, height, width = batch[0].shape
        batch = np.array(batch).reshape(len(batch), depth, height, width)
        return batch

    # Used to generate labels for the neural network
    def action_to_label(self, action):
        return int(action[0]*3 + action[1] + 4)
