from PIL import Image
import numpy as np

class DataProcessor(object):
    def __init__(self, dim=128, mode='rgb'):
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

        self.mode = mode
        self.state_buffer = np.zeros((3 * 3, self.dim, self.dim), dtype=np.float32)

    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):

        def resize_observation(observation):
            observation = Image.fromarray(observation)
            observation = np.array(observation.resize((self.dim, self.dim)))
            observation = observation.transpose((2, 0 ,1))
            return observation

        def normalize_observation(observation):
            observation = observation - 128.0 # normalize
            observation = observation / 128.0 # scale
            return np.copy(observation)

        observation = resize_observation(observation)
        observation = observation.astype(np.float32)

        self.state_buffer[2*3:(2+1)*3] = np.copy(self.state_buffer[1*3:(1+1)*3])
        self.state_buffer[1*3:(1+1)*3] = np.copy(self.state_buffer[0*3:(0+1)*3])
        self.state_buffer[0*3:(0+1)*3] = np.copy(observation)

        if self.mode == 'rgb':
            return normalize_observation( self.state_buffer)
        elif self.mode == 'gray-diff':
            transposed = np.uint8(self.state_buffer.transpose((1, 2, 0)))
            # Take the double difference
            diff = transposed[:,:,6:9] - transposed[:, :, 3:6] - transposed[:, :, 0:3]
            # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
            def rgb2gray(rgb):
                r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                return gray
            gray = rgb2gray(diff)\
                   .reshape(1, 128, 128)
            return normalize_observation(gray)
        else:
            raise ValueError('Processing mode is not supported')

    def process_reward(self, reward):
        return np.clip(reward, -1.0, 1.0)

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
