from gym_air_hockey.processor import DataProcessor
from gym_air_hockey.model import conv_model, convlstm_model

from gym.envs.registration import register

register(
    id='AirHockey-v0',
    entry_point='gym_air_hockey.envs:AirHockeyEnv'
)