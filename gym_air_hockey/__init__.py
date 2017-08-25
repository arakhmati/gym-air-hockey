from gym.envs.registration import register

register(
    id='AirHockey-v0',
    entry_point='gym_air_hockey.envs:AirHockeyEnv'
)

from gym_air_hockey.processor import DataProcessor