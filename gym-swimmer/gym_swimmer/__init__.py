from gym.envs.registration import register

register(
    id='Leon-swimmer-v1',
    entry_point='gym_swimmer.envs:SwimmerEnv'
)
