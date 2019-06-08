""" Add this to the register file"""

# Own created environment
# ----------------------------------------
register(
    id='LeonSwimmer-v0',
    entry_point='gym.envs.swimmer:SwimmerEnv',
    max_episode_steps=1000,
    kwargs={'direction': [1., 0.], 'n': 5, 'max_u': 5.,
            'l_i': 1., 'k': 10., 'm_i': 1., 'h': 0.001}
)
