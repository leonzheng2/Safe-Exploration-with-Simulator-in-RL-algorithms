""" Inifinte-horizon, discrete-time LQR"""

import gym
from gym import spaces
import numpy as np

class LinearQuadReg(gym.Env):

    def __init__(self, A, B, Q, R):
        # Parameters
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        # Space
        n_obs = A.shape[1]
        n_ac = B.shape[1]
        inf = 1000
        self.observation_space = spaces.Box(-inf, inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-inf, inf, shape=(n_ac,), dtype=np.float32)

    def step(self, action: np.ndarray):
        action = np.array(action)
        obs = self.A @ self.state + self.B @ action
        self.state = obs
        rew = - (self.state.transpose() @ self.Q @ self.state + action.transpose() @ self.R @ action)
        return obs, rew, False, {}

    def reset(self):
        self.state = np.random.rand(self.observation_space.shape[0])
        return self.state

