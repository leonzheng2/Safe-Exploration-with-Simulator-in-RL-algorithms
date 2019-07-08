""" Inifinte-horizon, discrete-time LQR"""

import gym
from gym import spaces
import numpy as np

class LinearQuadReg(gym.Env):

    def __init__(self, A, B, E, F):
        # Parameters
        self.A = A
        self.B = B
        self.E = E
        self.F = F

        # Space
        n_obs = A.shape[1]
        n_ac = B.shape[1]
        inf = 1000
        self.observation_space = spaces.Box(-inf, inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-inf, inf, shape=(n_ac,), dtype=np.float32)

    def step(self, action: np.ndarray):
        action = np.array(action)
        obs = self.A @ self.state + self.B @ action
        rew = - (self.state.transpose() @ self.E @ self.state + action.transpose() @ self.F @ action)
        return obs, rew, False, {}

    def reset(self):
        self.state = np.zeros(self.observation_space.shape[0])
        return self.state

