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

    def set_state(self, state):
        self.state = state


class EasyParamLinearQuadReg(LinearQuadReg):

    def __init__(self, theta):
        A = np.array([[0, 1], [1, 0]]) * theta
        B = np.array([[0], [1]]) * theta
        Q = np.array([[1, 0], [0, 1]])
        R = np.array([[1]])
        super().__init__(A, B, Q, R)
        self.op_norm_der_A = 1
        self.op_norm_der_B = 1


class BoundedEasyLinearQuadReg(EasyParamLinearQuadReg):

    def __init__(self, theta, max_s, max_a):
        super().__init__(theta)
        self.max_s = max_s
        self.max_a = max_a

    def reset_inbound(self, x: np.ndarray, M):
        for i in range(len(x)):
            if abs(x[i]) > M:
                x[i] = abs(x[i]) / x[i] * M
        return x

    def step(self, action: np.ndarray):
        action = np.array(action)
        obs = self.A @ self.state + self.B @ action
        obs = self.reset_inbound(obs, self.max_s)
        self.state = obs
        rew = - (self.state.transpose() @ self.Q @ self.state + action.transpose() @ self.R @ action)
        return obs, rew, False, {}
