"""
Inifinte-horizon, discrete-time LQR

`A \in R^{nxn}`, `B \in R^{nxm}`, `Q \in R^{nxn}`, `R \in R^{mxm}`
Linear system: `x[k+1] = A x[k] + B u[k]` where `x[k] \in R^n`: state, `u[k] \in R^m`: action/control input
Cost function: `c[k+1] = x[k]^T Q x[k] + u[k]^T R u[k]`
Controllable: $`\mathrm{rank}(B, AB, A^2B, ...., A^{n-1}B) = n`$
Q, R: symmetric, i.e., `Q^T = Q`, `R^T = R`
Q is positive semi-definite, R is positive definite
"""

import gym
from gym import spaces
import numpy as np

class LinearQuadReg(gym.Env):
    """
    Basic LQR class. Parameters: A, B, Q, R are matrices satisfying the definition above.
    """

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
        return obs, rew, False, {'action': action}

    def reset(self):
        self.state = np.random.rand(self.observation_space.shape[0])
        return self.state

    def set_state(self, state):
        self.state = state


class EasyParamLinearQuadReg(LinearQuadReg):
    """
    For studying sim-real transfer, we implement this toy problem. The matrices are parameterized by theta \in R.
    """

    def __init__(self, theta):
        A = np.array([[0, 1], [1, 0]]) * theta
        B = np.array([[0], [1]]) * theta
        Q = np.array([[1, 0], [0, 1]])
        R = np.array([[1]])
        super().__init__(A, B, Q, R)
        self.op_norm_der_A = 1
        self.op_norm_der_B = 1


class BoundedEasyLinearQuadReg(EasyParamLinearQuadReg):
    """
    LQR with bounded state-action space.
    """

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
        action = self.reset_inbound(action, self.max_a)
        obs = self.A @ self.state + self.B @ action
        obs = self.reset_inbound(obs, self.max_s)
        self.state = obs
        rew = - (self.state.transpose() @ self.Q @ self.state + action.transpose() @ self.R @ action)
        return obs, rew, False, {'action': action}
