"""
Swimmer implementation - OpenAI Gym environment
Model from Remy Coulom
By Leon Zheng
"""

import numpy as np
import gym
import math
from gym import spaces


class SwimmerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, direction=[1., 0.], n=3, max_u=5., l_i=1., k=10., m_i=1., h=0.001):
        """
        Constructor.
        Need to call set_parameters() to set environment parameters.
        """
        # Parameters of the environment
        self.direction = np.array(direction)
        self.n = n
        self.max_u = max_u
        self.l_i = l_i
        self.k = k
        self.m_i = m_i
        self.h =h

        # Observation and action space
        n_obs = 2 * self.n + 2
        n_action = self.n - 1
        inf = 1000
        self.observation_space = spaces.Box(-inf, inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-self.max_u, self.max_u, shape=(n_action,), dtype=np.float32)

    def step(self, action):
        """
        Doing one step of RL environment. Returns next step given the given and previous state
        :param action: array
        :return: array, float, boolean, dictionary
        """
        self.G_dot, self.theta, self.theta_dot = self.next_observation(action, self.G_dot, self.theta,
                                                                       self.theta_dot)
        ob = self.get_state()
        reward = self.get_reward()
        done = self.check_terminal()
        info = {}
        return ob, reward, done, info

    def reset(self):
        """
        Reset the environment to the initial state. Return the corresponding observation.
        :return: array
        """
        self.G_dot = np.full(2, 0.)
        self.theta = np.full(self.n, math.pi/2)
        self.theta_dot = np.full(self.n, 0.)
        return self.get_state()

    def next_observation(self, torque, G_dot, theta, theta_dot):
        """
        Helper method for doing one step. Compute accelerations and do semi-implicit Euler intergration
        :param torque: array
        :param G_dot: array
        :param theta: array
        :param theta_dot: array
        :return: array, array, array
        """
        G_dotdot, theta_dotdot = self.compute_accelerations(torque, G_dot, theta, theta_dot)
        # Semi-implicit Euler integration
        G_dot = G_dot + self.h * G_dotdot
        theta_dot = theta_dot + self.h * theta_dotdot
        theta = theta + self.h * theta_dot
        return G_dot, theta, theta_dot

    def compute_accelerations(self, torque, G_dot, theta, theta_dot):
        """
        Solve the linear equation to compute accelerations.
        :param torque: array
        :param G_dot: array
        :param theta: array
        :param theta_dot: array
        :return: array, array
        """

        A_dot_X, A_dot_Y, A_dotdot_X, A_dotdot_Y = self.compute_points_speed_acc(G_dot, theta, theta_dot)
        force_X, force_Y = self.compute_joint_force(theta, A_dot_X, A_dot_Y, A_dotdot_X, A_dotdot_Y)
        system = self.compute_dynamic_matrix(theta, theta_dot, torque, force_X, force_Y)

        G_dotdot, theta_dotdot = self.solve(system)

        return G_dotdot, theta_dotdot

    def compute_points_speed_acc(self, G_dot, theta, theta_dot):
        A_dot_X = np.zeros(self.n + 1)
        A_dot_Y = np.zeros(self.n + 1)
        A_dotdot_X = np.zeros((self.n + 1, self.n + 3))
        A_dotdot_Y = np.zeros((self.n + 1, self.n + 3))

        G_dot_X = 0.
        G_dot_Y = 0.
        G_dotdot_X = np.zeros(self.n + 3)
        G_dotdot_Y = np.zeros(self.n + 3)

        # Frame of swimmer's head
        for i in range(1, self.n + 1):
            # A_dot_i = A_dot_{i-1} + l * theta_dot_{i-1} * n_{i-1}
            A_dot_X[i] = A_dot_X[i - 1] - self.l_i * theta_dot[i - 1] * np.sin([theta[i - 1]])
            A_dot_Y[i] = A_dot_Y[i - 1] + self.l_i * theta_dot[i - 1] * np.cos([theta[i - 1]])

            # A_dot_i = A_dot_{i-1} + l * theta_dotdot_{i-1} * n_{i-1} - l * theta_dot_{i-1}^2 * p_{i-1}
            A_dotdot_X[i] = np.array(A_dotdot_X[i - 1], copy=True)
            A_dotdot_Y[i] = np.array(A_dotdot_Y[i - 1], copy=True)
            A_dotdot_X[i][2 + (i - 1)] -= self.l_i * np.sin(theta[i - 1])
            A_dotdot_Y[i][2 + (i - 1)] += self.l_i * np.cos(theta[i - 1])
            A_dotdot_X[i][self.n + 2] -= self.l_i * theta_dot[i - 1] ** 2 * np.cos(theta[i - 1])
            A_dotdot_Y[i][self.n + 2] -= self.l_i * theta_dot[i - 1] ** 2 * np.sin(theta[i - 1])

            # G_dot
            G_dot_X += 1 / self.n * (A_dot_X[i - 1] + A_dot_X[i]) / 2
            G_dot_Y += 1 / self.n * (A_dot_Y[i - 1] + A_dot_Y[i]) / 2

            # G_dotdot
            G_dotdot_X += 1 / self.n * (A_dotdot_X[i - 1] + A_dotdot_X[i]) / 2
            G_dotdot_Y += 1 / self.n * (A_dotdot_Y[i - 1] + A_dotdot_Y[i]) / 2

            # matprint(f"1.{i} A_dotdot_Y", A_dotdot_Y)

        # Change of frame
        A_dot_X += G_dot[0] - G_dot_X
        A_dot_Y += G_dot[1] - G_dot_Y
        for i in range(self.n + 1):
            A_dotdot_X[i][0] += 1
            A_dotdot_Y[i][1] += 1
            A_dotdot_X[i] -= G_dotdot_X
            A_dotdot_Y[i] -= G_dotdot_Y
            # matprint(f"2.{i} A_dotdot_Y", A_dotdot_Y)


        return A_dot_X, A_dot_Y, A_dotdot_X, A_dotdot_Y

    def compute_joint_force(self, theta, A_dot_X, A_dot_Y, A_dotdot_X, A_dotdot_Y):
        force_X = np.zeros((self.n + 1, self.n + 3))
        force_Y = np.zeros((self.n + 1, self.n + 3))

        for i in range(1, self.n + 1):
            force_X[i] = force_X[i - 1] + self.m_i * (A_dotdot_X[i - 1] + A_dotdot_X[i]) / 2
            force_Y[i] = force_Y[i - 1] + self.m_i * (A_dotdot_Y[i - 1] + A_dotdot_Y[i]) / 2
            # F is the friction force
            n_i = np.array([-np.sin(theta[i - 1]), np.cos(theta[i - 1])])
            G_dot_i = np.array([(A_dot_X[i - 1] + A_dot_X[i]) / 2, (A_dot_Y[i - 1] + A_dot_Y[i]) / 2])
            F = -self.k * self.l_i * np.dot(G_dot_i, n_i)
            force_X[i][self.n + 2] -= F * n_i[0]
            force_Y[i][self.n + 2] -= F * n_i[1]
            # matprint(f"1.{i} force_X", force_X)

        return force_X, force_Y

    def compute_dynamic_matrix(self, theta, theta_dot, torque, force_X, force_Y):
        system = np.zeros((self.n+2, self.n+3))

        system[0] = force_X[self.n]
        system[1] = force_Y[self.n]
        for i in range(1, self.n+1):
            system[2 + (i-1)] += self.l_i / 2 * (np.cos(theta[i-1]) * (force_Y[i] + force_Y[i-1])
                                                - np.sin(theta[i-1]) * (force_X[i] + force_X[i-1]))
            system[2 + (i-1)][2 + (i-1)] -= self.m_i * self.l_i**2 / 12
            system[2 + (i-1)][self.n + 2] += self.k * theta_dot[i-1] * self.l_i**3 / 12
            if i-2 >= 0:
                system[2 + (i - 1)][self.n + 2] += torque[i-2]
            if i-1 < self.n - 1:
                system[2 + (i - 1)][self.n + 2] -= torque[i-1]

        return system

    def solve(self, system):
        A = system[:self.n+2, :self.n+2]
        B = - system[:, self.n+2]
        X = np.linalg.solve(A, B)

        return X[:2], X[2:]

    def get_state(self):
        """
        Return the observation in an array form
        :return: array
        """
        ob = self.G_dot.tolist()
        for i in range(self.n):
            ob += [self.theta[i], self.theta_dot[i]]
        return ob

    def get_reward(self):
        """
        Compute the reward for the current observation
        :return: float
        """
        return self.G_dot.dot(self.direction)

    def check_terminal(self):
        """
        Check if the episode is finished or not.
        :return: boolean
        """
        # TODO
        return False

    def render(self, mode='human'):
        return


def matprint(name, mat, fmt="g"):
    print(f"------------ {name} ------------")
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")
    print("--------------------------------")
    print("")

if __name__ == '__main__':
    import gym_swimmer
    import math

    env = SwimmerEnv()
    # action = np.full(env.action_space.shape[0], env.max_u / 2.)
    # env.reset()
    # print(f"State: {env.get_state()}")
    # ob, reward, done, info = env.step(action)
    # print(f"State: {env.get_state()}")
    # print(f"Reward: {reward}")

    torque = np.full(env.action_space.shape[0], env.max_u / 2.)
    G_dot = np.full(2, 0.)
    theta = np.full(env.n, math.pi/2)
    theta_dot = np.full(env.n, 0.)
    print(G_dot, theta, theta_dot)
    A_dot_X, A_dot_Y, A_dotdot_X, A_dotdot_Y = env.compute_points_speed_acc(G_dot, theta, theta_dot)
    print(f"A_dot_X = {A_dot_X}")
    matprint("A_dotdot_X", A_dotdot_X)
    print(f"A_dot_Y = {A_dot_Y}")
    matprint("A_dotdot_Y", A_dotdot_Y)
    force_X, force_Y = env.compute_joint_force(theta, A_dot_X, A_dot_Y, A_dotdot_X, A_dotdot_Y)
    matprint("force_X", force_X)
    system = env.compute_dynamic_matrix(theta, theta_dot, torque, force_X, force_Y)
    matprint("system", system)
    G_dotdot, theta_dotdot = env.solve(system)
    print(f"G_dotdot = {G_dotdot}")
    print(f"theta_dotdot = {theta_dotdot}")
