"""
Swimmer implementation - OpenAI Gym environment
Model from Remy Coulom
By Leon Zheng
"""

import numpy as np
import gym
from gym import spaces


class SwimmerEnv(gym.Env):

    def __init__(self):
        # Parameters of the environment
        self.direction = np.array([1., 0.])
        self.n = 3
        self.max_u = 5.
        self.l_i = 3./self.n
        self.k = 10.
        self.m_i = 3./self.n
        self.h = 0.003

        # Observation and action space
        n_obs = 2 * self.n + 2
        n_action = self.n - 1
        inf = float("inf")
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
        self.G_dot = np.full(2, 0.001)
        self.theta = np.full(self.n, 0.001)
        self.theta_dot = np.full(self.n, 0.001)
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
        F_friction, M_friction = self.compute_friction(G_dot, theta, theta_dot)

        A = np.zeros((5 * self.n + 2, 5 * self.n + 2))
        B = np.zeros(5 * self.n + 2)

        for i in range(1, self.n + 1):
            A[i - 1, i - 1] = self.m_i * self.l_i ** 2 / 12.
            A[i - 1, self.n + 2 * (i - 1) + 0] = +self.l_i / 2. * np.sin(theta[i - 1])
            A[i - 1, self.n + 2 * (i - 1) + 2] = +self.l_i / 2. * np.sin(theta[i - 1])
            A[i - 1, self.n + 2 * (i - 1) + 1] = -self.l_i / 2. * np.cos(theta[i - 1])
            A[i - 1, self.n + 2 * (i - 1) + 3] = -self.l_i / 2. * np.cos(theta[i - 1])

            B[i - 1] = M_friction[i - 1]
            if i - 2 >= 0:
                B[i - 1] += torque[i - 2]
            if i - 1 < self.n - 1:
                B[i - 1] -= torque[i - 1]

        A[self.n][self.n] = 1
        A[self.n + 1][self.n + 1] = 1

        for i in range(1, self.n + 1):
            for d in range(2):
                A[self.n + 2 + 2 * (i - 1) + d][self.n + 2 * (i - 1) + d] = +1
                A[self.n + 2 + 2 * (i - 1) + d][self.n + 2 * i + d] = -1
                A[self.n + 2 + 2 * (i - 1) + d][3 * self.n + 2 * i + d] = self.m_i
                B[self.n + 2 + 2 * (i - 1) + d] = F_friction[i - 1][d]

        A[3 * self.n + 2][3 * self.n] = 1
        A[3 * self.n + 3][3 * self.n + 1] = 1

        for i in range(1, self.n):
            for d in range(2):
                A[3 * self.n + 4 + 2 * (i - 1) + d][3 * self.n + 2 * i + d] = +1
                A[3 * self.n + 4 + 2 * (i - 1) + d][3 * self.n + 2 * (i + 1) + d] = -1

                if d == 0:
                    A[3 * self.n + 4 + 2 * (i - 1) + d][i - 1] = -self.l_i / 2. * np.sin(theta[i - 1])
                    A[3 * self.n + 4 + 2 * (i - 1) + d][i] = -self.l_i / 2 * np.sin(theta[i])
                    B[3 * self.n + 4 + 2 * (i - 1) + d] = +self.l_i / 2. * (
                            np.cos(theta[i - 1]) * theta_dot[i - 1] ** 2 + np.cos(theta[i]) * theta_dot[i] ** 2)

                else:
                    A[3 * self.n + 4 + 2 * (i - 1) + d][i - 1] = +self.l_i / 2. * np.cos(theta[i - 1])
                    A[3 * self.n + 4 + 2 * (i - 1) + d][i] = +self.l_i / 2. * np.cos(theta[i])
                    B[3 * self.n + 4 + 2 * (i - 1) + d] = +self.l_i / 2. * (
                            np.sin(theta[i - 1]) * theta_dot[i - 1] ** 2 + np.sin(theta[i]) * theta_dot[i] ** 2)

        # print("-----------------------Matrix A------------------------")
        # matprint(A)
        # print("------------------------------------------------------")
        #
        # print("---------------------Matrix invA----------------------")
        # matprint(np.linalg.inv(A))
        # print("-----------------------------------------------------")
        #
        # print(f"B = {B}")
        try:
            X = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            print("-----------------------Matrix A------------------------")
            matprint(A)
            print("------------------------------------------------------")
            np.savetxt("invalid_A.txt", A)
            np.savetxt("invalid_state.txt", self.get_state())

        # print(f"X = {X}")
        theta_dotdot = np.array(X[:self.n])
        # print(f"theta_dotdot = {theta_dotdot}")
        G_i_dotdot = X[3 * self.n + 2:].reshape(self.n, 2)
        # print(f"G_i_dotdot = {G_i_dotdot}")
        G_dotdot = np.mean(G_i_dotdot, axis=0)
        # print(f"G_dotdot = {G_dotdot}")

        return G_dotdot, theta_dotdot

    def compute_friction(self, G_dot, theta, theta_dot):
        """
        Compute frictions for constructing the matrix used in compute_accelerations().
        :param G_dot: array
        :param theta: array
        :param theta_dot: array
        :return: array, array
        """

        normal = np.array([np.array([-np.sin(theta[i]), np.cos(theta[i])]) for i in range(self.n)])
        # print(f"normal = {normal}")

        M = np.zeros((self.n, self.n))
        vecX = np.zeros(self.n)
        vecY = np.zeros(self.n)
        for i in range(self.n):
            if i == 0:
                M[0] = np.full(self.n, 1 / self.n)
                vecX[0] = G_dot[0]
                vecY[0] = G_dot[1]
            else:
                M[i][i] = 1
                M[i][i - 1] = -1
                vecX[i] = self.l_i / 2 * (theta_dot[i - 1] * normal[i - 1][0] + theta_dot[i] * normal[i][0])
                vecY[i] = self.l_i / 2 * (theta_dot[i - 1] * normal[i - 1][1] + theta_dot[i] * normal[i][1])
        # print(f"theta = {theta}")
        # print(f"M = {M}")
        # print(f"vecX = {vecX}")
        G_i_dot_x = np.linalg.solve(M, vecX)
        G_i_dot_y = np.linalg.solve(M, vecY)
        G_i_dot = np.array([[G_i_dot_x[i], G_i_dot_y[i]] for i in range(self.n)])
        # print(f"Method 2 - G_i_dot = {G_i_dot}")

        F_friction = [-self.k * self.l_i * np.dot(G_i_dot[i], normal[i]) * normal[i] for i in range(self.n)]
        M_friction = [-self.k * theta_dot[i] * self.l_i ** 3 / 12. for i in range(self.n)]

        # print(f"F_friction = {np.array(F_friction)}")
        # print(f"M_friction = {M_friction}")

        return F_friction, M_friction

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

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

if __name__ == '__main__':
    import gym_swimmer
    env = gym.make('Leon-swimmer-v0')
    action = np.full(env.action_space.shape[0], env.max_u/2.)
    # env.reset()
    # print(f"State: {env.get_state()}")
    # ob, reward, done, info = env.step(action)
    # print(f"State: {env.get_state()}")
    # print(f"Reward: {reward}")

    G_dot = np.full(2, 0.001)
    theta = np.full(env.n, 0.001)
    theta_dot = np.full(env.n, 0.001)
    print(G_dot, theta, theta_dot)
    G_dotdot, theta_dotdot = env.compute_accelerations(action, G_dot, theta, theta_dot)
    print(f"Accelerations: G_dotdot = {G_dotdot}, theta_dotdot = {theta_dotdot}")
    G_dot, theta, theta_dot = env.next_observation(action, G_dot, theta, theta_dot)
    print(G_dot, theta, theta_dot)