import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import ray


class SwimmerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, direction=np.array([1., 0.]), n=3, max_u=5., l_i=1., k=10., m_i=1., h=0.003):
        # Parameters of the environment
        self.direction = direction
        self.n = n
        self.max_u = max_u
        self.l_i = l_i
        self.k = k
        self.m_i = m_i
        self.h = h

        # Observation and action space
        n_obs = 2 * self.n + 2
        n_action = self.n - 1
        inf = 1000
        self.observation_space = spaces.Box(-inf, inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-self.max_u, self.max_u, shape=(n_action,), dtype=np.float32)

    def step(self, action):
        self.G_dot, self.theta, self.theta_dot = self.next_observation(action, self.G_dot, self.theta, self.theta_dot)

        ob = self.get_state()
        reward = self.get_reward()
        done = self.check_terminal()
        info = {}
        return ob, reward, done, info

    def reset(self):
        self.G_dot = np.full(2, 0.001)
        self.theta = np.full(self.n, 0.001)
        self.theta_dot = np.full(self.n, 0.001)
        return self.get_state()

    def next_observation(self, torque, G_dot, theta, theta_dot):
        G_dotdot, theta_dotdot = self.compute_accelerations(torque, G_dot, theta, theta_dot)
        # Semi-implicit Euler intergration
        G_dot = G_dot + self.h * G_dotdot
        theta_dot = theta_dot + self.h * theta_dotdot
        theta = theta + self.h * theta_dot
        return G_dot, theta, theta_dot

    def compute_accelerations(self, torque, G_dot, theta, theta_dot):
        F_friction, M_friction = self.compute_friction(G_dot, theta, theta_dot)

        A = np.zeros((5 * self.n + 2, 5 * self.n + 2))
        B = np.zeros(5 * self.n + 2)

        for i in range(1, self.n + 1):
            A[i - 1, i - 1] = self.m_i * self.l_i ** 2 / 12.
            A[i - 1, self.n + 2 * i + 0] = +self.l_i / 2. * np.sin(theta[i - 1])
            A[i - 1, self.n + 2 * i + 2] = +self.l_i / 2. * np.sin(theta[i - 1])
            A[i - 1, self.n + 2 * i + 1] = -self.l_i / 2. * np.cos(theta[i - 1])
            A[i - 1, self.n + 2 * i + 3] = -self.l_i / 2. * np.cos(theta[i - 1])

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

        # print(f"A = {A}")
        # print(f"invA = {np.linalg.inv(A)}")
        # print(f"B = {B}")

        X = np.linalg.solve(A, B)

        # print(f"X = {X}")

        theta_dotdot = np.array(X[:self.n])
        # print(f"theta_dotdot = {theta_dotdot}")
        G_i_dotdot = X[3 * self.n + 2:].reshape(self.n, 2)
        # print(f"G_i_dotdot = {G_i_dotdot}")
        G_dotdot = np.mean(G_i_dotdot, axis=0)
        # print(f"G_dotdot = {G_dotdot}")

        return G_dotdot, theta_dotdot

    def compute_friction(self, G_dot, theta, theta_dot):

        normal = np.array([np.array([-np.sin(theta[i]), np.cos(theta[i])]) for i in range(self.n)])

        G1_dot = np.array(G_dot)
        for i in range(1, self.n + 1):
            sum = np.zeros(2)
            for j in range(i):
                e = 0.5 if j == 0 or j == i - 1 else 1.
                sum += e * theta_dot[j] * normal[j]
            G1_dot -= self.l_i / self.n * sum

        G_i_dot = [G1_dot]
        for i in range(1, self.n):
            G_i_dot.append(
                G_i_dot[i - 1] + self.l_i / 2. * theta_dot[i - 1] * normal[i - 1] + self.l_i / 2. * theta_dot[i] *
                normal[i])

        F_friction = [-self.k * self.l_i * np.dot(G_i_dot[i], normal[i]) * normal[i] for i in range(self.n)]
        M_friction = [-self.k * theta_dot[i] * self.l_i ** 3 / 12. for i in range(self.n)]

        # print(f"F_friction = {np.array(F_friction)}")
        # print(f"M_friction = {M_friction}")

        return F_friction, M_friction

    def get_state(self):
        ob = self.G_dot.tolist()
        for i in range(self.n):
            ob += [self.theta[i], self.theta_dot[i]]
        return ob

    def get_reward(self):
        return self.G_dot.dot(self.direction)

    def check_terminal(self):
        # TODO
        return False

    def render(self, mode='human'):
        return


# @ray.remote
class ARSAgent():

    def __init__(self, n_it=1000, N=1, b=1, H=1000, alpha=0.02, nu=0.02, seed=None, n=3,
                 l_i=1., m_i=1., h=0.01):
        self.env = SwimmerEnv(n=n, l_i=l_i, m_i=m_i, h=h)  # Environment
        self.policy = np.zeros((self.env.action_space.shape[0], self.env.observation_space.shape[0]))  # Linear policy
        self.n_it = n_it
        self.N = N
        self.b = b
        self.H = H
        self.alpha = alpha
        self.nu = nu
        self.seed = seed
        self.n = n
        np.random.seed(self.seed)

    def select_action(self, policy, observation):
        """
        Compute the action vector, using linear policy
        :param policy: matrix
        :param observation: vector
        :return: vector
        """
        observation = np.array(observation)
        action = np.matmul(policy, observation)
        return action

    def rollout(self, policy):
        """
        Doing self.H steps following the given policy, and return the final total reward.
        :param policy: matrix
        :return: float
        """
        total_reward = 0
        observation = self.env.reset()
        for t in range(self.H):
            # self.env.render()
            action = self.select_action(policy, observation)
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                return total_reward
        return total_reward

    def sort_directions(self, deltas, rewards):
        """
        Sort the directions deltas by max{r_k_+, r_k_-}
        :param deltas: array of matrices
        :param rewards: array of float
        :return: bijection of range(len(deltas))
        """
        return range(len(deltas))

    def update_policy(self, deltas, rewards, order):
        """
        Update the linear policy following the update step, after collecting the rewards
        :param deltas: array of matrices
        :param rewards: array of floats
        :param order: bijection of range(len(deltas))
        :return: void, self.policy is updated
        """
        used_rewards = []
        for i in order:
            used_rewards += [rewards[2 * i], rewards[2 * i + 1]]
        sigma_r = np.std(used_rewards)

        grad = np.zeros(self.policy.shape)
        for i in order:
            grad += (rewards[2 * i] - rewards[2 * i + 1]) * deltas[i]
        grad /= (self.b * sigma_r)

        self.policy += self.alpha * grad

    def runOneIteration(self):
        """
        Performing one whole iteration of the ARS algorithm
        :return: void, self.policy is updated
        """
        deltas = [2 * np.random.rand(*self.policy.shape) - 1 for i in range(self.N)]
        rewards = []
        for i in range(2 * self.N):
            if i % 2 == 0:
                policy = self.policy + self.nu * deltas[i // 2]
            else:
                policy = self.policy - self.nu * deltas[i // 2]
            rewards.append(self.rollout(policy))
        order = self.sort_directions(deltas, rewards)
        self.update_policy(deltas, rewards, order)

    def runTraining(self):
        """
        Run the training. After each iteration, evaluate the current policy by doing one rollout. Save the obtained reward after each iteration.
        :return: array of float. Rewards obtained after each iteration.
        """
        rewards = []
        for j in range(self.n_it):
            self.runOneIteration()
            r = self.rollout(self.policy)
            rewards.append(r)
            if j % (self.n_it // self.n_it) == 0:
                print(f"------ alpha={self.alpha}; nu={self.nu}; seed={self.seed}; n={self.n} ------")
                print(f"Iteration {j}: {r}")
        self.env.close()
        return np.array(rewards)


def plot_hyperparameters(alphas, nus):
    """
    Run training using several configurations of alpha and nu
    :param alphas: array
    :param nus: array
    :return: plot and save graphs
    """
    # Hyperparameters
    r_graphs = []
    for alpha in alphas:
        for nu in nus:
            # agent = SwimmerAgent(n_it=500, alpha=alpha, nu=nu)
            # r_graphs.append((alpha, nu, agent.runTraining()))
            agent = ARSAgent.remote(n_it=500, alpha=alpha, nu=nu)
            r_graphs.append((alpha, nu, agent.runTraining.remote()))

    # Plot graphs
    for (alpha, nu, rewards) in r_graphs:
        rewards = ray.get(rewards)
        plt.plot(rewards, label=f"alpha={alpha}; nu={nu}")
    plt.title("Hyperparameters tuning")
    plt.legend(loc='upper left')
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.savefig("../../results/ars_hyperparameters-.png")
    plt.show()


def plot_random_seed(n_seed, alpha=0.02, nu=0.02, n=3, M=3, L=3):
    # Seeds
    r_graphs = []
    for i in range(n_seed):
        agent = ARSAgent(n_it=500, seed=i, alpha=alpha, nu=nu, n=n, m_i=M/n, l_i=L/n, h=0.001)
        r_graphs.append(agent.runTraining())

    # Plot graphs
    for rewards in r_graphs:
        # rewards = ray.get(rewards)
        plt.plot(rewards)
    plt.title(f"n_seed={n_seed}, alpha={alpha}, nu={nu}, n={n}")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    np.save(f"rewards_n={n}.png", np.array(r_graphs))
    plt.savefig(f"ars_random_seeds_n={n}.png")
    # plt.show()


if __name__ == '__main__':
    # ray.init(num_cpus=6)
    for n in range(5, 8):
        plot_random_seed(1, n=n)
