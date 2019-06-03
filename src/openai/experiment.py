import numpy as np
import gym
import matplotlib.pyplot as plt
import ray
from gym.envs.registration import register
from dataclasses import dataclass

@dataclass
class EnvParam:
    # Environment parameters
    name: str
    n: int
    H: int
    l_i: float
    m_i: float
    h: float

@dataclass
class ARSParam:
    # Agent parameters
    select_V1: bool
    n_iter: int
    H: int
    N: int
    b: int
    alpha: float
    nu: float

@ray.remote
class Worker():

    def __init__(self, envName, policy, n, H, l_i, m_i, h):
        """
        Constructor.
        :param envName: string
        :param policy: matrix
        :param n: int
        :param H: int
        :param l_i: float
        :param m_i: float
        :param h: float
        """
        # print(f"New worker - {policy[0][0]}")
        self.policy = policy
        self.H = H
        self.env = gym.make(envName)

    def select_action(self, observation):
        """
        Compute the action vector, using linear policy
        :param observation: vector
        :return: vector
        """
        observation = np.array(observation)
        action = np.matmul(self.policy, observation)
        return action

    def rollout(self):
        """
        Doing self.H steps following the given policy, and return the final total reward.
        :return: float
        """
        total_reward = 0
        observation = self.env.reset()
        for t in range(self.H):
            # self.env.render()
            action = self.select_action(observation)
            observation, reward, done, info = self.env.step(action)
            # print(f"State = {observation}")
            total_reward += reward
            if done:
                return total_reward
        return total_reward

@ray.remote
class ARSAgent():

    def __init__(self, envName, agent_param, seed=None):
        # Environment
        self.env = gym.make(envName)

        # Agent
        self.policy = np.zeros((self.env.action_space.shape[0], self.env.observation_space.shape[0]))  # Linear policy
        self.n_it = agent_param.n_iter

        # Agent parameters
        self.N = agent_param.N
        self.b = agent_param.b
        self.alpha = agent_param.alpha
        self.nu = agent_param.nu
        self.H = agent_param.H

        # V2
        self.V1 = agent_param.select_V1
        self.mean = np.zeros(self.env.observation_space.shape[0])
        self.covariance = np.identity(self.env.observation_space.shape[0])
        self.saved_states = []

        self.n_seed = seed
        np.random.seed(self.n_seed)

    def select_action(self, policy, observation):
        """
        Compute the action vector, using linear policy
        :param policy: matrix
        :param observation: vector
        :return: vector
        """
        observation = np.array(observation)
        if self.V1 is True:
            return np.matmul(policy, observation)
        # ARS V2
        diag_covariance = np.diag(self.covariance)**(-1/2)
        policy = np.matmul(policy, np.diag(diag_covariance))
        action = np.matmul(policy, observation - self.mean)
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
            if self.V1 is not True:
                self.saved_states.append(observation)
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
        max_rewards = [max(rewards[2*i], rewards[2*i+1]) for i in range(len(deltas))]
        indices = np.argsort(max_rewards).tolist()
        return indices[::-1]

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
        #     worker = Worker.remote(policy, self.n, self.H, self.l_i, self.m_i, self.h)
        #     rewards.append(worker.rollout.remote())
        # rewards = ray.get(rewards)
        order = self.sort_directions(deltas, rewards)
        self.update_policy(deltas, rewards, order)

        if self.V1 is not True:
            states_array = np.array(self.saved_states)
            self.mean = np.mean(states_array, axis=0)
            self.covariance = np.cov(states_array.T)
            # print(f"mean = {self.mean}")
            # print(f"cov = {self.covariance}")

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
            if j % 10 == 0:
                print(f"Seed {self.n_seed} ------ V1 = {self.V1}; n={self.env.n}; h={self.env.h}; alpha={self.alpha}; nu={self.nu}; N={self.N}; b={self.b}; m_i={self.env.m_i}; l_i={self.env.l_i} ------ Iteration {j}/{self.n_it}: {r}")
        self.env.close()
        return np.array(rewards)

class Experiment():

    def __init__(self, env_param):
        """
        Constructor setting up parameters for the experience
        :param env_param: EnvParam
        """
        # Set up environment
        del gym.envs.registry.env_specs[env_param.name]
        register(
            id=env_param.name,
            entry_point='gym.envs.swimmer:SwimmerEnv', # HERE CHANGE THE PATH TO THE OWN CREATED ENVIRONMENT
            max_episode_steps=env_param.H,
            kwargs={'direction': [1., 0.], 'n': env_param.n, 'max_u': 5., 'l_i': env_param.l_i, 'k': 10., 'm_i': env_param.m_i, 'h': env_param.h}
        )
        self.env_param = env_param

    # @ray.remote
    def plot(self, n_seed, agent_param):
        """
        Plotting learning curve
        :param n_seed: number of seeds for plotting the curve, int
        :param agent_param: ARSParam
        :return: void
        """

        print(f"select_V1={agent_param.select_V1}")
        print(f"n={self.env_param.n}")
        print(f"h={self.env_param.h}")
        print(f"alpha={agent_param.alpha}")
        print(f"nu={agent_param.nu}")
        print(f"N={agent_param.N}")
        print(f"b={agent_param.b}")
        print(f"m_i={self.env_param.m_i}")
        print(f"l_i={self.env_param.l_i}")

        # Seeds
        r_graphs = []
        for i in range(n_seed):
            agent = ARSAgent.remote('LeonSwimmer-v0', agent_param, seed=i)
            r_graphs.append(agent.runTraining.remote())
        r_graphs = np.array(ray.get(r_graphs))

        # Plot graphs
        plt.figure(figsize=(10,8))
        for rewards in r_graphs:
            plt.plot(rewards)
        plt.title(f"V1={agent_param.select_V1}, n={self.env_param.n}, seeds={n_seed}, h={self.env_param.h}, alpha={agent_param.alpha}, "
                  f"nu={agent_param.nu}, N={agent_param.N}, b={agent_param.b}, n_iter={agent_param.n_iter}, "
                  f"m_i={round(self.env_param.m_i, 2)}, l_i={round(self.env_param.l_i,2)}")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        np.save(f"array/ars_n={self.env_param.n}_V1={agent_param.select_V1}_random_seeds={n_seed}_h={self.env_param.h}_alpha={agent_param.alpha}_"
                f"nu={agent_param.nu}_N={agent_param.N}_b={agent_param.b}_n_iter={agent_param.n_iter}_"
                f"m_i={round(self.env_param.m_i, 2)}_l_i={round(self.env_param.l_i,2)}", r_graphs)
        plt.savefig(f"new/ars_n={self.env_param.n}_V1={agent_param.select_V1}_random_seeds={n_seed}_h={self.env_param.h}_alpha={agent_param.alpha}_"
                f"nu={agent_param.nu}_N={agent_param.N}_b={agent_param.b}_n_iter={agent_param.n_iter}_"
                f"m_i={round(self.env_param.m_i, 2)}_l_i={round(self.env_param.l_i,2)}.png")
        # plt.show()
        plt.close()

        # Plot mean and std
        plt.figure(figsize=(10,8))
        x = np.linspace(0, agent_param.n_iter-1, agent_param.n_iter)
        mean = np.mean(r_graphs, axis=0)
        std = np.std(r_graphs, axis=0)
        plt.plot(x, mean, 'k', color='#CC4F1B')
        plt.fill_between(x, mean-std, mean+std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title(f"V1={agent_param.select_V1}, n={self.env_param.n}, seeds={n_seed}, h={self.env_param.h}, alpha={agent_param.alpha}, "
                  f"nu={agent_param.nu}, N={agent_param.N}, b={agent_param.b}, n_iter={agent_param.n_iter}, "
                  f"m_i={round(self.env_param.m_i, 2)}, l_i={round(self.env_param.l_i,2)}")
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.savefig(f"new/ars_n={self.env_param.n}_V1={agent_param.select_V1}_random_seeds={n_seed}_h={self.env_param.h}_alpha={agent_param.alpha}_"
                f"nu={agent_param.nu}_N={agent_param.N}_b={agent_param.b}_n_iter={agent_param.n_iter}_"
                f"m_i={round(self.env_param.m_i, 2)}_l_i={round(self.env_param.l_i,2)}_average.png")
        # plt.show()
        plt.close()

if __name__ == '__main__':
    ray.init(num_cpus=1)

    for n in range(3,4):
        env_param = EnvParam('LeonSwimmer-v0', n=n, H=1000, l_i=1., m_i=1., h=1e-3)
        agent_param = ARSParam(select_V1=True, n_iter=100, H=1000, N=1, b=1, alpha=0.0075, nu=0.01)
        exp = Experiment(env_param)
        exp.plot(n_seed=1, agent_param=agent_param)

    # n_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    # max_rewards = []
    # train_90 = []
    # train_95 = []
    # train_99 = []
    #
    # for n in n_list:
    #     if n == 8 or n == 9 or n>10:
    #         rewards = np.load(f"save/n/ars_n={n}_random_seeds=8_h=0.001_alpha=0.0075_nu=0.01_N=1_b=1_n_iter=1000_m_i=1.0_l_i=1.0.npy")
    #     else:
    #         rewards = np.load(f"save/n/ars_n={n}_random_seeds=6_h=0.001_alpha=0.0075_nu=0.01_N=1_b=1_n_iter=1000_m_i=1.0_l_i=1.0.npy")
    #
    #     mean = np.mean(rewards, axis=0)
    #     max_rewards.append(np.max(mean))
    #
    #     k = 20
    #     smooth_mean = np.convolve(mean, np.ones((len(mean) // k,)) / (len(mean) // k), 'valid')
    #     max = np.max(smooth_mean)
    #     for alpha in [0.9, 0.95, 0.99]:
    #         n_train = np.argmax(mean > alpha*max)
    #         print(f"Time to achieve {alpha} of max: {n_train}")
    #         if alpha == 0.9:
    #             train_90.append(n_train)
    #         elif alpha == 0.95:
    #             train_95.append(n_train)
    #         else:
    #             train_99.append(n_train)
    #
    # plt.plot(n_list, train_90, '-o', label="lambda=0.90")
    # plt.plot(n_list, train_95, '-o', label="lambda=0.95")
    # plt.plot(n_list, train_99, '-o', label="lambda=0.99")
    # plt.title("Time to reach lambda*max of smoothed mean")
    # plt.xlabel("Segments")
    # plt.ylabel("Iterations")
    # plt.legend()
    # plt.show()
    # plt.close()
    #
    # plt.plot(n_list, max_rewards, '-o', label="max_rewards")
    # plt.title("Max average reward achieved with ARS")
    # plt.xlabel("Segments")
    # plt.ylabel("Max average reward")
    # plt.legend()
    # plt.show()
    # plt.close()

    # grad_smooth = np.gradient(smooth_mean)
    # plt.plot(grad_smooth)
    # plt.show()
    #
    # k = 10
    # smooth_grad_smooth = np.convolve(grad_smooth, np.ones((len(grad_smooth) // k,)) / (len(grad_smooth) // k), 'valid')
    # plt.plot(smooth_grad_smooth)
    # plt.show()
    #
    # for k in [10, 20, 30, 40, 50]:
    #     smooth_mean = np.convolve(mean, np.ones((len(mean) // k,)) / (len(mean) // k), 'valid')
    #     x = np.linspace(len(mean)//k//2, len(grad) - len(mean)//k//2, len(mean) - len(mean)//k + 1)
    #     plt.plot(mean, label="original")
    #     # plt.title(f"k={k}")
    #     plt.plot(x, smooth_mean, label="smooth")
    #     plt.xlabel("Iteration")
    #     plt.ylabel("Reward")
    #     plt.legend()
    #     plt.show()
    #     print(f"Max of smooth k={k}: {np.max(smooth_mean)}")
    #
    # for alpha in [0.9, 0.95, 0.99]:
    #     n_train = np.argmax(mean > alpha*max)
    #     print(f"Time to achieve {alpha} of max: {n_train}")
    #
    # for k in [5, 8, 10, 12, 15]:
    #     smooth_grad = np.convolve(grad, np.ones((len(grad) // k,)) / (len(grad) // k), 'valid')
    #     # x = np.linspace(len(grad)//k//2, len(grad) - len(grad)//k//2, len(grad) - len(grad)//k + 1)
    #     # # print(x)
    #     # plt.plot(x, smooth, label="smooth")
    #     # plt.show()
    #     first_zero = np.argmax(smooth_grad < 0)
    #     if first_zero > 0:
    #         print(f"Time when smooth gradient (k={k}) is zero: {first_zero + len(grad)//k}")
    #     else:
    #         print(f"Time when smooth gradient (k={k}) is minimum: {np.argmin(smooth_grad) + len(grad)//k}")
    #
    #
    # # print(smooth)
    # # plt.plot(mean, label="mean")
    # # plt.plot(grad, label="grad")
    # # print(len(grad))
    # # print(len(smooth))
    # # x = np.linspace(len(grad)//k//2, len(grad) - len(grad)//k//2, len(grad) - len(grad)//k + 1)
    # # # print(x)
    # # plt.plot(x, smooth, label="smooth")
    # # plt.show()
    #


