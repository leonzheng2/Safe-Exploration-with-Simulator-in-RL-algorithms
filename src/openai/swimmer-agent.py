"""
    Implementation of the Augmented Random Search algorithm applied on the Swimmer Task
    Leon Zheng
"""

import ray
import gym
import numpy as np
import matplotlib.pyplot as plt

@ray.remote
class SwimmerAgent():

    def __init__(self, n_it=1000, N=1, b=1, H=1000, alpha=0.02, nu=0.02, seed=None):
        self.env = gym.make('Swimmer-v2') # Environment
        self.policy = np.zeros((self.env.action_space.shape[0], self.env.observation_space.shape[0])) # Linear policy
        self.n_it = n_it
        self.N = N
        self.b = b
        self.H = H
        self.alpha = alpha
        self.nu = nu
        self.seed = seed
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
            used_rewards += [rewards[2*i], rewards[2*i+1]]
        sigma_r = np.std(used_rewards)

        grad = np.zeros(self.policy.shape)
        for i in order:
            grad += (rewards[2*i] - rewards[2*i+1])*deltas[i]
        grad /= (self.b*sigma_r)

        self.policy += self.alpha*grad

    def runOneIteration(self):
        """
        Performing one whole iteration of the ARS algorithm
        :return: void, self.policy is updated
        """
        deltas = [2*np.random.rand(*self.policy.shape)-1 for i in range(self.N)]
        rewards = []
        for i in range(2*self.N):
            if i%2==0:
                policy = self.policy + self.nu*deltas[i//2]
            else:
                policy = self.policy - self.nu*deltas[i//2]
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
            if j%(self.n_it//10) == 0:
                print(f"------ alpha={self.alpha}; nu={self.nu}; seed={self.seed} ------")
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
            agent = SwimmerAgent.remote(n_it=500, alpha=alpha, nu=nu)
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

def plot_random_seed(n_seed, alpha=0.02, nu=0.02):
    # Seeds
    r_graphs = []
    for i in range(n_seed):
        agent = SwimmerAgent.remote(n_it=500, seed=i, alpha=alpha, nu=nu)
        r_graphs.append(agent.runTraining.remote())

    # Plot graphs
    for rewards in r_graphs:
        rewards = ray.get(rewards)
        plt.plot(rewards)
    plt.title(f"n_seed={n_seed}, alpha={alpha}, nu={nu}")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.savefig("../../results/ars_random_seeds-.png")
    plt.show()

if __name__ == '__main__':
    ray.init()
    plot_random_seed(50)
    