"""
    Implementation of the Augmented Random Search algorithm applied on the Swimmer Task
    Leon Zheng
"""

import gym
import numpy as np


class SwimmerAgent():

    def __init__(self, n_it=1000, N=1, b=1, H=2000, alpha=0.02, nu=0.02):
        self.env = gym.make('Swimmer-v2') # Environment
        self.policy = np.zeros((self.env.action_space.shape[0], self.env.observation_space.shape[0])) # Linear policy
        self.n_it = n_it
        self.N = N
        self.b = b
        self.H = H
        self.alpha = alpha
        self.nu = nu

    def select_action(self, policy, observation):
        observation = np.array(observation)
        action = np.matmul(policy, observation)
        return action

    def rollout(self, policy):
        reward = 0
        observation = self.env.reset()
        for t in range(self.H):
            # self.env.render()
            action = self.select_action(policy, observation)
            observation, reward, done, info = self.env.step(action)
            if done:
                return reward
        return reward

    def sort_directions(self, deltas, rewards):
        return range(len(deltas))

    def update_policy(self, deltas, rewards, order):
        used_rewards = []
        for i in range(self.b):
            used_rewards += [rewards[2*order[i]], rewards[2*order[i]+1]]
        sigma_r = np.std(used_rewards)

        grad = np.zeros(self.policy.shape)
        for i in range(self.b):
            grad += (rewards[2*order[i]] - rewards[2*order[i]+1])*deltas[order[i]]
        grad /= (self.b*sigma_r)

        self.policy += self.alpha*grad

    def runOneIteration(self):
        deltas = [np.random.rand(self.policy.shape[0], self.policy.shape[1]) for i in range(self.N)]
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
        for j in range(self.n_it):
            self.runOneIteration()
            r = self.rollout(self.policy)
            print(f"Iteration {j}: {r}")
            # print(f"Policy: {agent.policy}")
        self.env.close()

if __name__ == '__main__':
    agent = SwimmerAgent()
    agent.runTraining()