"""

"""


import numpy as np


class Basic_ARS():
    """
    Basic ARS Agent for solving continuous control tasks. Without Safe Exploration.
    """
    def rollout(self, real_env, policy, H, render=False):
        """

        :param real_env:
        :param policy:
        :param H:
        :param render:
        :return:
        """
        obs = real_env.reset()
        R = 0
        states = []
        for i in range(H):
            if render:
                real_env.render()
            ac = policy @ obs
            new_obs, rew, done, info = real_env.step(ac)
            R += rew
            if done:
                break
            obs = new_obs
            states.append(obs)
        return R, states

    def sort_directions(self, deltas, rewards):
        """
        Sort the directions deltas by max{r_k_+, r_k_-}
        :param deltas: array of matrices
        :param rewards: array of float
        :return: bijection of range(len(deltas))
        """
        max_rewards = [max(rewards[2 * i], rewards[2 * i + 1]) for i in range(len(deltas))]
        indices = np.argsort(max_rewards).tolist()
        return indices[::-1]

    def update_policy(self, deltas, returns, order, alpha):
        """
        Update the linear policy following the update step, after collecting the rewards
        :param deltas: array of matrices
        :param rewards: array of floats
        :param order: bijection of range(len(deltas))
        :param alpha: float, step size
        :return: void, self.policy is updated
        """
        used_rewards = []
        for i in order:
            used_rewards += [returns[2 * i], returns[2 * i + 1]]
        sigma_r = np.std(used_rewards)
        grad = np.zeros(self.policy.shape)
        for i in order:
            grad += (returns[2 * i] - returns[2 * i + 1]) * deltas[i]
        grad /= (len(order) * sigma_r)
        self.policy += alpha * grad

    def train(self, n_iter, real_env, N, b, alpha, nu, H):
        """

        :param n_iter:
        :param real_env:
        :param N:
        :param b:
        :param alpha:
        :param nu:
        :return:
        """
        n_obs = real_env.observation_space.shape[0]
        n_ac = real_env.action_space.shape[0]
        self.policy = np.zeros((n_ac, n_obs))
        all_returns = []
        states = []
        for n in range(n_iter): # One iteration of training
            deltas = [2 * np.random.rand(*self.policy.shape) - 1 for i in range(N)]
            returns = []
            for i in range(N):
                policy_1 = self.policy + nu * deltas[i]
                policy_2 = self.policy - nu * deltas[i]
                r_1, states_1 = self.rollout(real_env, policy_1, H)
                r_2, states_2 = self.rollout(real_env, policy_2, H)
                returns.append(r_1)
                returns.append(r_2)
                states.append(states_1)
                states.append(states_2)
            order = self.sort_directions(deltas, returns)
            self.update_policy(deltas, returns, order[:b], alpha)
            all_returns.append(np.mean(returns))
            if n%10==0:
                print(f"Iteration {n}/{n_iter}: return = {all_returns[-1]}")
        return np.array(all_returns), np.array(states)


class Safe_ARS(Basic_ARS):

    def __init__(self, cost, real_threshold, sim_threshold, sim_env):
        self.cost = cost
        self.real_thresh = real_threshold
        self.sim_thresh = sim_threshold
        self.sim_env = sim_env

    def isSafe(self, cost, thresh, env, state, action):
        """

        :param cost:
        :param sim_thresh:
        :param sim_env:
        :param action:
        :return:
        """
        env.set_state(state)
        obs, _, _, _ = env.step(action)
        return cost(obs) <= thresh

    def rollout(self, real_env, policy, H, render=False):
        """

        :param real_env:
        :param sim_env:
        :param policy:
        :param H:
        :param render:
        :return:
        """
        obs = real_env.reset()
        R = 0
        states = []
        for i in range(H):
            if render:
                real_env.render()
            ac = policy @ obs
            # Safe exploration
            if self.isSafe(self.cost, self.sim_thresh, self.sim_env, obs, ac):
                new_obs, rew, done, info = real_env.step(ac)
                if self.cost(new_obs) > self.real_thresh:
                    print(f"Safety constraint not satisfied: {self.cost(new_obs)} > {self.real_thresh}")
                R += rew
                if done:
                    break
                obs = new_obs
                states.append(obs)
            else:
                states.append(states[-1] if len(states) > 0 else obs)
        return R, states
