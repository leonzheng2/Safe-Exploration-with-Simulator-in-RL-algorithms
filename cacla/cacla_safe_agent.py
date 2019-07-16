"""
Classes for implementing Safe Exploration with CACLA to solve easy instances of parametrized LQR.
The safety state constraint is implemented by the class `Constraint`.
For using Safe Exploration with CACLA on LQR with assumption of bounded state and action spaces, use class `CACLA_Bounded_LQR_SE_agent`.
For using Safe Exploration with CACLA on LQR without assumption of bounded state and action spaces, use class `CACLA_LQR_SE_agent`.
"""


import numpy as np
from cacla.cacla_agent import CACLA_LQR_agent
from envs.gym_lqr.lqr_env import EasyParamLinearQuadReg, BoundedEasyLinearQuadReg

class Constraint():
    """
    Class for implementing a safety state constraint.
    """

    def __init__(self, cost, l, L_c):
        """
        Constructor
        :param cost: function of state, outputs real numbers
        :param l: safety threshold
        :param L_c: Lipschitz constant of the cost function
        """
        self.cost = cost
        self.l = l
        self.L_c = L_c

    def satisfied(self, state):
        """
        Return True if the constraint is satisfied for the given state.
        :param state:
        :return:
        """
        return self.cost(state) <= self.l


class CACLA_LQR_SE_agent(CACLA_LQR_agent):
    """
    Implementation of CACLA using Safe Exploration for solving LQR, without assumption of bounded state and action spaces.
    The agent always satifies a given safety constraint.
    It computes a simulator threshold at each timestep, given the current state and action.
    """

    def __init__(self, real_env: EasyParamLinearQuadReg, simulator: EasyParamLinearQuadReg, epsilon, constraint):
        """
        Constructor.
        :param real_env: EasyParamLinearQuadReg, representing the real world environment, with unknown real world parameter
        :param simulator: EasyParamLinearQuadReg, representing the simulator environmnet, with estimated real world parameter
        :param epsilon: approximation error
        :param constraint: Constraint
        """
        super(CACLA_LQR_SE_agent, self).__init__(real_env)
        self.simulator = simulator
        self.constraint = constraint
        self.epsilon = epsilon
        # Function approximators
        n_obs = real_env.observation_space.shape[0]
        n_ac = real_env.action_space.shape[0]
        self.F = np.zeros((n_ac, n_obs))
        self.V = np.zeros(n_obs)

    def compute_sim_threshold(self, L_c, epsilon, state, action):
        """
        Compute the simulator threshold at each timestep.
        :param L_c: Lipschitz constant of the cost function
        :param epsilon: real world parameter approximation error
        :param state: current state, numpy array
        :param action: current action, numpy array
        :return: simulator threhsold, float
        """
        L_theta = self.simulator.op_norm_der_A * np.linalg.norm(state, 2) + self.simulator.op_norm_der_B * np.linalg.norm(action, 2) # Lipschitz constant
        return self.constraint.l - epsilon * L_c * L_theta

    def run(self, n_iter, gamma, alpha, sigma, H=1000):
        """
        Run the training, using Safe Exploration.
        :param n_iter: number of iterations
        :param gamma: discount factor, between 0 and 1
        :param alpha: step size
        :param sigma: standard deviation of Gaussian policy
        :param H: print training progress at each timestep H
        :return: void
        """
        # Save
        states = []
        actions = []
        rewards = []

        state = self.env.reset() # Initialization
        for i in range(n_iter):
            FA_act = self.forward_action_FA(state)  # Actor function approximation
            action = np.random.multivariate_normal(FA_act, sigma * np.identity(len(FA_act)))  # Gaussian policy
            # Safe Exploration
            sim_threshold = self.compute_sim_threshold(self.constraint.L_c, self.epsilon, state, action)
            sim_constraint = Constraint(self.constraint.cost, sim_threshold, self.constraint.L_c)
            self.simulator.set_state(state)
            sim_state, _, _, _ = self.simulator.step(action)  # Simulator step
            if sim_constraint.satisfied(sim_state):
                new_state, reward, done, info = self.env.step(action)  # Real world step
                if not self.constraint.satisfied(new_state):
                    print(f"Constraint not satisfied for state: {new_state}")
                temp_diff = reward + gamma * self.forward_value_FA(new_state) - self.forward_value_FA(state) # Temporal difference
                self.backward_value_FA(alpha, temp_diff, state)  # Update critic FA
                if temp_diff > 0:
                    self.backward_action_FA(alpha, action, state, FA_act)  # CACLA

                states.append(state)
                actions.append(info['action'])
                rewards.append(reward)
                state = new_state
            else:
                if len(states) > 0:
                    states.append(states[-1])
                if len(actions) > 0:
                    actions.append(actions[-1])
                if len(rewards) > 0:
                    rewards.append(rewards[-1])

            if i%H == 0 and i > 0:
                print(f"Iteration {i}/{n_iter}: reward: {reward}")

        return np.array(states), np.array(actions), np.array(rewards)


class CACLA_Bounded_LQR_SE_agent(CACLA_LQR_SE_agent):
    """
    Implementation of CACLA using Safe Exploration for solving LQR, with assumption of bounded state and action spaces.
    The agent always satifies a given safety constraint.
    It computes a simulator threshold at each timestep, given the current state and action.
    """

    def __init__(self, real_env: BoundedEasyLinearQuadReg, simulator: BoundedEasyLinearQuadReg, epsilon, constraint):
        """
        Constructor.
        :param real_env: BoundedEasyLinearQuadReg, representing the real world environment, with unknown real world parameter
        :param simulator: BoundedEasyLinearQuadReg, representing the simulator environmnet, with estimated real world parameter
        :param epsilon: approximation error
        :param constraint: Constraint
        """
        super().__init__(real_env, simulator, epsilon, constraint)
        self.simulator = simulator
        L_theta = self.simulator.op_norm_der_A * real_env.max_s + self.simulator.op_norm_der_B * real_env.max_a
        self.sim_threshold = self.constraint.l - epsilon * constraint.L_c * L_theta

    def run(self, n_iter, gamma, alpha, sigma, H=1000):
        """
        Run the training, using Safe Exploration.
        :param n_iter: number of iterations
        :param gamma: discount factor, between 0 and 1
        :param alpha: step size
        :param sigma: standard deviation of Gaussian policy
        :param H: print training progress at each timestep H
        :return: void
        """
        # Save
        states = []
        actions = []
        rewards = []

        state = self.env.reset() # Initialization
        for i in range(n_iter):
            FA_act = self.forward_action_FA(state)  # Actor function approximation
            action = np.random.multivariate_normal(FA_act, sigma * np.identity(len(FA_act)))  # Gaussian policy
            # Safe Exploration
            sim_constraint = Constraint(self.constraint.cost, self.sim_threshold, self.constraint.L_c)
            self.simulator.set_state(state)
            sim_state, _, _, _ = self.simulator.step(action)  # Simulator step
            if sim_constraint.satisfied(sim_state):
                new_state, reward, done, info = self.env.step(action)  # Real world step
                if not self.constraint.satisfied(new_state):
                    print(f"Constraint not satisfied for state: {new_state}")
                temp_diff = reward + gamma * self.forward_value_FA(new_state) - self.forward_value_FA(state) # Temporal difference
                self.backward_value_FA(alpha, temp_diff, state)  # Update critic FA
                if temp_diff > 0:
                    self.backward_action_FA(alpha, action, state, FA_act)  # CACLA

                states.append(state)
                actions.append(info['action'])
                rewards.append(reward)
                state = new_state
            else:
                if len(states) > 0:
                    states.append(states[-1])
                if len(actions) > 0:
                    actions.append(actions[-1])
                if len(rewards) > 0:
                    rewards.append(rewards[-1])

            if i%H == 0 and i > 0 and len(rewards) > 0:
                print(f"Iteration {i}/{n_iter}: reward: {reward}")

        return np.array(states), np.array(actions), np.array(rewards)
