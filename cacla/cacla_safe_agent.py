import numpy as np
from envs.gym_lqr.lqr_env import EasyParamLinearQuadReg, BoundedEasyLinearQuadReg

class Constraint():

    def __init__(self, cost, l, L_c):
        self.cost = cost
        self.l = l
        self.L_c = L_c

    def satisfied(self, state):
        return self.cost(state) <= self.l


class CACLA_LQR_SE_agent:

    def __init__(self, real_env: EasyParamLinearQuadReg, simulator: EasyParamLinearQuadReg, epsilon, constraint):
        self.real_env = real_env
        self.simulator = simulator
        self.constraint = constraint
        self.epsilon = epsilon
        # Function approximators
        n_obs = real_env.observation_space.shape[0]
        n_ac = real_env.action_space.shape[0]
        self.F = np.zeros((n_ac, n_obs))
        self.V = np.zeros(n_obs)

    def forward_action_FA(self, state):
        state = np.array(state)
        return self.F @ state

    def backward_action_FA(self, alpha, action, state, FA_action):
        (n_ac, n_obs) = self.F.shape
        for i in range(n_ac):
            for j in range(n_obs):
                self.F[i, j] += alpha * (action[i] - FA_action[i]) * (state[j])

    def forward_value_FA(self, state):
        state_squared = state ** 2
        return self.V.transpose() @ state_squared

    def backward_value_FA(self, alpha, delta, state):
        n_obs = self.V.shape[0]
        for j in range(n_obs):
            self.V[j] += alpha * delta * state[j] ** 2

    def compute_sim_threshold(self, L_c, epsilon, state, action):
        L_theta = self.simulator.op_norm_der_A * np.linalg.norm(state, 2) + self.simulator.op_norm_der_B * np.linalg.norm(action, 2)
        return self.constraint.l - epsilon * L_c * L_theta

    def run(self, n_iter, gamma, alpha, sigma, H=1000):
        # Save
        states = []
        actions = []
        rewards = []

        state = self.real_env.reset() # Initialization
        for i in range(n_iter):
            FA_act = self.forward_action_FA(state)  # Actor function approximation
            action = np.random.multivariate_normal(FA_act, sigma * np.identity(len(FA_act)))  # Gaussian policy
            # Safe Exploration
            sim_threshold = self.compute_sim_threshold(self.constraint.L_c, self.epsilon, state, action)
            sim_constraint = Constraint(self.constraint.cost, sim_threshold, self.constraint.L_c)
            self.simulator.set_state(state)
            sim_state, _, _, _ = self.simulator.step(action)  # Simulator step
            if sim_constraint.satisfied(sim_state):
                new_state, reward, done, info = self.real_env.step(action)  # Real world step
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

    def __init__(self, real_env: BoundedEasyLinearQuadReg, simulator: BoundedEasyLinearQuadReg, epsilon, constraint):
        super().__init__(real_env, simulator, epsilon, constraint)
        self.real_env = real_env
        self.simulator = simulator
        L_theta = self.simulator.op_norm_der_A * real_env.max_s + self.simulator.op_norm_der_B * real_env.max_a
        self.sim_threshold = self.constraint.l - epsilon * constraint.L_c * L_theta

    def run(self, n_iter, gamma, alpha, sigma, H=1000):
        # Save
        states = []
        actions = []
        rewards = []

        state = self.real_env.reset() # Initialization
        for i in range(n_iter):
            FA_act = self.forward_action_FA(state)  # Actor function approximation
            action = np.random.multivariate_normal(FA_act, sigma * np.identity(len(FA_act)))  # Gaussian policy
            # Safe Exploration
            sim_constraint = Constraint(self.constraint.cost, self.sim_threshold, self.constraint.L_c)
            self.simulator.set_state(state)
            sim_state, _, _, _ = self.simulator.step(action)  # Simulator step
            if sim_constraint.satisfied(sim_state):
                new_state, reward, done, info = self.real_env.step(action)  # Real world step
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