"""
Classes containing the implementation of CACLA Algorithm, based on the paper `Reinforcement Learning
in Continuous Action Spaces` by Hado van Hasselt and Marco A. Wiering.

Class `CACLA_agent` implements CACLA algorithm using neural networks for function approximation.
It uses classes `TwoLayersNet`, `Actor_FA` and `Critic_FA` for the implementation.

Class `CACLA_LQR_agent` implements CACLA algorithm for solving LQR, by using prior knowledge of the optimal value
function and controller from control theory.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TwoLayersNet(nn.Module):
    """
    Using Pytorch library to create a simple neural network with one hidden layer.
    """
    def __init__(self, n_in, H1):
        """
        Constructor
        :param n_in: number of neurons in the input layer
        :param H1: number of neurons in the hidden layer
        """
        super(TwoLayersNet, self).__init__()
        self.linear1 = nn.Linear(n_in, H1).double()
        self.linear2 = nn.Linear(H1, 1).double()

    def forward(self, x):
        """
        Forward
        :param x: input tensor
        :return: output tensor
        """
        x = x.double()
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def update_weights(self, step, x):
        """
        Backpropagate to compute gradients and update the weights using a given step.
        :param step: double
        :param x: tensor
        :return: void
        """
        self.zero_grad()
        target = self.forward(x)
        target.backward(torch.ones(1).double())
        with torch.no_grad():
            for param in self.parameters():
                # print(param)
                # print(f"Gradient: {param.grad.data.sum()}")
                param += step * param.grad



class ActorFA():
    """
    Function approximator using simple neural network to compute the action, given an input state.
    """
    def __init__(self, n_obs, H1, n_ac):
        """
        Constructor
        :param n_obs: dimension of the state space
        :param H1: number of neurons in the hidden layer
        :param n_ac: dimension of the action space
        """
        self.network = [TwoLayersNet(n_obs, H1) for i in range(n_ac)]
        self.n_ac = n_ac

    def approximate_action(self, state):
        """
        Compute the action given the input state.
        :param state: array
        :return: numpy array
        """
        x = torch.tensor(state)
        seq = [self.network[i](x) for i in range(self.n_ac)]
        out = torch.cat(seq)
        return out.detach().numpy()

    def update_weights(self, alpha, state, action, FA_act):
        """
        Update the weights of the function approximator.
        :param alpha: float, step size
        :param state: current state
        :param action: selected action
        :param FA_act: function approximator evaluated at `state`
        :return: void
        """
        x = torch.tensor(state)
        for i in range(self.n_ac):
            step = alpha * (action[i] - FA_act[i])
            self.network[i].update_weights(step, x)


class CriticFA():
    """
    Function approximator to compute the value function of a given state.
    """
    def __init__(self, n_obs, H1):
        """
        Constructor.
        :param n_obs: neurons of the input layer
        :param H1: neurons of the hidden layer
        """
        self.network = TwoLayersNet(n_obs, H1)

    def approximate_value(self, state):
        """
        Compute the value of a given input state.
        :param state: array
        :return: float
        """
        x = torch.tensor(state)
        return self.network(x).item()

    def update_weigths(self, alpha, delta, state):
        """
        Update the weights of the function approximator.
        :param alpha: float, step size
        :param delta: float, temporal difference
        :param state: array
        :return: void
        """
        x = torch.tensor(state)
        step = alpha * delta
        self.network.update_weights(step, x)


class CACLA_agent:
    """
    Implementation of CACLA algorithm using neural networks for function approximators.
    """

    def __init__(self, gamma, alpha, sigma):
        """
        Constructor.
        :param gamma: discount factor, float between 0 and 1
        :param alpha: step size, float
        :param sigma: standard deviation for the Gaussian policy
        """
        # Parameters
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma

    def run(self, env, n_iter, H=1000, train=True, render=False):
        """
        Run the training of CACLA agent.
        :param env: OpenAI Gym environment
        :param n_iter: number of iterations, integer
        :param H: integer, printing training progress at each H iterations
        :param train: set True for updating function approximator weigths
        :param render: set True for rendering the training
        :return: void
        """
        # Function approximator
        n_obs = env.observation_space.shape[0]
        n_ac = env.action_space.shape[0]
        actor = ActorFA(n_obs, 12, n_ac)
        critic = CriticFA(n_obs, 12)

        rewards = []
        state = env.reset() # Initialization
        for i in range(n_iter):
            if render:
                env.render() # Rendering
            if not train:
                with torch.no_grad():
                    FA_act = actor.approximate_action(state)  # Actor function approximation
                    action = np.random.multivariate_normal(FA_act,
                                                           self.sigma * np.identity(len(FA_act)))  # Gaussian policy
                    new_state, reward, done, info = env.step(action)  # Environment step
            else:
                FA_act = actor.approximate_action(state)  # Actor function approximation
                # print(f"FA_act = {FA_act}")
                action = np.random.multivariate_normal(FA_act, self.sigma * np.identity(len(FA_act)))  # Gaussian policy
                # print(action)
                new_state, reward, done, info = env.step(action)  # Environment step
                # print(f"Value: {critic.approximate_value(state)}")
                temp_diff = reward + self.gamma * critic.approximate_value(new_state) - critic.approximate_value(state) # Temporal difference
                critic.update_weigths(self.alpha, temp_diff, state)  # Update critic FA
                # print(f"Temporal difference: {temp_diff}")
                if temp_diff > 0:
                    actor.update_weights(self.alpha, state, action, FA_act)  # CACLA
                state = new_state

            rewards.append(reward)
            # if done:
            #     break
            if i%H == 0 and i > 0:
                print(f"Iteration {i}/{n_iter}: reward: {reward}")

        return rewards


class CACLA_LQR_agent:
    """
    Implementation of CACLA algorithm for solving LQR, using knowledge from control theory about the optimal value function and policy.
    """

    def __init__(self, env):
        self.env = env
        # Function approximators
        n_obs = env.observation_space.shape[0]
        n_ac = env.action_space.shape[0]
        self.F = np.zeros((n_ac, n_obs))
        self.V = np.zeros(n_obs)


    def forward_action_FA(self, state):
        """
        Simple linear actor: approxmiate action from given state.
        :param state: array
        :return: array
        """
        state = np.array(state)
        return self.F @ state

    def backward_action_FA(self, alpha, action, state, FA_action):
        """
        Backpropagate and update the weights of the actor.
        :param alpha: float, step size
        :param action: array, selected action from policy
        :param state: array
        :param FA_action: array, computed action from function approximator
        :return: void
        """
        (n_ac, n_obs) = self.F.shape
        for i in range(n_ac):
            for j in range(n_obs):
                self.F[i, j] += alpha * (action[i] - FA_action[i]) * (state[j])

    def forward_value_FA(self, state):
        """
        Compute value function of a given state. Quadratic value function.
        :param state: array
        :return: float
        """
        state_squared = state ** 2
        return self.V.transpose() @ state_squared

    def backward_value_FA(self, alpha, delta, state):
        """
        Backpropagate and update the weights of the value function.
        :param alpha: float, step size
        :param delta: float, temporal difference
        :param state: array
        :return: void
        """
        n_obs = self.V.shape[0]
        for j in range(n_obs):
            self.V[j] += alpha * delta * state[j] ** 2

    def run(self, n_iter, gamma, alpha, sigma, H=1000):
        """
        Run the training of CACLA.
        :param n_iter: number of iterations
        :param gamma: discount factor, [0, 1]
        :param alpha: step size
        :param sigma: standard deviation for Gaussian policy
        :param H: print training progress every H steps
        :return: void
        """
        # Save
        states = []
        actions = []
        rewards = []

        state = self.env.reset() # Initialization
        for i in range(n_iter):
            FA_act = self.forward_action_FA(state)  # Actor function approximation
            # print(f"FA_act = {FA_act}")
            action = np.random.multivariate_normal(FA_act, sigma * np.identity(len(FA_act)))  # Gaussian policy
            new_state, reward, done, info = self.env.step(action)  # Environment step
            temp_diff = reward + gamma * self.forward_value_FA(new_state) - self.forward_value_FA(state) # Temporal difference
            self.backward_value_FA(alpha, temp_diff, state)  # Update critic FA
            # print(f"Temporal difference: {temp_diff}")
            if temp_diff > 0:
                self.backward_action_FA(alpha, action, state, FA_act)  # CACLA

            states.append(state)
            actions.append(info['action'])
            rewards.append(reward)
            state = new_state
            # if done:
            #     break
            if i%H == 0 and i > 0:
                print(f"Iteration {i}/{n_iter}: reward: {reward}")

        return np.array(states), np.array(actions), np.array(rewards)
