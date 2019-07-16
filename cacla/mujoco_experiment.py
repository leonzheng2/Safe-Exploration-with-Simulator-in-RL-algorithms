"""
Script for training CACLA agent on OpenAI Gym MuJoCo task without Safe Exploration.

Select the environment `env`.

Choose hyperparameters of CACLA:
    - discount factor `gamma`
    - standard deviation for Gaussian policy `sigma`
    - step size for weights update `alpha`
    - number of iterations `n_iter`
"""


import gym
from cacla.cacla_agent import CACLA_agent
import matplotlib.pyplot as plt
import numpy as np
from cacla.window import window_convolution

def experience(env, gamma, alpha, sigma, n_iter, H):
    """
    Helper method wrapping one experience, containing training of the CACLA agent to solve a given MuJoCo environment
    :param env: OpenAI Gym environment, MuJoCo. For example, use `Swimmer-v2`
    :param gamma: discount factor [0,1]
    :param alpha: step size, float
    :param sigma: std deviation of Gaussian policy, step size
    :param n_iter: number of iterations, integer
    :param H: print training progress at each timestep H
    :return: void
    """
    # Train the agent
    rewards = []
    for seed in range(3):
        agent = CACLA_agent(gamma=gamma, alpha=alpha, sigma=sigma)
        rewards.append(agent.run(env, n_iter, H=H, train=True, render=False))

    # Window convolve
    for i in range(len(rewards)):
        rewards[i] = window_convolution(rewards[i], H)

    # Timesteps, mean, std
    t = np.linspace(H, n_iter, n_iter - H)
    mean = np.nanmean(rewards, axis=0)
    std = np.nanstd(rewards, axis=0)
    print(f"Last mean reward obtained for gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma}: {mean[-1]}")

    # Plot graph
    plt.plot(t, mean, 'k', color='#CC4F1B', label=f"gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma}")
    plt.fill_between(t, mean - std, mean + std, alpha=0.5,
                     edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel(f"Sum of last {H} rewards")
    plt.title(f"CACLA on Swimmer-v2 learning curve")
    plt.savefig(f"results/cacla/mujoco/gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}.png")
    plt.close()

''' Experience '''

# Initialization of the environment
env = gym.make('Swimmer-v2')

# CACLA agent parameters
gammas = [0.0, 0.8, 0.9, 0.95, 0.99]
alpha = 0.01
sigma = 0.1
gamma = gammas[1]
n_iter = 100000

# Training the agent
experience(env, gamma, alpha, sigma, n_iter, 1000)
