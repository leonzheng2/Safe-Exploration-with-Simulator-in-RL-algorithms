import gym
from cacla.cacla_agent import CACLA_agent
import matplotlib.pyplot as plt
import numpy as np
import queue

def window_convolution(a, H):
    v = []
    sum_H = 0
    q = queue.Queue(H)
    for i in range(len(a)):
        if q.full():
            sum_H -= q.get()
            q.put(a[i])
            sum_H += a[i]
            v.append(sum_H)
        else:
            q.put(a[i])
            sum_H += a[i]
    return np.array(v)

def experience(env, gamma, alpha, sigma, n_iter, H):
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

# Initialization
env = gym.make('Swimmer-v2')

# Train
gammas = [0.0, 0.8, 0.9, 0.95, 0.99]
alpha = 0.01
sigma = 0.1
gamma = gammas[1]

experience(env, gamma, alpha, sigma, 100000, 1000)
