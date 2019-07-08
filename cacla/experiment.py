from gym.envs.swimmer.remy_swimmer_env import SwimmerEnv
from cacla.cacla_agent import CACLA_agent
import matplotlib.pyplot as plt
import numpy as np
import ray


@ray.remote
def experience(env, gamma, alpha, sigma, n_iter, H):
    # Train the agent
    rewards = []
    for seed in range(3):
        agent = CACLA_agent(gamma=gamma, alpha=alpha, sigma=sigma)
        rewards.append(agent.run(env, n_iter, H=H, train=True))

    # Timesteps, mean, std
    t = np.linspace(0, n_iter, n_iter)
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
    plt.title(f"CACLA on {env.envName} learning curve")
    plt.savefig(f"results/cacla/gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}.png")
    plt.close()


if __name__ == '__main__':
    ray.init()
    env = SwimmerEnv()
    gammas = np.linspace(0.1, 0.95, 10)
    alphas = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
    sigmas = [1, 0.1, 0.001, 0.0001]
    tasks = []
    for g, a, s in [(g, a, s) for g in gammas for a in alphas for s in sigmas]:
        tasks.append(experience.remote(env, g, a, s))
    ray.get(tasks)