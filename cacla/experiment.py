from gym.envs.swimmer.remy_swimmer_env import SwimmerEnv
from cacla.cacla_agent import CACLA_agent
import matplotlib.pyplot as plt
import numpy as np
import ray

@ray.remote
def experience(gamma, alpha, sigma):
    # Initialization
    H = 100
    env = SwimmerEnv()

    # Train the agent
    n_iter = 10000
    agent = CACLA_agent(gamma=gamma, alpha=alpha, sigma=sigma, H=H)
    rewards = agent.run(env, n_iter, train=True)
    print(f"Last rewards of the experience gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma}: {rewards[-1]}")

    # Plot the graph
    timesteps = np.linspace(0, n_iter, n_iter//H)
    plt.plot(timesteps, rewards, label=f"gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma}")
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel(f"Sum of last {H} rewards")
    plt.title(f"CACLA on {env.envName} learning curve")
    plt.savefig(f"results/cacla/gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}.png")
    plt.close()


if __name__ == '__main__':
    ray.init()
    gammas = np.linspace(0.1, 0.95, 10)
    alphas = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
    sigmas = [1, 0.1, 0.001, 0.0001]
    tasks = []
    for g, a, s in [(g, a, s) for g in gammas for a in alphas for s in sigmas]:
        tasks.append(experience.remote(g, a, s))
    ray.get(tasks)