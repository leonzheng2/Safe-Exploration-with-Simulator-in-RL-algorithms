from envs.gym_lqr.lqr_env import LinearQuadReg
import numpy as np
import sklearn.datasets
from cacla.cacla_agent import CACLA_agent
import matplotlib.pyplot as plt
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

# Environment
n_obs = 8
n_ac = 2
A = np.random.rand(n_obs, n_obs)
B = np.random.rand(n_obs, n_ac)
E = sklearn.datasets.make_spd_matrix(n_obs)
F = sklearn.datasets.make_spd_matrix(n_ac)
lqr = LinearQuadReg(A, B, E, F)

# Agent
n_iter = 100000
gamma = 0.8
alpha = 0.01
sigma = 0.1
agent = CACLA_agent(gamma=gamma, alpha=alpha, sigma=sigma)
results = agent.run(lqr, n_iter)
print(results)

# Plot graph
H = 1000
t = np.linspace(H, n_iter, n_iter - H)
plt.plot(t, window_convolution(results, H), label=f"gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma}")
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel(f"Rewards")
plt.title(f"CACLA on LQR learning curve")
plt.savefig(f"results/cacla/LQR/gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}.png")
plt.show()
plt.close()