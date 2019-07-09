from envs.gym_lqr.lqr_env import LinearQuadReg
import numpy as np
import sklearn.datasets
from cacla.cacla_agent import CACLA_LQR_agent
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
n_obs = 1
n_ac = 1
A = np.ones((1,1))
B = np.ones((1,1))
Q = np.ones((1,1))
R = np.ones((1,1))
lqr = LinearQuadReg(A, B, Q, R)

# Agent
n_iter = 100000
gamma = 0.8
alpha = 0.0001
sigma = 0.1
agent = CACLA_LQR_agent(lqr)
results = agent.run(n_iter, gamma, alpha, sigma)
print(agent.F)

# Plot graph
H = 1000
t = np.linspace(H, n_iter, n_iter - H)
plt.plot(t, window_convolution(results, H), label=f"gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma}")
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel(f"Sum of the last {H} rewards")
plt.title(f"CACLA on LQR learning curve")
plt.savefig(f"results/cacla/LQR/gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}.png")
plt.show()
plt.close()