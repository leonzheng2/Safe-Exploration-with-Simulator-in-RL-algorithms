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
    return np.array(v)/H

# Instance 2
x = 1.0
A_2 = np.array([[0, 1], [1, 0]]) * x
B_2 = np.array([[0], [1]]) * x
Q_2 = np.array([[1, 0], [0, 1]])
R_2 = np.array([[1]])
lqr_2 = LinearQuadReg(A_2, B_2, Q_2, R_2)

# Agent
n_iter = 200000
gamma = 1
sigma = 0.1
alpha = 0.0001
agent = CACLA_LQR_agent(lqr_2)
states, actions, rewards = agent.run(n_iter, gamma, alpha, sigma)
print(agent.F)

# Plot states
plt.figure(figsize=(8, 6))
plt.scatter(states[:,0], states[:,1], c=np.linspace(0, 1, len(states)))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(f'States during CACLA training on LQR, {n_iter} iterations')
plt.savefig(f"results/cacla/param_LQR/2_x={x}_gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}_states.png")
plt.show()
plt.close()

# Plot actions
plt.figure(figsize=(8, 6))
plt.scatter(range(len(states)), actions, c=np.linspace(0, 1, len(states)))
plt.xlabel('Timesteps')
plt.ylabel('Actions')
plt.title(f'Actions during CACLA training on LQR, {n_iter} iterations')
plt.savefig(f"results/cacla/param_LQR/2_x={x}_gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}_actions.png")
plt.show()
plt.close()

# Plot rewards
H = 1000
t = np.linspace(H, n_iter, n_iter - H)
plt.plot(t, window_convolution(rewards, H), label=f"gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma}")
plt.legend()
plt.xlabel("Timesteps")
plt.ylabel(f"Average of the last {H} rewards")
plt.title(f"CACLA on LQR learning curve")
plt.savefig(f"results/cacla/param_LQR/2_x={x}_gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}_rewards.png")
plt.show()
plt.close()
