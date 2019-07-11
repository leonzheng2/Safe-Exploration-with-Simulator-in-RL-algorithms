import numpy as np
import matplotlib.pyplot as plt
import queue
from envs.gym_lqr.lqr_env import EasyParamLinearQuadReg
from cacla.cacla_agent import CACLA_LQR_agent
from cacla.cacla_safe_agent import Constraint, CACLA_LQR_SE_agent


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

seed = np.random.randint(2**32)

### LQR Real World
theta_real = 1.0
lqr_real = EasyParamLinearQuadReg(theta_real)

### LQR Simulator
theta_sim = 0.99
lqr_sim = EasyParamLinearQuadReg(theta_sim)

### Agent
n_iter = 200000
gamma = 1
sigma = 0.1
alpha = 0.0001

# CACLA without Safe Exploration
np.random.seed(seed)
agent = CACLA_LQR_agent(lqr_real)
states_1, actions_1, rewards_1 = agent.run(n_iter, gamma, alpha, sigma)
print(agent.F)

# CACLA with Safe Exploration
np.random.seed(seed)
epsilon = abs(theta_real - theta_sim)
cost = lambda x: np.linalg.norm(x, np.inf)
L_c = 1
l = 1
constraint = Constraint(cost, l, L_c)
safe_agent = CACLA_LQR_SE_agent(lqr_real, lqr_sim, epsilon, constraint)
states_2, actions_2, rewards_2 = safe_agent.run(n_iter, gamma, alpha, sigma)
print(safe_agent.F)

### Results - Comparison

plt.close('all')
fig, ax = plt.subplots(3, 2, figsize=(10, 12))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Plot states
ax[0,0].scatter(states_1[:,0], states_1[:,1], c=np.linspace(0, 1, len(states_1)))
ax[0,0].set_xlabel('x1')
ax[0,0].set_ylabel('x2')
ax[0,0].set_title(f'States, without Safe Exploration')

ax[0,1].scatter(states_2[:,0], states_2[:,1], c=np.linspace(0, 1, len(states_2)))
ax[0,1].set_xlabel('x1')
ax[0,1].set_ylabel('x2')
ax[0,1].set_title(f'States, with Safe Exploration')

# Plot actions
ax[1,0].scatter(range(len(actions_1)), actions_1, c=np.linspace(0, 1, len(actions_1)))
ax[1,0].set_xlabel('Timesteps')
ax[1,0].set_ylabel('Actions')
ax[1,0].set_title(f'Actions, without Safe Exploration')

ax[1,1].scatter(range(len(actions_2)), actions_2, c=np.linspace(0, 1, len(actions_2)))
ax[1,1].set_xlabel('Timesteps')
ax[1,1].set_ylabel('Actions')
ax[1,1].set_title(f'Actions, with Safe Exploration')

# Plot rewards
H = 1000
t = np.linspace(H, n_iter, n_iter - H)

ax[2,0].plot(t[n_iter - len(rewards_1):], window_convolution(rewards_1, H))
ax[2,0].set_xlabel("Timesteps")
ax[2,0].set_ylabel(f"Average of the last {H} rewards")
ax[2,0].set_title(f"Average rewards, without Safe Exploration")

ax[2,1].plot(t[n_iter - len(rewards_2):], window_convolution(rewards_2, H))
ax[2,1].set_xlabel("Timesteps")
ax[2,1].set_ylabel(f"Average of the last {H} rewards")
ax[2,1].set_title(f"Average rewards, without Safe Exploration")

plt.suptitle(f"Easy parameterized LQR (theta_real={theta_real}, theta_sim={theta_sim})\nCACLA (gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma})")
plt.savefig(f"results/cacla/Safe_LQR/2_theta_real={theta_real}_theta_sim={theta_sim}_gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}_rewards.png")
plt.show()
plt.close()
