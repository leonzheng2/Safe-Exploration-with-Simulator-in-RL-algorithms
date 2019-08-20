"""
Script for showing the validity of Safe Exploration on CACLA solving easy parameterized LQR.

Choose the random seed:
    - `seed` is either a given integer or `numpy.random.randint(2**32)`

Choose the class for the environments `lqr_real` and `lqr_sim`:
    - `EasyParamLinearQuadReg`
    - `BoundedEasyLinearQuadReg`
    - `BoundedActionEasyLinearQuadReg`
    - or `EasyAffineQuadReg`
    - and the environment parameters: `theta_real` for the unknown real world parameter, and `theta_sim` for the estimated real world parameter

Choose the correspondant CACLA agent for `safe_agent`:
    - when environments are `EasyParamLinearQuadReg`, use `CACLA_LQR_SE_agent`
    - when environments are `BoundedEasyLinearQuadReg` or `BoundedActionEasyLinearQuadReg`, use `CACLA_Bounded_LQR_SE_agent`
    - when environments are `EasyAffineQuadReg`, use `CACLA_AffineQR_SE_agent`

Choose the CACLA agent parameters:
    - discount factor `gamma`
    - standard deviation for Gaussian policy `sigma`
    - step size for weights update `alpha`
    - number of iterations `n_iter`

Choose the safety state constraints which has to be satisfied by the real world agent:
    - cost function `cost`
    - Lipschitz constant of cost function `L_c`
    - threshold `l`

Choose the path and the names for saving the figures.
"""
print("Starting experience of Safe Exploration with CACLA on LQR")

import numpy as np
import matplotlib.pyplot as plt
from envs.gym_lqr.lqr_env import EasyParamLinearQuadReg, BoundedEasyLinearQuadReg, BoundedActionEasyLinearQuadReg, EasyAffineQuadReg
from cacla.cacla_agent import CACLA_LQR_agent
from cacla.cacla_safe_agent import Constraint, CACLA_LQR_SE_agent, CACLA_Bounded_LQR_SE_agent, CACLA_AffineQR_SE_agent
from cacla.window import window_convolution

### Random seeds
seed = 8943948
# seed = np.random.randint(2**32)

### LQR Real World
theta_real = 1.0
lqr_real = EasyAffineQuadReg(theta_real)

### LQR Simulator
theta_sim = 0.99
lqr_sim = EasyAffineQuadReg(theta_sim)

### Agent
n_iter = 20000
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
l = 4
constraint = Constraint(cost, l, L_c)
safe_agent = CACLA_AffineQR_SE_agent(lqr_real, lqr_sim, epsilon, constraint)
states_2, actions_2, rewards_2 = safe_agent.run(n_iter, gamma, alpha, sigma)
print(safe_agent.F)

### Policy
opt_F = np.array([1-np.sqrt(3), 0])
print(f"Optimal: {opt_F}")
print(f"CACLA without Safe Exploration: {agent.F}; distance = {np.linalg.norm(agent.F - opt_F, 2)}")
print(f"CACLA with Safe Exploration: {safe_agent.F}; distance = {np.linalg.norm(safe_agent.F - opt_F, 2)}")

### Results - Comparison

plt.close('all')
fig, ax = plt.subplots(3, 2, figsize=(10, 10))
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
ax[2,1].set_title(f"Average rewards, with Safe Exploration")

plt.suptitle(f"Easy parameterized LQR (theta_real={theta_real}, theta_sim={theta_sim})\nCACLA (gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma})")
plt.savefig(f"results/cacla/Safe_LQR/Problem-C/1_theta_real={theta_real}_theta_sim={theta_sim}_gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}_rewards.png")
# plt.show()
plt.close()
