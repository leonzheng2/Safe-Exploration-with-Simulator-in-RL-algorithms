"""
(Description)
"""


from safe_ars.ars import Basic_ARS, Safe_ARS
from envs.gym_swimmer.swimmer.remy_swimmer_env import SwimmerEnv
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ARS paramet
parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", help="precision of parameter estimation", type=float)
parser.add_argument("--thresh", help="safety threshold: the state cost should never be higher than this threshold", type=float)
parser.add_argument("--n_iter", help="number of ARS training iterations", type=int)
parser.add_argument("--n_rollout", help="length of one rollout for ARS training", type=int)
parser.add_argument("--N", help="number of policy perturbations sampled", type=int)
parser.add_argument("--b", help="number of pertubations used for policy update", type=int)
parser.add_argument("--alpha", help="step size", type=float)
parser.add_argument("--nu", help="perturbations standard deviation", type=float)
parser.add_argument("--path", help="directory for saving the results graph", type=str)
args = parser.parse_args()

# Seed
seed = np.random.randint(2**32-1)

# Number of segments
n = 3

# Real world environment
print("Creating environments...")
theta_real = [1., 1., 10.]
print(f"Real world parameter: {theta_real}")
real_env = SwimmerEnv("RealWorld", n=3, m_i=theta_real[0], l_i=theta_real[1], k=theta_real[2])

# Simulator environment
delta = np.random.rand(len(theta_real))
theta_sim = theta_real + delta / np.linalg.norm(delta, ord=2) * args.epsilon
print(f"Estimated parameter: {theta_sim}")
sim_env = SwimmerEnv("Simulator", n=3, m_i=theta_sim[0], l_i=theta_sim[1], k=theta_sim[2])
print("Done!\n")

# Cost: maximum speed angle
# obs = [Gdot_x, Gdot_y, theta_1, theta_dot_1, ..., theta_n, theta_dot_n]
cost = lambda x: np.max([abs(x[3+2*i]) for i in range(n)])

# Set Basic ARS
unsafe_agent = Basic_ARS()

# Set Safe ARS
sim_thresh = args.thresh - 1
safe_agent = Safe_ARS(cost, args.thresh, sim_thresh, sim_env)

# Train
print("Training basic ARS for Swimmer...")
np.random.seed(seed)
unsafe_returns, unsafe_states = unsafe_agent.train(args.n_iter, real_env, args.N, args.b, args.alpha, args.nu, args.n_rollout)
print("Done!\n")

print("Training ARS for Swimmer with Safe Exploration...")
np.random.seed(seed)
safe_returns, safe_states = safe_agent.train(args.n_iter, real_env, args.N, args.b, args.alpha, args.nu, args.n_rollout)
print("Done!\n")

# Plot
print(f"Length of returns: {safe_returns.shape}")
print(f"Length of states: {safe_states.shape}")
print("Plotting results...")
plt.close('all')
fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharey='row')

# Plot costs
c_timesteps = np.linspace(0, 2*args.n_iter*args.n_rollout, 2*args.n_iter*args.n_rollout)
# Unsafe
unsafe_costs = [cost(unsafe_states[i,j]) for i in range(2*args.n_iter) for j in range(args.n_rollout)]
ax[0, 0].plot(c_timesteps, unsafe_costs)
ax[0, 0].plot(c_timesteps, [args.thresh] * len(c_timesteps), color='k')
ax[0, 0].set_ylabel("Cost")
ax[0, 0].set_title("State cost, without Safe Exploration")
# Safe
safe_costs = [cost(safe_states[i,j]) for i in range(2*args.n_iter) for j in range(args.n_rollout)]
ax[0, 1].plot(c_timesteps, safe_costs)
ax[0, 1].plot(c_timesteps, [args.thresh] * len(c_timesteps), color='k')
ax[0, 1].set_title("State cost, with Safe Exploration")

# Plot returns
r_timesteps = np.linspace(0, 2*args.n_iter*args.n_rollout, args.n_iter)
# Unsafe
ax[1, 0].plot(r_timesteps, unsafe_returns)
ax[1, 0].set_ylabel("Returns (mean of each iteration)")
ax[1, 0].set_xlabel("Timesteps")
ax[1, 0].set_title("Return, without Safe Exploration")
# Safe
ax[1, 1].plot(r_timesteps, safe_returns)
ax[1, 1].set_xlabel("Timesteps")
ax[1, 1].set_title("Return, with Safe Exploration")

save_path = f"{args.path}safe_ars_swimmer_epsilon={args.epsilon}_thresh={args.thresh}.png"
plt.savefig(save_path)
plt.show()
print(f"Done! Graph saved at {save_path}")
