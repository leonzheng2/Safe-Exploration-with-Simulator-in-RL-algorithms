"""
Script for Safe Exploration, ARS on Swimmer task
Set real world parameters at variable `real_env_param`.
Set the initial guess of the real world parameters at variable `guess_param`
Set the ARS parameters of the first agent `hand_agent`, which doesn't use Safe Exploration.
We save the obtained policy.
Then we initialize `real_agent` to use this transfered policy and use Safe exploration. Set its parameters at variable `real_agent`.
"""


import numpy as np
import matplotlib.pyplot as plt
import ray
from ars.parameters import EnvParam, ARSParam, Threshold
from ars.experiment import Experiment

ray.init(num_cpus=8)

guess_param = EnvParam('LeonSwimmer-Simulator', n=3, H=1000, l_i=1.,
                       m_i=1.,
                       h=1e-3, k=10., epsilon=0.01)
real_env_param = EnvParam('LeonSwimmer-RealWorld', n=3, H=1000, l_i=.8,
                          m_i=1.2,
                          h=1e-3, k=10.2, epsilon=0.001)

# Get the initial weight - As if it is the hand controller

hand_agent = ARSParam('HandControl', V1=True, n_iter=200, H=1000, N=1, b=1,
                      alpha=0.0075, nu=0.01, safe=False, threshold=0,
                      initial_w='Zero')
hand_exp = Experiment(real_env_param,
                      data_path=None,
                      save_data_path="src/ars/data/real_world_2.npz",
                      save_policy_path='src/ars/data/saved_hand_policy',
                      guess_param=None)
returns = hand_exp.plot(n_seed=1, agent_param=hand_agent)

# Get safety threshold based on known controller/experience
mean_returns = np.mean(returns, axis=0)
l = mean_returns[-1] * 0.99
print(f"\nSafety threshold: {l}")
np.savetxt("src/ars/data/threshold.txt", np.array([l]))

# l = 285

# Train the real agent in real world. Use transfered weight. Unknown real world parameters.

K = 1
A = 1
B = 0.001
H = 1000

# Use several simulator threshold to find a valid and efficient one.
for A in [0.1, 0.3, 0.5, 0.7]:
  sim_thresh = Threshold(K=K, A=A, B=B)
  alpha = sim_thresh.compute_alpha(H)
  print(f"B = {B}; alpha = {alpha}")
  epsilon_range = np.linspace(0.0001, 0.01, 10)
  sim_thresh_range = [l + alpha * e for e in epsilon_range]
  min_return = []
  max_mean_returns = []

  #
  for epsilon in epsilon_range:
    real_env_param = EnvParam('LeonSwimmer-RealWorld', n=3, H=H, l_i=.8,
                              m_i=1.2,
                              h=1e-3, k=10.2, epsilon=epsilon)
    real_agent = ARSParam(f'RLControl', V1=True, n_iter=400,
                          H=H, N=1, b=1,
                          alpha=0.0075, nu=0.01, safe=True, threshold=l,
                          initial_w='src/ars/data/saved_hand_policy.npy')
    real_exp = Experiment(real_env_param,
                          data_path="src/ars/data/real_world_2.npz",
                          save_data_path=None,
                          save_policy_path=None,
                          guess_param=None,
                          approx_error=epsilon,
                          sim_thresh=sim_thresh)
    r_graphs = real_exp.plot(n_seed=8, agent_param=real_agent)
    min_return.append(np.nanmin(r_graphs))
    mean = np.mean(r_graphs, axis=0)
    max_mean_returns.append(np.nanmax(mean))

  # Plot the results
  plt.figure(figsize=(10, 8))
  plt.plot(epsilon_range, min_return, marker='o', label="Minimum return")
  plt.plot(epsilon_range, max_mean_returns, marker='o',
           label="Max of mean learning curve")
  plt.plot(epsilon_range, sim_thresh_range, linestyle='--', marker='D',
           label="Simulator threshold")
  plt.plot(epsilon_range, [l] * len(epsilon_range), color='black',
           linewidth=2, label="Safety threshold")
  plt.legend()
  plt.xlabel("epsilon")
  plt.ylabel("Average return")
  plt.title(
    f"Safe ARS with approximation error of epsilon, with constants H={H}, K={K}, A={A}, B={B}")
  plt.savefig(f"results/epsilon_sim_threshold_H={H}_K={K}_A={A}_B={B}.png")
  # plt.show()
  plt.close()