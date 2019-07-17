"""
Script for transfering the policy learned in LeonSwimmer to MuJoCo Swimmer.
"""


import numpy as np
import gym
from ars.experiment import Experiment
from ars.parameters import EnvParam, ARSParam

H = 1000

# Learning the Swimmer task on a simulator
sim_param = EnvParam("LeonSwimmer", n=3, H=H, l_i=1.0, m_i=1.0, h=0.001, k=10.0, epsilon=0)
agent_param = ARSParam("ARS", V1=True, n_iter=100, H=H, N=1, b=1, alpha=0.01, nu=0.01, safe=False, threshold=0, initial_w='Zero')
exp = Experiment(sim_param, save_policy_path="ars/data/saved_sim_policy")
exp.plot(1, agent_param)

# Transfer policy
policy = np.load("src/ars/data/saved_sim_policy.npz")

# Use the learned policy for MuJoCo Swimmer
real_env = gym.make('Swimmer-v2')
o = real_env.reset()
R = 0
for i in range(H):
  real_env.render()
  a = np.matmul(policy, o)
  o, r, done, info = real_env.step(a)
  R += r
  if done:
    break
print(f"Return: {R}")