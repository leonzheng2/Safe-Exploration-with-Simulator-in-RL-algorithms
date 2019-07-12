"""
Parameters classes for easier implementation.
Script for plotting alpha depending on B.
"""


from dataclasses import dataclass


@dataclass
class EnvParam:
  # Environment parameters
  name: str
  n: int # number of segments
  H: int # length of rollout
  l_i: float # length of a segment
  m_i: float # mass of a segment
  h: float # time interval for integration
  k: float # viscosity coefficient
  epsilon: float # approximation error


@dataclass
class ARSParam:
  # Agent parameters
  name: str
  V1: bool # Set True if using the V1 version of ARS
  n_iter: int # Number of training iterations
  H: int # Length of rollouts
  N: int # Number of policy perturbations sampled
  b: int # Number of policy perturbations considered to improve policy
  alpha: float # Step size for policy improvements
  nu: float # Standard deviation of policy perturbations
  safe: bool # Set True to use Safe Exploration
  threshold: float # Set the threshold of the safety constraint
  initial_w: str # 'Zero' for initial policy equal to zero. Otherwise, choose path to '.npy' file to load initial policy.

@dataclass
class Threshold:
  K: float # Lipchitz constant of reward function
  A: float # Lipschitz constant of transition function, with respect to parameters
  B: float # Lipschitz constant of transition function, with respect to states

  def compute_alpha(self, H):
    return self.K * self.A / (1 - self.B) * (H - self.B * (1-self.B**H)/(1-self.B))

if __name__ == '__main__':
  """ Plotting alpha depending on B"""

  import numpy as np
  import matplotlib.pyplot as plt
  K = 1
  A = 1
  H = 1000
  B_range = np.linspace(1e-6, 0.9, 1000)
  alphas = []
  for B in B_range:
    t = Threshold(K, A, B)
    alphas.append(t.compute_alpha(H))
  plt.figure(figsize=(10, 8))
  plt.plot(B_range, alphas, label=f"H={H}, K={K}, A={A}")
  plt.legend()
  plt.xlabel("B")
  plt.ylabel("alpha")
  plt.title(f"alpha(H={H}, K={K}, A={A}, B)")
  plt.savefig("results/alpha_analysis")
  plt.show()
  plt.close()