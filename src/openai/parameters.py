from dataclasses import dataclass


@dataclass
class EnvParam:
  # Environment parameters
  name: str
  n: int
  H: int
  l_i: float
  m_i: float
  h: float
  k: float
  epsilon: float


@dataclass
class ARSParam:
  # Agent parameters
  name: str
  V1: bool
  n_iter: int
  H: int
  N: int
  b: int
  alpha: float
  nu: float
  safe: bool
  threshold: float
  initial_w: str

@dataclass
class Threshold:
  K: float
  A: float
  B: float

  def compute_alpha(self, H):
    return self.K * self.A / (1 - self.B) * (H - self.B * (1-self.B**H)/(1-self.B))

if __name__ == '__main__':
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