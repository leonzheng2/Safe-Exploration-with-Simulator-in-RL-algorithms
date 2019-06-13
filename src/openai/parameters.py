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
