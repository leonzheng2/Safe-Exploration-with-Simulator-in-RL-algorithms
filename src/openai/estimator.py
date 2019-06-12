import numpy as np
import dataclasses
from src.openai.database import Database
from src.openai.environment import Environment
from src.openai.parameters import EnvParam

class Estimator:

  def __init__(self, database, real_param, unknowns=('m_i', 'l_i', 'k')):
    assert database.size > 0, "Database is empty"
    assert len(database.trajectories[0]) == real_param.H, "Rollouts are not the same"
    self.real_param = real_param
    self.unknowns = unknowns
    self.database = database

  def J(self, x):
    sum = 0
    for k in range(self.database.size):
      print(k)
      policy = self.database.policies[k]
      real_states = np.array(self.database.trajectories[k])

      sim_param = self.convert_to_env_param(x)
      sim_env = Environment(sim_param)
      _, sim_states = sim_env.rollout(policy) # /!\ /!\ /!\ Real world database should include V1 type interaction
      sim_states = np.array(sim_states)

      dist = np.linalg.norm(np.linalg.norm(sim_states-real_states, ord=2, axis=1), ord=2)
      sum += dist

    return sum

  def estimate_real_env_param(self):
    # TODO
    print(self.database.size)

    # TODO better initialization of x
    x = np.zeros(len(self.unknowns)) # Estimate of the unknown parameters

    x[0] = 1.0
    x[1] = 1.0
    x[2] = 10.

    print(self.J(x))

    env_param = self.convert_to_env_param(x)
    print(env_param)
    return env_param

  def convert_to_env_param(self, x):
    dict = dataclasses.asdict(self.real_param)
    for i in range(len(self.unknowns)):
      dict[self.unknowns[i]] = x[i]
    return EnvParam(**dict)

if __name__ == '__main__':
  real_world_param = EnvParam(name="Real world", n=3, H=1000, m_i=1.0, l_i=1.0, h=0.001, k=10.)
  database = Database()
  database.load("src/openai/real_world_sub_10.npz")
  estimator = Estimator(database, real_world_param)
  estimate = estimator.estimate_real_env_param()