import numpy as np
import dataclasses
from src.openai.database import Database
from src.openai.environment import Environment
from src.openai.parameters import EnvParam
import cma


class Estimator:

  def __init__(self, database, guess_param, unknowns=('m_i', 'l_i', 'k'),
               capacity=2):
    assert database.size > 0, "Database is empty"
    assert len(
      database.trajectories[0]) == guess_param.H, "Rollouts are not the same"
    self.guess_param = guess_param
    self.unknowns = unknowns
    self.database = database
    self.subset = np.random.randint(0, self.database.size, capacity)
    self.iter = 0

  def J(self, x):
    distances = []
    for k in self.subset:
      policy = self.database.policies[k]
      real_states = np.array(self.database.trajectories[k])

      sim_param = self.convert_to_env_param(x)
      sim_env = Environment(sim_param)
      _, sim_states = sim_env.rollout(
        policy)  # /!\ Database should include only V1 type interaction
      sim_states = np.array(sim_states)

      dist = np.linalg.norm(
        np.linalg.norm(sim_states - real_states, ord=2, axis=1), ord=2) / len(
        real_states)

      distances.append(dist)
    s = np.mean(distances)
    return s

  def estimate_real_env_param(self):
    print(f"------ Estimating the real world environment parameters ------")
    assert self.guess_param is not None, "No initial guess for real world parameters (no initialization for optimization problem)"
    print("Extracting the initial guess of real world parameters...")
    d = dataclasses.asdict(self.guess_param)
    x0 = np.zeros(len(self.unknowns))
    for i in range(len(x0)):
      x0[i] = d[self.unknowns[i]]
    print(f"Initial estimation extracted: {x0}")
    print("Starting estimation...")
    es = cma.CMAEvolutionStrategy(x0, 1).optimize(self.J)
    res = es.result
    print(res)
    est_x, _, _ = es.best.get()
    print("Estimation finished")
    env_param = self.convert_to_env_param(est_x)
    print(f"Estimated parameters: {env_param}")
    return env_param

  def convert_to_env_param(self, x):
    dict = dataclasses.asdict(self.guess_param)
    for i in range(len(self.unknowns)):
      dict[self.unknowns[i]] = x[i]
    return EnvParam(**dict)


if __name__ == '__main__':
  guess_param = EnvParam(name="Simulator with estimation", n=3, H=1000,
                         m_i=1.2,
                         l_i=.8, h=0.001, k=10.2)
  database = Database()
  database.load("src/openai/real_world.npz")
  estimator = Estimator(database, guess_param)

  estimate = estimator.estimate_real_env_param()
