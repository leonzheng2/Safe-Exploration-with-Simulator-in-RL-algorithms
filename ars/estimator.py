"""
Estimate the real world parameter using simulations.
Two objective functions to optimize are possible. I(x) has better results.
Algorithm used to solve the optimization problem: CMA-ES.
"""


import numpy as np
import dataclasses
from ars.database import Database
from ars.environment import Environment
from ars.parameters import EnvParam
import cma
import time


class Estimator:

  def __init__(self, database, guess_param, capacity, unknowns=('m_i', 'l_i', 'k')):
    """
    Constructor.
    :param database: Database of trajectories
    :param guess_param: initilization for the algorithm
    :param capacity: number of trajectories included in the objective function
    :param unknowns: parameters names
    """
    assert database.size > 0, "Database is empty"
    assert len(
      database.trajectories[0]) == guess_param.H, "Rollouts are not the same"
    self.guess_param = guess_param
    self.unknowns = unknowns
    self.database = database
    self.subset = np.random.randint(0, self.database.size, capacity)
    self.iter = 0

  def I(self, x):
    """
    Objective function. Faster and more stable.
    :param x: vector
    :return: real value
    """
    distances = []
    for k in self.subset:
      policy = self.database.policies[k]
      trajectory = np.array(self.database.trajectories[k])

      sim_param = self.convert_to_env_param(x)
      sim_env = Environment(sim_param)

      sim_states = []
      for s in trajectory[:-1]:
        ac = sim_env.select_action(policy, s)
        sim_env.env.set_state(s)
        next_s, _, _, _ = sim_env.env.step(ac)
        sim_states.append(next_s)
      sim_states = np.array(sim_states)

      real_states = trajectory[1:]

      dist = np.sum(np.linalg.norm(sim_states - real_states, ord=2, axis=1))
      distances.append(dist)
    return np.sum(distances)

  def J(self, x):
    """
    Deprecated objective function.
    :param x: vector
    :return: real value
    """
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
    """
    Apply CMA-ES to obtain the minimum of the objective function.
    :return: vector, estimation of real world parameters
    """
    print(f"------ Estimating the real world environment parameters ------")
    assert self.guess_param is not None, "No initial guess for real world parameters (no initialization for optimization problem)"
    print("Extracting the initial guess of real world parameters...")
    d = dataclasses.asdict(self.guess_param)
    x0 = np.zeros(len(self.unknowns))
    for i in range(len(x0)):
      x0[i] = d[self.unknowns[i]]
    print(f"Initial estimation extracted: {x0}")
    print("Starting estimation...")
    es = cma.CMAEvolutionStrategy(x0, 1).optimize(self.I)
    res = es.result
    print(res)
    est_x, _, _ = es.best.get()
    print("Estimation finished")
    env_param = self.convert_to_env_param(est_x)
    print(f"Estimated parameters: {env_param}")
    return env_param

  def convert_to_env_param(self, x):
    """
    Helper function for manipulating environment parameters object.
    :param x: vector
    :return: EnvParam
    """
    dict = dataclasses.asdict(self.guess_param)
    for i in range(len(self.unknowns)):
      dict[self.unknowns[i]] = x[i]
    return EnvParam(**dict)


if __name__ == '__main__':
  guess_param = EnvParam(name="Simulator with estimation", n=3, H=1000,
                         m_i=1.01,
                         l_i=1.01, h=0.001, k=10.01, epsilon=0.01)
  database = Database()
  database.load("ars/real_world.npz")
  estimator = Estimator(database, guess_param, capacity=1)

  print("Counting time to compute objective function evaluated at real world param...")
  start_t = time.time()
  x = [1.0, 1.0, 10.0]
  value = estimator.I(x)
  assert value == 0.0, f"Minimum isn't achieved, having {value} instead of 0.0"
  print(f"Compute distance requires {time.time() - start_t} seconds")

  estimate = estimator.estimate_real_env_param()
