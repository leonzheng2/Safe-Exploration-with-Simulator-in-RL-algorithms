import numpy as np


class Database():

  def __init__(self):
    self.policies = []
    self.trajectories = []

  def load(self, path):
    # Load trajectories
    npzfile = np.load(path)
    assert ('policies' in npzfile.files and 'trajectories'), \
      "The file loaded doesn't contain " \
      "the array 'policies' and 'trajectories'"
    policies = npzfile['policies']
    trajectories = npzfile['trajectories']
    assert (len(policies) == len(trajectories)), \
      "'policies' and 'trajectories' doesn't have the same length"

    self.real_traj = []
    for policy, trajectory in zip(policies, trajectories):
      self.add_trajectory(trajectory, policy)

  def add_trajectory(self, trajectory, policy):
    self.trajectories.append(trajectory)
    self.policies.append(policy)

  def save(self, path):
    np.savez(path, policies=self.policies, trajectories=self.trajectories)
