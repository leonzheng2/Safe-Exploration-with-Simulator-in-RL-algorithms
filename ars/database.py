"""
Class for manipulating trajectories. Save, load, add trajectories.
"""

import numpy as np


class Database():
  # TODO add a limit capacity if it is too much

  def __init__(self):
    self.policies = []
    self.trajectories = []
    self.size = 0

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
    self.size += 1

  def save(self, path):
    np.savez(path, policies=self.policies, trajectories=self.trajectories)


def pick_sub_database(data_path, size, sub_data_path):
  data = Database()
  data.load(data_path)

  sub_data = Database()
  selected = np.random.randint(0, data.size, size)
  for i in selected:
    sub_data.add_trajectory(data.trajectories[i], data.policies[i])
  sub_data.save(sub_data_path)

if __name__ == '__main__':
  size = 10
  pick_sub_database("src/ars/real_world.npz", size, f"src/ars/real_world_sub_{size}")
