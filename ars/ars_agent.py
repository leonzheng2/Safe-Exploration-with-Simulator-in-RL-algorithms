"""
Implementation of Augmented Random Search
With or without Safe Exploration
The constraint considered here is that the return of a real world rollout should be larger than a fixed value.
"""


import ray
import numpy as np
from ars.environment import Environment
from ars.database import Database
from ars.estimator import Estimator


@ray.remote
class ARSAgent():
  """
  ARS Agent for solving RL tasks. With or without Safe Exploration.
  """

  def __init__(self, real_env_param, agent_param, data_path=None,
               seed=None, guess_param=None, approx_error=None, sim_thresh=None):
    """
    Constructor with all parameters.
    :param real_env_param: Swimmer environment parameters for the real world
    :param agent_param: ARS parameters
    :param data_path: path for loading the real world trajectories in order to use environment parameters estimation
    :param seed: random seed
    :param guess_param: approximate real world parameters
    :param approx_error: approximation error
    :param sim_thresh: choose a given simulator threshold
    """

    # Environment
    self.real_env_param = real_env_param
    self.real_world = Environment(real_env_param)

    # Database
    self.database = Database()
    if agent_param.safe:
      self.database.load(data_path)

      # Estimator
      if guess_param is not None and data_path is not None:
        print("Using computed estimation...")
        self.estimator = Estimator(self.database, guess_param, capacity=1)
        self.estimated_param = self.estimator.estimate_real_env_param()
      else:
        if approx_error is not None:
          print("Using approximated estimation...")
          unknowns = ('m_i', 'l_i', 'k')
          delta = np.random.rand(len(unknowns))
          delta = delta / np.linalg.norm(delta, ord=2) * approx_error
          self.estimated_param = self.real_env_param
          self.estimated_param.name = 'LeonSwimmer-Simulator'
          self.estimated_param.m_i += delta[0]
          self.estimated_param.l_i += delta[1]
          self.estimated_param.k += delta[2]
        else:
          print("Using exact estimation...")
          self.estimated_param = self.real_env_param
      print(f"Used estimation: {self.estimated_param}")

      # Set simulation threshold
      if sim_thresh is not None:
        epsilon = real_env_param.epsilon
        alpha = sim_thresh.compute_alpha(agent_param.H)
        self.sim_threshold = agent_param.threshold + alpha*epsilon
        print(f"Simulator threshold is {self.sim_threshold}")
      else: # TODO compute sim_threshold
        ...

    # Agent linear policy
    if agent_param.initial_w == 'Zero':
      self.policy = np.zeros((self.real_world.env.action_space.shape[0],
                              self.real_world.env.observation_space.shape[0]))
    else:
      self.policy = np.load(agent_param.initial_w)
      assert self.policy.shape == (self.real_world.env.action_space.shape[0],
                                   self.real_world.env.observation_space.shape[
                                     0])

    # Agent parameters
    self.agent_param = agent_param

    # V2
    self.mean = None if self.agent_param.V1 else \
      np.zeros(self.real_world.env.observation_space.shape[0])
    self.covariance = None if self.agent_param.V1 else \
      np.identity(self.real_world.env.observation_space.shape[0])
    self.saved_states = []

    # Randomness
    self.n_seed = seed
    np.random.seed(self.n_seed)

  def sort_directions(self, deltas, rewards):
    """
    Sort the directions deltas by max{r_k_+, r_k_-}

    :param deltas: array of matrices
    :param rewards: array of float
    :return: bijection of range(len(deltas))
    """
    max_rewards = [max(rewards[2 * i], rewards[2 * i + 1])
                   for i in range(len(deltas))]
    indices = np.argsort(max_rewards).tolist()
    return indices[::-1]

  def update_policy(self, deltas, rewards, order):
    """
    Update the linear policy following the update step,
    after collecting the rewards

    :param deltas: array of matrices
    :param rewards: array of floats
    :param order: bijection of range(len(deltas))
    :return: void, self.policy is updated
    """
    used_rewards = []
    for i in order:
      used_rewards += [rewards[2 * i], rewards[2 * i + 1]]
    sigma_r = np.std(used_rewards)

    grad = np.zeros(self.policy.shape)
    for i in order:
      grad += (rewards[2 * i] - rewards[2 * i + 1]) * deltas[i]
    grad /= (self.agent_param.b * sigma_r)

    self.policy += self.agent_param.alpha * grad

  def runOneIteration(self):
    """
    Performing one whole iteration of the ARS algorithm
    :return: void, self.policy is updated
    """
    deltas = [2 * np.random.rand(*self.policy.shape) -
              1 for i in range(self.agent_param.N)]
    rewards = []
    for i in range(self.agent_param.N):
      policy_1 = self.policy + self.agent_param.nu * deltas[i]
      policy_2 = self.policy - self.agent_param.nu * deltas[i]

      # Safe ARS - Safe exploration
      do_real_rollout = True
      if self.agent_param.safe:
        simulator = Environment(self.estimated_param)
        reward_1, _ = simulator.rollout(policy_1, covariance=self.covariance,
                                        mean=self.mean)
        if reward_1 <= self.sim_threshold:
          do_real_rollout = False
        else:
          reward_2, _ = simulator.rollout(policy_2,
                                          covariance=self.covariance,
                                          mean=self.mean)
          if reward_2 <= self.sim_threshold:
            do_real_rollout = False

      if do_real_rollout:
        # TODO: MODIFY HERE FOR PARALLEL IMPLEMENTATION
        for policy in [policy_1, policy_2]:
          reward, saved_states = \
            self.real_world.rollout(policy, covariance=self.covariance,
                                    mean=self.mean)
          if self.agent_param.safe and reward < self.agent_param.threshold:
            print(f"Obtained in real world rollout a "
                  f"return of {reward}, below the "
                  f"threshold {self.agent_param.threshold}")
          rewards.append(reward)
          if not self.agent_param.V1:
            self.saved_states += saved_states
          self.database.add_trajectory(saved_states, policy)

    if len(rewards) > 0:
      # print(rewards)
      order = self.sort_directions(deltas, rewards)
      self.update_policy(deltas, rewards, order)

      if self.agent_param.V1 is False:
        states_array = np.array(self.saved_states)
        self.mean = np.mean(states_array, axis=0)
        self.covariance = np.cov(states_array.T)
        # print(f"mean = {self.mean}")
        # print(f"cov = {self.covariance}")
    return rewards

  def runTraining(self, save_data_path=None, save_policy_path=None):
    """
    Run the training. After each iteration, evaluate the current policy by
    doing one rollout. Save the obtained reward after each iteration.

    :return: array of float. Rewards obtained after each iteration.
    """
    # Initialization
    rewards = [np.mean(self.runOneIteration())]

    # Training
    for j in range(1, self.agent_param.n_iter + 1):
      all_rewards = self.runOneIteration()
      r = np.mean(all_rewards) if len(all_rewards) > 0 else rewards[-1]
      rewards.append(r)
      if j % 10 == 0:
        print(f"Seed {self.n_seed} ------ V1 = {self.agent_param.V1}; "
              f"n={self.real_env_param.n}; "
              f"h={self.real_env_param.h}; "
              f"alpha={self.agent_param.alpha}; "
              f"nu={self.agent_param.nu}; "
              f"N={self.agent_param.N}; "
              f"b={self.agent_param.b}; "
              f"m_i={self.real_env_param.m_i}; "
              f"l_i={self.real_env_param.l_i} "
              f"------ Iteration {j}/{self.agent_param.n_iter}: {r}")
        if save_data_path is not None:
          self.database.save(save_data_path)

    # End of the training
    self.real_world.close()
    if save_policy_path is not None:
      np.save(save_policy_path, self.policy)
    return np.array(rewards)
