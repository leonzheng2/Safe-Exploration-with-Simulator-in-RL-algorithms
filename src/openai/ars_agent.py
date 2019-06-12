import ray
import numpy as np
from src.openai.environment import Environment
from src.openai.database import Database
from src.openai.estimator import Estimator


@ray.remote
class ARSAgent():

  def __init__(self, real_env_param, agent_param, data_path=None,
               seed=None):
    # Environment
    self.real_env_param = real_env_param
    self.real_world = Environment(real_env_param)

    # Database
    assert (((data_path is None) and (not agent_param.safe)) or
            ((data_path is not None) and (agent_param.safe))), \
      "Please provide a dataset if using safe ARS, and don't provide if not."
    self.database = Database()
    if data_path is not None:
      self.database.load(data_path)

    # Agent linear policy
    self.policy = np.zeros((self.real_world.env.action_space.shape[0],
                            self.real_world.env.observation_space.shape[0]))

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
    if self.agent_param.safe:  # Safe ARS - Estimation
      estimator = Estimator(self.real_world.env_param, self.database)
      x_tilde = estimator.estimate_real_env_param()

    deltas = [2 * np.random.rand(*self.policy.shape) -
              1 for i in range(self.agent_param.N)]
    rewards = []
    for i in range(self.agent_param.N):
      policy_1 = self.policy + self.agent_param.nu * deltas[i]
      policy_2 = self.policy - self.agent_param.nu * deltas[i]

      # Safe ARS - Safe exploration
      do_real_rollout = True
      if self.agent_param.safe:
        simulator = Environment(x_tilde)
        reward_1, _ = simulator.rollout(policy_1, covariance=self.covariance,
                                        mean=self.mean)
        if reward_1 <= self.agent_param.threshold:
          do_real_rollout = False
        else:
          reward_2, _ = simulator.rollout(policy_2,
                                          covariance=self.covariance,
                                          mean=self.mean)
          if reward_2 <= self.agent_param.threshold:
            do_real_rollout = False

      if do_real_rollout:
        # TODO: MODIFY HERE FOR PARALLEL IMPLEMENTATION
        for policy in [policy_1, policy_2]:
          reward, saved_states = \
            self.real_world.rollout(policy, covariance=self.covariance,
                                    mean=self.mean)
          assert ((reward > self.agent_param.threshold and
                   self.agent_param.safe) or
                  (reward <= self.agent_param.threshold and
                   not self.agent_param.safe)), \
            f"Obtained in real world rollout a " \
              f"return of {reward}, below the " \
              f"threshold {self.agent_param.threshold}"
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

  def runTraining(self, save_data_path=None):
    """
    Run the training. After each iteration, evaluate the current policy by
    doing one rollout. Save the obtained reward after each iteration.
    :return: array of float. Rewards obtained after each iteration.
    """
    # Initialization
    # TODO Weights initialization
    rewards = [0]

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
    return np.array(rewards)
