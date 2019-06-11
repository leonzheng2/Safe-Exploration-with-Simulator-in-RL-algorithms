import numpy as np
import matplotlib.pyplot as plt
import ray
from dataclasses import dataclass
from gym.envs.swimmer.remy_swimmer_env import SwimmerEnv


@dataclass
class EnvParam:
  # Environment parameters
  name: str
  n: int
  H: int
  l_i: float
  m_i: float
  h: float


@dataclass
class ARSParam:
  # Agent parameters
  V1: bool
  n_iter: int
  H: int
  N: int
  b: int
  alpha: float
  nu: float
  safe: bool
  threshold: float


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


class Estimator:

  @staticmethod
  def estimate_real_env_param(database, real_env):
    # TODO
    # print(len(database.trajectories))
    # print(len(database.policies))
    env_param = real_env
    return env_param


# @ray.remote
class Environment():

  def __init__(self, env_param):
    self.env_param = env_param
    self.env = SwimmerEnv(envName=env_param.name, n=env_param.n,
                          l_i=env_param.l_i, m_i=env_param.m_i, h=env_param.h)

  def select_action(self, policy, observation, covariance=None, mean=None):
    """
    Compute the action vector, using linear policy

    :param policy: matrix
    :param observation: vector
    :return: vector
    """
    observation = np.array(observation)
    if covariance is None or mean is None:  # ARS V1
      return np.matmul(policy, observation)

    # Else, ARS V2
    diag_covariance = np.diag(covariance) ** (-1 / 2)
    policy = np.matmul(policy, np.diag(diag_covariance))
    action = np.matmul(policy, observation - mean)
    return action

  def rollout(self, policy, covariance=None, mean=None):
    """
    Doing self.agent_param.H steps following the given policy,
    and return the final total reward.

    :param policy: matrix
    :return: float
    """
    total_reward = 0
    observation = self.env.reset()
    saved_states = []
    for t in range(self.env_param.H):
      # self.env.render()
      action = self.select_action(policy, observation, covariance=covariance,
                                  mean=mean)
      observation, reward, done, info = self.env.step(action)
      saved_states.append(observation)
      total_reward += reward
      if done:
        return total_reward
    return total_reward, saved_states

  def close(self):
    self.env.close()


@ray.remote
class ARSAgent():

  def __init__(self, real_env_param, agent_param, data_path=None,
               seed=None):
    # Environment
    self.real_env_param = real_env_param
    self.real_world = Environment(real_env_param)

    # Database
    assert(((data_path is None) and (not agent_param.safe)) or
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
      x_tilde = Estimator.estimate_real_env_param(self.database,
                                                  self.real_env_param)

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
        reward_1, _ = simulator.rollout(policy_1)
        if reward_1 <= self.agent_param.threshold:
          do_real_rollout = False
        else:
          reward_2, _ = simulator.rollout(policy_2)
          if reward_2 <= self.agent_param.threshold:
            do_real_rollout = False

      if do_real_rollout:
        # TODO: MODIFY HERE FOR PARALLEL IMPLEMENTATION
        for policy in [policy_1, policy_2]:
          reward, saved_states = \
            self.real_world.rollout(policy, covariance=self.covariance,
                                    mean=self.mean)
          assert((reward > self.agent_param.threshold and
                  self.agent_param.safe) or
                 (reward <= self.agent_param.threshold and
                  not self.agent_param.safe))
          rewards.append(reward), f"Obtained in real world rollout a " \
            f"return of {reward}, below the " \
            f"threshold {self.agent_param.threshold}"
          if not self.agent_param.V1:
            self.saved_states += saved_states
          self.database.add_trajectory(saved_states, policy)

    if len(rewards) > 0:
      # print(rewards)
      order = self.sort_directions(deltas, rewards)
      self.update_policy(deltas, rewards, order)

      if self.agent_param.V1 is not True:
        states_array = np.array(self.saved_states)
        self.mean = np.mean(states_array, axis=0)
        self.covariance = np.cov(states_array.T)
        # print(f"mean = {self.mean}")
        # print(f"cov = {self.covariance}")

  def runTraining(self, save_data_path=None):
    """
    Run the training. After each iteration, evaluate the current policy by
    doing one rollout. Save the obtained reward after each iteration.
    :return: array of float. Rewards obtained after each iteration.
    """
    rewards = []
    for j in range(self.agent_param.n_iter):
      self.runOneIteration()
      r, _ = self.real_world.rollout(self.policy)
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
    self.real_world.close()
    return np.array(rewards)


class Experiment():

  def __init__(self, real_env_param, results_path="results/gym/",
               data_path=None, save_data_path=None):
    """
    Constructor setting up parameters for the experience
    :param env_param: EnvParam
    """
    # Set up environment
    self.real_env_param = real_env_param
    self.results_path = results_path

    # Data
    self.data_path = data_path
    self.save_data_path = save_data_path

  # @ray.remote
  def plot(self, n_seed, agent_param):
    """
    Plotting learning curve
    :param n_seed: number of seeds for plotting the curve, int
    :param agent_param: ARSParam
    :return: void
    """
    ARS = f"ARS_{'V1' if agent_param.V1 else 'V2'}" \
      f"{'-t' if agent_param.b < agent_param.N else ''}, " \
      f"n_directions={agent_param.N}, " \
      f"deltas_used={agent_param.b}, " \
      f"step_size={agent_param.alpha}, " \
      f"delta_std={agent_param.nu}"
    environment = f"{self.real_env_param.name}, " \
      f"n_segments={self.real_env_param.n}, " \
      f"m_i={round(self.real_env_param.m_i, 2)}, " \
      f"l_i={round(self.real_env_param.l_i, 2)}, " \
      f"deltaT={self.real_env_param.h}"

    print(f"\n------ {environment} ------")
    print(ARS + '\n')

    # Seeds
    r_graphs = []
    for i in range(n_seed):
      agent = ARSAgent.remote(self.real_env_param, agent_param,
                              seed=i, data_path=self.data_path)
      r_graphs.append(agent.runTraining.remote(save_data_path=
                                               self.save_data_path))
    r_graphs = np.array(ray.get(r_graphs))

    # Plot graphs
    plt.figure(figsize=(10, 8))
    for rewards in r_graphs:
      plt.plot(rewards)
    plt.title(f"------ {environment} ------\n{ARS}")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    np.save(f"{self.results_path}array/"
            f"{environment.replace(', ', '-')}-"
            f"{ARS.replace(', ', '-')}", r_graphs)
    plt.savefig(f"{self.results_path}new/"
                f"{environment.replace(', ', '-')}-"
                f"{ARS.replace(', ', '-')}.png")
    # plt.show()
    plt.close()

    # Plot mean and std
    plt.figure(figsize=(10, 8))
    x = np.linspace(0, agent_param.n_iter - 1, agent_param.n_iter)
    mean = np.mean(r_graphs, axis=0)
    std = np.std(r_graphs, axis=0)
    plt.plot(x, mean, 'k', color='#CC4F1B')
    plt.fill_between(x, mean - std, mean + std, alpha=0.5,
                     edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title(f"------ {environment} ------\n{ARS}")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.savefig(f"{self.results_path}new/"
                f"{environment.replace(', ', '-')}-"
                f"{ARS.replace(', ', '-')}-average.png")
    # plt.show()
    plt.close()


if __name__ == '__main__':
  ray.init(num_cpus=4)

  real_env_param = EnvParam(f'LeonSwimmer', n=3, H=1000, l_i=1., m_i=1.,
                            h=1e-3)
  agent_param = ARSParam(V1=False, n_iter=20, H=1000, N=1, b=1,
                         alpha=0.0075, nu=0.01, safe=True, threshold=2)
  exp = Experiment(real_env_param,
                   data_path="src/openai/real_world.npz",
                   save_data_path="src/openai/real_world")
  exp.plot(n_seed=1, agent_param=agent_param)
