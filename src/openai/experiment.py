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
  select_V1: bool
  n_iter: int
  H: int
  N: int
  b: int
  alpha: float
  nu: float


class Estimator:

  def __init__(self, trajectories_path):
    # Load trajectories
    npzfile = np.load(trajectories_path)
    assert (
          'policies' in npzfile.files and 'trajectories'), "The file loaded doesn't contain the array 'policies' and 'trajectories'"
    policies = npzfile['policies']
    trajectories = npzfile['trajectories']
    assert (len(policies) == len(
      trajectories)), "'policies' and 'trajectories' doesn't have the same length"

    self.real_traj = []
    for policy, trajectory in zip(policies, trajectories):
      self.add_trajectory(trajectory, policy)

  def add_trajectory(self, trajectory, policy):
    self.real_traj.append((trajectory, policy))

  def estimate_real_env_param(self, real_env):
    # TODO
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
    Doing self.H steps following the given policy,
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

  def __init__(self, real_env_param, agent_param, trajectories=None,
               seed=None):
    # Environment
    self.real_env_param = real_env_param
    self.real_world = Environment(real_env_param)

    # Estimator
    self.estimator = None if trajectories is None else Estimator(trajectories)

    # Agent linear policy
    self.policy = np.zeros((self.real_world.env.action_space.shape[0],
                            self.real_world.env.observation_space.shape[0]))
    self.n_it = agent_param.n_iter

    # Agent parameters
    self.N = agent_param.N
    self.b = agent_param.b
    self.alpha = agent_param.alpha
    self.nu = agent_param.nu
    self.H = agent_param.H

    # V2
    self.V1 = agent_param.select_V1
    self.mean = None if self.V1 else np.zeros(
      self.env.observation_space.shape[0])
    self.covariance = None if self.V1 else np.identity(
      self.env.observation_space.shape[0])
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
    grad /= (self.b * sigma_r)

    self.policy += self.alpha * grad

  def runOneIteration(self):
    """
    Performing one whole iteration of the ARS algorithm
    :return: void, self.policy is updated
    """
    if self.estimator != None:  # Safe ARS - Estimation
      x_tilde = self.estimator.estimate_env_param(self.real_env_param)

    deltas = [2 * np.random.rand(*self.policy.shape) -
              1 for i in range(self.N)]
    rewards = []
    for i in range(2 * self.N):
      if i % 2 == 0:
        policy = self.policy + self.nu * deltas[i // 2]
      else:
        policy = self.policy - self.nu * deltas[i // 2]

      # Safe ARS - Safe exploration
      do_real_rollout = True
      if self.estimator is not None:
        simulator = Environment(x_tilde)
        reward = simulator.rollout(policy)
        if reward <= self.threshold:
          do_real_rollout = False

      if do_real_rollout:
        # TODO: MODIFY HERE FOR PARALLEL IMPLEMENTATION
        reward, saved_states = \
          self.real_world.rollout(policy, covariance=self.covariance,
                                  mean=self.mean)
        rewards.append(reward)
        if not self.V1:
          self.saved_states += saved_states
        if self.estimator is not None:
          self.estimator.add_trajectory(saved_states, policy)

    order = self.sort_directions(deltas, rewards)
    self.update_policy(deltas, rewards, order)

    if self.V1 is not True:
      states_array = np.array(self.saved_states)
      self.mean = np.mean(states_array, axis=0)
      self.covariance = np.cov(states_array.T)
      # print(f"mean = {self.mean}")
      # print(f"cov = {self.covariance}")

  def runTraining(self):
    """
    Run the training. After each iteration, evaluate the current policy by
    doing one rollout. Save the obtained reward after each iteration.
    :return: array of float. Rewards obtained after each iteration.
    """
    rewards = []
    for j in range(self.n_it):
      self.runOneIteration()
      r, _ = self.real_world.rollout(self.policy)
      rewards.append(r)
      if j % 10 == 0:
        print(f"Seed {self.n_seed} ------ V1 = {self.V1}; "
              f"n={self.real_env_param.n}; " +
              f"h={self.real_env_param.h}; alpha={self.alpha}; nu={self.nu}; "
              f"N={self.N}" +
              f"; b={self.b}; m_i={self.real_env_param.m_i}; "
              f"l_i={self.real_env_param.l_i} " +
              f"------ Iteration {j}/{self.n_it}: {r}")
    self.real_world.close()
    return np.array(rewards)


class Experiment():

  def __init__(self, real_env_param, results_path="results/gym/"):
    """
    Constructor setting up parameters for the experience
    :param env_param: EnvParam
    """
    # Set up environment
    self.real_env_param = real_env_param
    self.results_path = results_path

  # @ray.remote
  def plot(self, n_seed, agent_param):
    """
    Plotting learning curve
    :param n_seed: number of seeds for plotting the curve, int
    :param agent_param: ARSParam
    :return: void
    """
    ARS = f"ARS_{'V1' if agent_param.select_V1 else 'V2'}" \
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
      agent = ARSAgent.remote(self.real_env_param, agent_param, seed=i)
      r_graphs.append(agent.runTraining.remote())
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
  ray.init(num_cpus=1)

  for n in range(3, 5):
    real_env_param = EnvParam(f'LeonSwimmer', n=n, H=1000, l_i=1., m_i=1.,
                              h=1e-3)
    agent_param = ARSParam(select_V1=True, n_iter=20,
                           H=1000, N=1, b=1, alpha=0.0075, nu=0.01)
    exp = Experiment(real_env_param)
    exp.plot(n_seed=1, agent_param=agent_param)
