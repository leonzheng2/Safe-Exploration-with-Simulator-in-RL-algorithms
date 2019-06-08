import math
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import ray


def matprint(mat, fmt="g"):
  col_maxes = [max([len(("{:" + fmt + "}").format(x))
                    for x in col]) for col in mat.T]
  for x in mat:
    for i, y in enumerate(x):
      print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
    print("")


class SwimmerEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, direction=np.array([1., 0.]), n=3, max_u=5.,
               l_i=1., k=10., m_i=1., h=0.003):
    """
    Constructor
    :param direction: array of length 2
    :param n: int
    :param max_u: float
    :param l_i: float
    :param k: float
    :param m_i: float
    :param h: float
    """
    # Parameters of the environment
    self.direction = direction
    self.n = n
    self.max_u = max_u
    self.l_i = l_i
    self.k = k
    self.m_i = m_i
    self.h = h

    # Observation and action space
    n_obs = 2 * self.n + 2
    n_action = self.n - 1
    inf = 1000
    self.observation_space = spaces.Box(-inf,
                                        inf, shape=(n_obs,), dtype=np.float32)
    self.action_space = spaces.Box(-self.max_u, self.max_u,
                                   shape=(n_action,), dtype=np.float32)

  def step(self, action):
    """
    Doing one step of RL environment. Returns next step given the given and
    previous state

    :param action: array
    :return: array, float, boolean, dictionary
    """
    self.G_dot, self.theta, self.theta_dot = \
        self.next_observation(action, self.G_dot, self.theta, self.theta_dot)
    ob = self.get_state()
    reward = self.get_reward()
    done = self.check_terminal()
    info = {}
    return ob, reward, done, info

  def reset(self):
    """
    Reset the environment to the initial state. Return the corresponding
    observation.

    :return: array
    """
    self.G_dot = np.full(2, 0.)
    self.theta = np.full(self.n, math.pi / 2)
    self.theta_dot = np.full(self.n, 0.)
    return self.get_state()

  def next_observation(self, torque, G_dot, theta, theta_dot):
    """
    Helper method for doing one step. Compute accelerations and do
    semi-implicit Euler intergration

    :param torque: array
    :param G_dot: array
    :param theta: array
    :param theta_dot: array
    :return: array, array, array
    """
    G_dotdot, theta_dotdot = self.compute_accelerations(
        torque, G_dot, theta, theta_dot)
    # Semi-implicit Euler integration
    G_dot = G_dot + self.h * G_dotdot
    theta_dot = theta_dot + self.h * theta_dotdot
    theta = theta + self.h * theta_dot
    return G_dot, theta, theta_dot

  def compute_accelerations(self, torque, G_dot, theta, theta_dot):
    """
    Solve the linear equation to compute accelerations.
    :param torque: array
    :param G_dot: array
    :param theta: array
    :param theta_dot: array
    :return: array, array
    """

    A_dot_X, A_dot_Y, A_dotdot_X, A_dotdot_Y = self.compute_points_speed_acc(
        G_dot, theta, theta_dot)
    force_X, force_Y = self.compute_joint_force(
        theta, A_dot_X, A_dot_Y, A_dotdot_X, A_dotdot_Y)
    system = self.compute_dynamic_matrix(
        theta, theta_dot, torque, force_X, force_Y)

    G_dotdot, theta_dotdot = self.solve(system)

    return G_dotdot, theta_dotdot

  def compute_points_speed_acc(self, G_dot, theta, theta_dot):
    A_dot_X = np.zeros(self.n + 1)
    A_dot_Y = np.zeros(self.n + 1)
    A_dotdot_X = np.zeros((self.n + 1, self.n + 3))
    A_dotdot_Y = np.zeros((self.n + 1, self.n + 3))

    G_dot_X = 0.
    G_dot_Y = 0.
    G_dotdot_X = np.zeros(self.n + 3)
    G_dotdot_Y = np.zeros(self.n + 3)

    # Frame of swimmer's head
    for i in range(1, self.n + 1):
      # A_dot_i = A_dot_{i-1} + l * theta_dot_{i-1} * n_{i-1}
      A_dot_X[i] = A_dot_X[i - 1] - self.l_i * \
          theta_dot[i - 1] * np.sin([theta[i - 1]])
      A_dot_Y[i] = A_dot_Y[i - 1] + self.l_i * \
          theta_dot[i - 1] * np.cos([theta[i - 1]])

      # A_dot_i = A_dot_{i-1} + l * theta_dotdot_{i-1} * n_{i-1}
      # - l * theta_dot_{i-1}^2 * p_{i-1}
      A_dotdot_X[i] = np.array(A_dotdot_X[i - 1], copy=True)
      A_dotdot_Y[i] = np.array(A_dotdot_Y[i - 1], copy=True)
      A_dotdot_X[i][2 + (i - 1)] -= self.l_i * np.sin(theta[i - 1])
      A_dotdot_Y[i][2 + (i - 1)] += self.l_i * np.cos(theta[i - 1])
      A_dotdot_X[i][self.n + 2] -= self.l_i * \
          theta_dot[i - 1] ** 2 * np.cos(theta[i - 1])
      A_dotdot_Y[i][self.n + 2] -= self.l_i * \
          theta_dot[i - 1] ** 2 * np.sin(theta[i - 1])

      # G_dot
      G_dot_X += 1 / self.n * (A_dot_X[i - 1] + A_dot_X[i]) / 2
      G_dot_Y += 1 / self.n * (A_dot_Y[i - 1] + A_dot_Y[i]) / 2

      # G_dotdot
      G_dotdot_X += 1 / self.n * (A_dotdot_X[i - 1] + A_dotdot_X[i]) / 2
      G_dotdot_Y += 1 / self.n * (A_dotdot_Y[i - 1] + A_dotdot_Y[i]) / 2

      # matprint(f"1.{i} A_dotdot_Y", A_dotdot_Y)

    # Change of frame
    A_dot_X += G_dot[0] - G_dot_X
    A_dot_Y += G_dot[1] - G_dot_Y
    for i in range(self.n + 1):
      A_dotdot_X[i][0] += 1
      A_dotdot_Y[i][1] += 1
      A_dotdot_X[i] -= G_dotdot_X
      A_dotdot_Y[i] -= G_dotdot_Y
      # matprint(f"2.{i} A_dotdot_Y", A_dotdot_Y)

    return A_dot_X, A_dot_Y, A_dotdot_X, A_dotdot_Y

  def compute_joint_force(self, theta, A_dot_X, A_dot_Y,
                          A_dotdot_X, A_dotdot_Y):
    force_X = np.zeros((self.n + 1, self.n + 3))
    force_Y = np.zeros((self.n + 1, self.n + 3))

    for i in range(1, self.n + 1):
      force_X[i] = force_X[i - 1] + self.m_i * \
          (A_dotdot_X[i - 1] + A_dotdot_X[i]) / 2
      force_Y[i] = force_Y[i - 1] + self.m_i * \
          (A_dotdot_Y[i - 1] + A_dotdot_Y[i]) / 2
      # F is the friction force
      n_i = np.array([-np.sin(theta[i - 1]), np.cos(theta[i - 1])])
      G_dot_i = np.array([(A_dot_X[i - 1] + A_dot_X[i]) / 2,
                          (A_dot_Y[i - 1] + A_dot_Y[i]) / 2])
      F = -self.k * self.l_i * np.dot(G_dot_i, n_i)
      force_X[i][self.n + 2] -= F * n_i[0]
      force_Y[i][self.n + 2] -= F * n_i[1]
      # matprint(f"1.{i} force_X", force_X)

    return force_X, force_Y

  def compute_dynamic_matrix(self, theta, theta_dot, torque, force_X, force_Y):
    system = np.zeros((self.n + 2, self.n + 3))

    system[0] = force_X[self.n]
    system[1] = force_Y[self.n]
    for i in range(1, self.n + 1):
      system[2 + (i - 1)] += self.l_i / 2 * (
          np.cos(theta[i - 1]) * (force_Y[i] + force_Y[i - 1]) -
          np.sin(theta[i - 1]) * (force_X[i] + force_X[i - 1]))
      system[2 + (i - 1)][2 + (i - 1)] -= self.m_i * self.l_i ** 2 / 12
      system[2 + (i - 1)][self.n + 2] += self.k * \
          theta_dot[i - 1] * self.l_i ** 3 / 12
      if i - 2 >= 0:
        system[2 + (i - 1)][self.n + 2] += torque[i - 2]
      if i - 1 < self.n - 1:
        system[2 + (i - 1)][self.n + 2] -= torque[i - 1]

    return system

  def solve(self, system):
    A = system[:self.n + 2, :self.n + 2]
    B = -system[:, self.n + 2]
    X = np.linalg.solve(A, B)

    return X[:2], X[2:]

  def get_state(self):
    """
    Return the observation in an array form
    :return: array
    """
    ob = self.G_dot.tolist()
    for i in range(self.n):
      ob += [self.theta[i], self.theta_dot[i]]
    return ob

  def get_reward(self):
    """
    Compute the reward for the current observation
    :return: float
    """
    return self.G_dot.dot(self.direction)

  def check_terminal(self):
    """
    Check if the episode is finished or not.
    :return: boolean
    """
    # TODO
    return False

  def render(self, mode='human'):
    return


@ray.remote
class Worker():

  def __init__(self, policy, n, H, l_i, m_i, h):
    # print(f"New worker - {policy[0][0]}")
    self.policy = policy
    self.H = H
    self.env = SwimmerEnv(n=n, l_i=l_i, m_i=m_i, h=h)

  def select_action(self, observation):
    """
    Compute the action vector, using linear policy
    :param observation: vector
    :return: vector
    """
    observation = np.array(observation)
    action = np.matmul(self.policy, observation)
    return action

  def rollout(self):
    """
    Doing self.H steps following the given policy, and return the final
    total reward.

    :return: float
    """
    total_reward = 0
    observation = self.env.reset()
    for t in range(self.H):
      # self.env.render()
      action = self.select_action(observation)
      observation, reward, done, info = self.env.step(action)
      # print(f"State = {observation}")
      total_reward += reward
      if done:
        return total_reward
    return total_reward


@ray.remote
class ARSAgent():

  def __init__(self, n_it=1000,
               N=1, b=1, alpha=0.02, nu=0.02,
               n=3, H=1000, l_i=1., m_i=1., h=0.01,
               select_V1=True,
               seed=None):
    self.env = SwimmerEnv(n=n, l_i=l_i, m_i=m_i, h=h)  # Environment
    # Linear policy
    self.policy = np.zeros((self.env.action_space.shape[0],
                            self.env.observation_space.shape[0]))
    self.n_it = n_it

    # Agent parameters
    self.N = N
    self.b = b
    self.alpha = alpha
    self.nu = nu

    # Environment parameters
    self.n = n
    self.H = H
    self.l_i = l_i
    self.m_i = m_i
    self.h = h

    # V2
    self.V1 = select_V1
    self.mean = np.zeros(self.env.observation_space.shape[0])
    self.covariance = np.identity(self.env.observation_space.shape[0])
    self.saved_states = []

    self.n_seed = seed
    np.random.seed(self.n_seed)

  def select_action(self, policy, observation):
    """
    Compute the action vector, using linear policy
    :param policy: matrix
    :param observation: vector
    :return: vector
    """
    observation = np.array(observation)
    if self.V1 is True:
      return np.matmul(policy, observation)
    # ARS V2
    diag_covariance = np.diag(self.covariance) ** (-1 / 2)
    policy = np.matmul(policy, np.diag(diag_covariance))
    action = np.matmul(policy, observation - self.mean)
    return action

  def rollout(self, policy):
    """
    Doing self.H steps following the given policy, and return the final
    total reward.

    :param policy: matrix
    :return: float
    """
    total_reward = 0
    observation = self.env.reset()
    for t in range(self.H):
      # self.env.render()
      action = self.select_action(policy, observation)
      observation, reward, done, info = self.env.step(action)
      if self.V1 is not True:
        self.saved_states.append(observation)
      total_reward += reward
      if done:
        return total_reward
    return total_reward

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
    Update the linear policy following the update step, after collecting
    the rewards

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
    deltas = [2 * np.random.rand(*self.policy.shape) -
              1 for i in range(self.N)]
    rewards = []
    for i in range(2 * self.N):
      if i % 2 == 0:
        policy = self.policy + self.nu * deltas[i // 2]
      else:
        policy = self.policy - self.nu * deltas[i // 2]
      rewards.append(self.rollout(policy))
    # worker = Worker.remote(policy, self.n, self.H,
    # self.l_i, self.m_i, self.h)
    # rewards.append(worker.rollout.remote())
    # rewards = ray.get(rewards)
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
      r = self.rollout(self.policy)
      rewards.append(r)
      if j % 10 == 0:
        print(f"Seed {self.n_seed} ------ V1 = {self.V1}; n={self.n}; "
              f"h={self.env.h}; alpha={self.alpha}; nu={self.nu}; N={self.N}; "
              f"b={self.b}; m_i={self.env.m_i}; l_i={self.env.l_i} ------ "
              f"Iteration {j}/{self.n_it}: {r}")
    self.env.close()
    return np.array(rewards)

# @ray.remote


def plot(select_V1, n_seed, n, h, n_iter, alpha, nu, N, b, m_i, l_i):
  print(f"select_V1={select_V1}")
  print(f"n={n}")
  print(f"h={h}")
  print(f"alpha={alpha}")
  print(f"nu={nu}")
  print(f"N={N}")
  print(f"b={b}")
  print(f"m_i={m_i}")
  print(f"l_i={l_i}")

  # Seeds
  r_graphs = []
  for i in range(n_seed):
    agent = ARSAgent.remote(select_V1=select_V1, n_it=n_iter, seed=i,
                            alpha=alpha, nu=nu, n=n, h=h, N=N, b=b,
                            m_i=m_i, l_i=l_i)
    r_graphs.append(agent.runTraining.remote())
  r_graphs = np.array(ray.get(r_graphs))

  # Plot graphs
  plt.figure(figsize=(10, 8))
  for rewards in r_graphs:
    plt.plot(rewards)
  plt.title(f"V1={select_V1}, n={n}, seeds={n_seed}, h={h}, alpha={alpha}, "
            f"nu={nu}, N={N}, b={b}, n_iter={n_iter}, m_i={round(m_i, 2)}, "
            f"l_i={round(l_i,2)}")
  plt.xlabel("Iteration")
  plt.ylabel("Reward")
  np.save(f"array/ars_n={n}_V1={select_V1}_random_seeds={n_seed}_h={h}_"
          f"alpha={alpha}_nu={nu}_N={N}_b={b}_n_iter={n_iter}_"
          f"m_i={round(m_i, 2)}_l_i={round(l_i,2)}", r_graphs)
  plt.savefig(
      f"new/ars_n={n}_V1={select_V1}_random_seeds={n_seed}_h={h}_"
      f"alpha={alpha}_nu={nu}_N={N}_b={b}_n_iter={n_iter}_"
      f"m_i={round(m_i, 2)}_l_i={round(l_i,2)}.png")
  # plt.show()
  plt.close()

  # Plot mean and std
  plt.figure(figsize=(10, 8))
  x = np.linspace(0, n_iter - 1, n_iter)
  mean = np.mean(r_graphs, axis=0)
  std = np.std(r_graphs, axis=0)
  plt.plot(x, mean, 'k', color='#CC4F1B')
  plt.fill_between(x, mean - std, mean + std, alpha=0.5,
                   edgecolor='#CC4F1B', facecolor='#FF9848')
  plt.title(f"n={n}, seeds={n_seed}, h={h}, alpha={alpha}, nu={nu}, N={N}, "
            f"b={b}, n_iter={n_iter}, m_i={round(m_i, 2)}, l_i={round(l_i,2)}")
  plt.xlabel("Iteration")
  plt.ylabel("Reward")
  plt.savefig(
      f"new/ars_n={n}_V1={select_V1}_random_seeds={n_seed}_"
      f"h={h}_alpha={alpha}_"
      f"nu={nu}_N={N}_b={b}_n_iter={n_iter}_m_i={round(m_i, 2)}_"
      f"l_i={round(l_i,2)}_average.png")
  # plt.show()
  plt.close()


if __name__ == '__main__':
  ray.init(num_cpus=7)
  for n in range(4, 10):
    plot(select_V1=False, n_seed=7, n=n, h=0.001, n_iter=500,
         N=4, b=2, nu=0.01, alpha=0.0075, m_i=1., l_i=1.)

  # plot(select_V1=False, n_seed=4, n=3, h=0.001, n_iter=100, N=4,
  #  b=2, nu=0.01, alpha=0.0075, m_i=1., l_i=1.)

  # n_list = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  # max_rewards = []
  # train_90 = []
  # train_95 = []
  # train_99 = []
  #
  # for n in n_list:
  #     if n == 8 or n == 9 or n>10:
  #         rewards = np.load(f"save/n/ars_n={n}_random_seeds=8_h=0.001_
  #  alpha=0.0075_nu=0.01_N=1_b=1_n_iter=1000_m_i=1.0_l_i=1.0.npy")
  #     else:
  #         rewards = np.load(f"save/n/ars_n={n}_random_seeds=6_h=0.001_
  #  alpha=0.0075_nu=0.01_N=1_b=1_n_iter=1000_m_i=1.0_l_i=1.0.npy")
  #
  #     mean = np.mean(rewards, axis=0)
  #     max_rewards.append(np.max(mean))
  #
  #     k = 20
  #     smooth_mean = np.convolve(mean, np.ones((len(mean) // k,)) /
  #  (len(mean) // k), 'valid')
  #     max = np.max(smooth_mean)
  #     for alpha in [0.9, 0.95, 0.99]:
  #         n_train = np.argmax(mean > alpha * max)
  #         print(f"Time to achieve {alpha} of max: {n_train}")
  #         if alpha == 0.9:
  #             train_90.append(n_train)
  #         elif alpha == 0.95:
  #             train_95.append(n_train)
  #         else:
  #             train_99.append(n_train)
  #
  # plt.plot(n_list, train_90, '-o', label="lambda=0.90")
  # plt.plot(n_list, train_95, '-o', label="lambda=0.95")
  # plt.plot(n_list, train_99, '-o', label="lambda=0.99")
  # plt.title("Time to reach lambda * max of smoothed mean")
  # plt.xlabel("Segments")
  # plt.ylabel("Iterations")
  # plt.legend()
  # plt.show()
  # plt.close()
  #
  # plt.plot(n_list, max_rewards, '-o', label="max_rewards")
  # plt.title("Max average reward achieved with ARS")
  # plt.xlabel("Segments")
  # plt.ylabel("Max average reward")
  # plt.legend()
  # plt.show()
  # plt.close()

  # grad_smooth = np.gradient(smooth_mean)
  # plt.plot(grad_smooth)
  # plt.show()
  #
  # k = 10
  # smooth_grad_smooth = np.convolve(grad_smooth, np.ones((len(grad_smooth)
  #  // k,)) / (len(grad_smooth) // k), 'valid')
  # plt.plot(smooth_grad_smooth)
  # plt.show()
  #
  # for k in [10, 20, 30, 40, 50]:
  #     smooth_mean = np.convolve(mean, np.ones((len(mean) // k,))
  #  / (len(mean) // k), 'valid')
  #     x = np.linspace(len(mean)//k//2, len(grad) - len(mean)//k//2,
  #  len(mean) - len(mean)//k + 1)
  #     plt.plot(mean, label="original")
  #     # plt.title(f"k={k}")
  #     plt.plot(x, smooth_mean, label="smooth")
  #     plt.xlabel("Iteration")
  #     plt.ylabel("Reward")
  #     plt.legend()
  #     plt.show()
  #     print(f"Max of smooth k={k}: {np.max(smooth_mean)}")
  #
  # for alpha in [0.9, 0.95, 0.99]:
  #     n_train = np.argmax(mean > alpha * max)
  #     print(f"Time to achieve {alpha} of max: {n_train}")
  #
  # for k in [5, 8, 10, 12, 15]:
  #     smooth_grad = np.convolve(grad, np.ones((len(grad) // k,)) /
  #  (len(grad) // k), 'valid')
  #     # x = np.linspace(len(grad)//k//2, len(grad) - len(grad)//k//2,
  #  len(grad) - len(grad)//k + 1)
  #     # # print(x)
  #     # plt.plot(x, smooth, label="smooth")
  #     # plt.show()
  #     first_zero = np.argmax(smooth_grad < 0)
  #     if first_zero > 0:
  #         print(f"Time when smooth gradient (k={k}) is zero: {first_zero
  #  + len(grad)//k}")
  #     else:
  #         print(f"Time when smooth gradient (k={k}) is minimum:
  #  {np.argmin(smooth_grad) + len(grad)//k}")
  #
  #
  # # print(smooth)
  # # plt.plot(mean, label="mean")
  # # plt.plot(grad, label="grad")
  # # print(len(grad))
  # # print(len(smooth))
  # # x = np.linspace(len(grad)//k//2, len(grad) - len(grad)//k//2,
  #  len(grad) - len(grad)//k + 1)
  # # # print(x)
  # # plt.plot(x, smooth, label="smooth")
  # # plt.show()
  #
