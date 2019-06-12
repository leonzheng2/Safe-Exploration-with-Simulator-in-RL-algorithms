from gym.envs.swimmer.remy_swimmer_env import SwimmerEnv
import numpy as np


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
