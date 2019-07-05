import numpy as np
import matplotlib.pyplot as plt
import ray
from ars.ars_agent import ARSAgent


class Experiment():

  def __init__(self, real_env_param, results_path="results/gym/",
               data_path=None, save_data_path=None, save_policy_path=None,
               guess_param=None, approx_error=None, sim_thresh=None):
    """
    Constructor setting up parameters for the experience
    :param env_param: EnvParam
    """
    # Set up environment
    self.real_env_param = real_env_param
    self.results_path = results_path
    # Estimation
    self.guess_param = guess_param
    self.approx_error = approx_error
    # Simulator threshold
    self.sim_thresh = sim_thresh
    # Policy
    self.save_policy_path = save_policy_path
    # Data
    self.data_path = data_path
    self.save_data_path = save_data_path

  # @ray.remote
  def plot(self, n_seed, agent_param, plot_mean = True):
    """
    Plotting learning curve
    :param n_seed: number of seeds for plotting the curve, int
    :param agent_param: ARSParam
    :return: void
    """
    ARS = f"{agent_param.name}, " \
      f"ARS_{'V1' if agent_param.V1 else 'V2'}" \
      f"{'-t' if agent_param.b < agent_param.N else ''}, " \
      f"n_directions={agent_param.N}, " \
      f"deltas_used={agent_param.b}, " \
      f"step_size={agent_param.alpha}, " \
      f"delta_std={agent_param.nu}"
    environment = f"{self.real_env_param.name}, " \
      f"n_segments={self.real_env_param.n}, " \
      f"m_i={round(self.real_env_param.m_i, 2)}, " \
      f"l_i={round(self.real_env_param.l_i, 2)}, " \
      f"epsilon={round(self.real_env_param.epsilon, 4)}, " \
      f"deltaT={self.real_env_param.h}"

    print(f"\n------ {environment} ------")
    print(ARS + '\n')

    # Seeds
    r_graphs = []
    for i in range(n_seed):
      agent = ARSAgent.remote(self.real_env_param, agent_param,
                              seed=i, data_path=self.data_path,
                              guess_param=self.guess_param,
                              approx_error=self.approx_error,
                              sim_thresh=self.sim_thresh)
      r_graphs.append(agent.runTraining.remote(save_data_path=
                                               self.save_data_path,
                                               save_policy_path=self.save_policy_path))
    r_graphs = np.array(ray.get(r_graphs))

    # Plot graphs
    t = np.linspace(0,
                    agent_param.n_iter * 2 * agent_param.N * agent_param.H,
                    agent_param.n_iter + 1)
    plt.figure(figsize=(10, 8))
    for rewards in r_graphs:
      plt.plot(t, rewards)
    if agent_param.safe:
      plt.plot(t, [agent_param.threshold] * len(t), color='black',
               linewidth=3, label="Safety threshold")
      plt.plot(t, [agent_param.threshold + self.real_env_param.epsilon * self.sim_thresh.compute_alpha(self.real_env_param.H)] * len(t), color='red',
               linewidth=3, label="Simulator threshold")
      plt.legend()
    plt.title(f"------ {environment} ------\n{ARS}")
    plt.xlabel("Timesteps")
    plt.ylabel("Rollouts Average Return")
    np.save(f"{self.results_path}array/"
            f"{environment.replace(', ', '-')}-"
            f"{ARS.replace(', ', '-')}", r_graphs)
    plt.savefig(f"{self.results_path}new/"
                f"{environment.replace(', ', '-')}-"
                f"{ARS.replace(', ', '-')}.png")
    # plt.show()
    plt.close()

    # Plot mean and std
    if plot_mean:
      plt.figure(figsize=(10, 8))
      mean = np.mean(r_graphs, axis=0)
      std = np.std(r_graphs, axis=0)
      plt.plot(t, mean, 'k', color='#CC4F1B')
      plt.fill_between(t, mean - std, mean + std, alpha=0.5,
                       edgecolor='#CC4F1B', facecolor='#FF9848')
      if agent_param.safe:
        plt.plot(t, [agent_param.threshold] * len(t), color='black',
                 linewidth=3, label="Safety threshold")
        plt.plot(t, [agent_param.threshold + self.real_env_param.epsilon * self.sim_thresh.compute_alpha(self.real_env_param.H)] * len(t), color='red',
                 linewidth=3, label="Simulator threshold")
        plt.legend()
      plt.title(f"------ {environment} ------\n{ARS}")
      plt.xlabel("Timesteps")
      plt.ylabel("Rollouts Average Return")
      plt.savefig(f"{self.results_path}new/"
                  f"{environment.replace(', ', '-')}-"
                  f"{ARS.replace(', ', '-')}-average.png")
      # plt.show()
      plt.close()

    return r_graphs
