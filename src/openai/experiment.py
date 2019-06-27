import numpy as np
import matplotlib.pyplot as plt
import ray
from src.openai.parameters import EnvParam, ARSParam, Threshold
from src.openai.ars_agent import ARSAgent


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
  def plot(self, n_seed, agent_param):
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
      plt.plot(t, [agent_param.threshold + real_env_param.epsilon * sim_thresh.compute_alpha(real_env_param.H)] * len(t), color='red',
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
    plt.figure(figsize=(10, 8))
    mean = np.mean(r_graphs, axis=0)
    std = np.std(r_graphs, axis=0)
    plt.plot(t, mean, 'k', color='#CC4F1B')
    plt.fill_between(t, mean - std, mean + std, alpha=0.5,
                     edgecolor='#CC4F1B', facecolor='#FF9848')
    if agent_param.safe:
      plt.plot(t, [agent_param.threshold] * len(t), color='black',
               linewidth=3, label="Safety threshold")
      plt.plot(t, [agent_param.threshold + real_env_param.epsilon * sim_thresh.compute_alpha(real_env_param.H)] * len(t), color='red',
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


if __name__ == '__main__':
  ray.init(num_cpus=8)

  guess_param = EnvParam('LeonSwimmer-Simulator', n=3, H=1000, l_i=1.,
                         m_i=1.,
                         h=1e-3, k=10., epsilon=0.01)
  real_env_param = EnvParam('LeonSwimmer-RealWorld', n=3, H=1000, l_i=.8,
                            m_i=1.2,
                            h=1e-3, k=10.2, epsilon=0.001)

  # Get the initial weight - As if it is the hand controller

  hand_agent = ARSParam('HandControl', V1=True, n_iter=200, H=1000, N=1, b=1,
                        alpha=0.0075, nu=0.01, safe=False, threshold=0,
                        initial_w='Zero')
  hand_exp = Experiment(real_env_param,
                        data_path=None,
                        save_data_path="src/openai/real_world_2.npz",
                        save_policy_path='src/openai/saved_hand_policy',
                        guess_param=None)
  returns = hand_exp.plot(n_seed=1, agent_param=hand_agent)

  # Get safety threshold based on known controller/experience
  mean_returns = np.mean(returns, axis=0)
  l = mean_returns[-1] * 0.99
  print(f"\nSafety threshold: {l}")
  np.savetxt("src/openai/threshold.txt", np.array([l]))

  # l = 285

  # Train the real agent in real world. Use transfered weight. Unknown real world parameters.

  K=1
  A=1
  B=0.001
  H=1000
  for A in [0.1, 0.3, 0.5, 0.7]:
    sim_thresh = Threshold(K=K, A=A, B=B)
    alpha = sim_thresh.compute_alpha(H)
    print(f"B = {B}; alpha = {alpha}")
    epsilon_range = np.linspace(0.0001, 0.01, 10)
    sim_thresh_range = [l + alpha * e for e in epsilon_range]
    min_return = []
    max_mean_returns = []

    for epsilon in epsilon_range:
      real_env_param = EnvParam('LeonSwimmer-RealWorld', n=3, H=H, l_i=.8,
                                m_i=1.2,
                                h=1e-3, k=10.2, epsilon=epsilon)
      real_agent = ARSParam(f'RLControl', V1=True, n_iter=400,
                            H=H, N=1, b=1,
                            alpha=0.0075, nu=0.01, safe=True, threshold=l,
                            initial_w='src/openai/saved_hand_policy.npy')
      real_exp = Experiment(real_env_param,
                            data_path="src/openai/real_world_2.npz",
                            save_data_path=None,
                            save_policy_path=None,
                            guess_param=None,
                            approx_error=epsilon,
                            sim_thresh=sim_thresh)
      r_graphs = real_exp.plot(n_seed=8, agent_param=real_agent)
      min_return.append(np.nanmin(r_graphs))
      mean = np.mean(r_graphs, axis=0)
      max_mean_returns.append(np.nanmax(mean))

    plt.figure(figsize=(10, 8))
    plt.plot(epsilon_range, min_return, marker='o', label="Minimum return")
    plt.plot(epsilon_range, max_mean_returns, marker='o', label="Max of mean learning curve")
    plt.plot(epsilon_range, sim_thresh_range, linestyle='--', marker='D', label="Simulator threshold")
    plt.plot(epsilon_range, [l] * len(epsilon_range), color='black',
             linewidth=2, label="Safety threshold")
    plt.legend()
    plt.xlabel("epsilon")
    plt.ylabel("Average return")
    plt.title(f"Safe ARS with approximation error of epsilon, with constants H={H}, K={K}, A={A}, B={B}")
    plt.savefig(f"results/epsilon_sim_threshold_H={H}_K={K}_A={A}_B={B}.png")
    # plt.show()
    plt.close()