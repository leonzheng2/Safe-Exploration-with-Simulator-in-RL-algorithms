import numpy as np
import matplotlib.pyplot as plt
import ray
from src.openai.parameters import EnvParam, ARSParam
from src.openai.ars_agent import ARSAgent


class Experiment():

  def __init__(self, real_env_param, results_path="results/gym/",
               data_path=None, save_data_path=None, save_policy_path=None,
               guess_param=None):
    """
    Constructor setting up parameters for the experience
    :param env_param: EnvParam
    """
    # Set up environment
    self.real_env_param = real_env_param
    self.results_path = results_path

    # Guess parameters
    self.guess_param = guess_param

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
                              seed=i, data_path=self.data_path,
                              guess_param=self.guess_param)
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
    plt.title(f"------ {environment} ------\n{ARS}")
    plt.xlabel("Timesteps")
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
    mean = np.mean(r_graphs, axis=0)
    std = np.std(r_graphs, axis=0)
    plt.plot(t, mean, 'k', color='#CC4F1B')
    plt.fill_between(t, mean - std, mean + std, alpha=0.5,
                     edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title(f"------ {environment} ------\n{ARS}")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.savefig(f"{self.results_path}new/"
                f"{environment.replace(', ', '-')}-"
                f"{ARS.replace(', ', '-')}-average.png")
    # plt.show()
    plt.close()

    return mean

if __name__ == '__main__':
  ray.init(num_cpus=8)
  real_env_param = EnvParam(f'LeonSwimmer-Simulator', n=3, H=1000, l_i=.8,
                         m_i=1.2,
                         h=1e-3, k=10.2)
  guess_param = EnvParam(f'LeonSwimmer-RealWorld', n=3, H=1000, l_i=1.,
                            m_i=1.,
                            h=1e-3, k=10.)

  # Get the initial weight - As if it is the hand controller

  hand_agent = ARSParam(V1=True, n_iter=1000, H=1000, N=1, b=1,
                       alpha=0.0075, nu=0.01, safe=False, threshold=0,
                       initial_w='Zero')
  hand_exp = Experiment(real_env_param,
                       data_path=None,
                       save_data_path=None,
                       save_policy_path='src/openai/saved_hand_policy',
                       guess_param=None)
  mean_returns = hand_exp.plot(n_seed=1, agent_param=hand_agent)

  # Get safety threshold based on known controller/experience
  l = mean_returns[-1] * 0.9
  np.savetxt("src/openai/threshold.txt", l)
  # l = 250

  # Train the real agent in real world. Unknown real world parameters.
  real_agent = ARSParam(V1=True, n_iter=100, H=1000, N=1, b=1,
                        alpha=0.0075, nu=0.01, safe=False, threshold=l,
                        initial_w='src/openai/saved_hand_policy.npy')
  real_exp = Experiment(real_env_param,
                        data_path="src/openai/real_world.npz",
                        save_data_path=None,
                        guess_param=guess_param)  # For cheating, use real_env_param
  real_exp.plot(n_seed=1, agent_param=real_agent)
