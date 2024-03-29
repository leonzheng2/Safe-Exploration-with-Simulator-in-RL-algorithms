"""
Script for plotting the learning curve for ARS on Swimmer task.
Choose parameters in the variables `real_env_param` and `ars_agent_param`.
"""

from ars.experiment import Experiment
from ars.parameters import EnvParam, ARSParam
import ray

# Initialize ray for parallel computations over seeds
ray.init(num_cpus=8)

# Running one experiment
real_env_param = EnvParam('LeonSwimmer-RealWorld', n=3, H=1000, l_i=.8,
                          m_i=1.2,
                          h=1e-3, k=10.2, epsilon=0) # Environment parameters
ars_agent_param = ARSParam(f'RLControl', V1=True, n_iter=100, H=1000,
                      N=1, b=1,
                      alpha=0.0075, nu=0.01, safe=False, threshold=0,
                      initial_w='Zero') # ARS Agent parameters
exp = Experiment(real_env_param, results_path="results/gym/") # Creating an instance of the current experience
exp.plot(4, ars_agent_param) # Several seeds for the previous parameters, results in a graph saved in the trajectory
