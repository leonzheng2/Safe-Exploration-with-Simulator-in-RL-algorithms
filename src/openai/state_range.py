import ray
from src.openai.parameters import EnvParam, ARSParam
from src.openai.experiment import Experiment
from src.openai.environment import Environment
import numpy as np

ray.init()

# Train a good policy
env_param = EnvParam("LeonSwimmer", n=3, H=1000, l_i=1., m_i=1., h=1e-3, k=10., epsilon=0)
ars_param = ARSParam("ARS", V1=True, n_iter=300, H=1000, N=1, b=1, alpha=0.01, nu=0.01, safe=False, threshold=0, initial_w='Zero')
exp = Experiment(env_param, save_policy_path="src/openai/data/saved_good_policy")
exp.plot(1, ars_param, plot_mean=False)

# Get the trained policy
policy = np.load("src/openai/data/saved_good_policy.npy")

# Do a rollout with the good policy
print("\nDoing a rollout with good policy...")
env = Environment(env_param)
total_rewards, saved_states = env.rollout(policy)
print(f"Return obtained: {total_rewards}")

# Get the range value of state variables when running with good policy
print("\nExtracting the range of state variables...")
min = np.min(saved_states, axis=0)
max = np.max(saved_states, axis=0)
print(f"Min values: {min}")
print(f"Max values: {max}")

print(f"\nRange of dot(G)_x: [{min[0]}, {max[0]}]")
print(f"Range of dot(G)_y: [{min[1]}, {max[1]}]")
for i in range((len(min)-2)//2):
  print(f"Range of dot(theta)_{i+1}: [{min[3+2*i]}, {max[3+2*i]}]")
