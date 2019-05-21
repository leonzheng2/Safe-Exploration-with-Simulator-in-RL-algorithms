import gym

tasks = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Humanoid-v2", "HumanoidStandup-v2", "InvertedDoublePendulum-v2", "InvertedPendulum-v2", "Reacher-v2", "Swimmer-v2", "Walker2d-v2"]

for task in tasks:
    env = gym.make(task)
    n_obs = env.observation_space.shape[0]
    n_action = env.action_space.shape[0]
    print(f"Task \"{task}\": n_obseration={n_obs}, n_action={n_action}")