import matplotlib.pyplot as plt
import numpy as np

f = open("results.txt", "r")
rewards = []
for line in f:
	tokens = line.split(":")
	rewards.append(float(tokens[1][1:-2]))
f.close() 

rewards = np.array(rewards)
print(rewards)
plt.plot(rewards)
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("ASR RLGlue implementation, direction (1.0 0)")
plt.savefig("RLGlue-reward.png")
plt.show()

