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
plt.savefig("RLGlue-reward.png")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.title("ASR RLGlue implementation")
plt.show()

