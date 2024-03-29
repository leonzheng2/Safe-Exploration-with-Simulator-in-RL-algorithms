"""
Solving LQR with CACLA, without Safe Exploration.

Choose instances of LQR to solve.

Choose hyperparameters of CACLA:
    - discount factor `gamma`
    - standard deviation for Gaussian policy `sigma`
    - step size for weights update `alpha`
    - number of iterations `n_iter`

Here is an example of hyperparameter tuning. The results is plotted in a graph.
"""


from envs.gym_lqr.lqr_env import LinearQuadReg
import numpy as np
from cacla.cacla_agent import CACLA_LQR_agent
import matplotlib.pyplot as plt
from cacla.window import window_convolution


''' Easy instances of LQR to solve '''
# Environment
# Instance 1
A_1 = np.ones((1,1))
B_1 = np.ones((1,1))
Q_1 = np.ones((1,1))
R_1 = np.ones((1,1))
lqr_1 = LinearQuadReg(A_1, B_1, Q_1, R_1)

# Instance 2
A_2 = np.array([[0, 1], [1, 0]])
B_2 = np.array([[0], [1]])
Q_2 = np.array([[1, 0], [0, 1]])
R_2 = np.array([[1]])
lqr_2 = LinearQuadReg(A_2, B_2, Q_2, R_2)

''' CACLA agent for solving easy instances of LQR '''
# Grid search
optimal_F = np.array([[1-np.sqrt(3), 0]])
n_iter = 200000
gamma = 1
sigma = 0.1
alphas = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]
distance = []

''' Hyperparameter tuning '''
for alpha in alphas:
    # Agent
    agent = CACLA_LQR_agent(lqr_2)
    _, _, results = agent.run(n_iter, gamma, alpha, sigma)
    print(agent.F)
    distance.append(np.linalg.norm(agent.F - optimal_F))

    # Plot graph
    H = 1000
    t = np.linspace(H, n_iter, n_iter - H)
    plt.plot(t, window_convolution(results, H), label=f"gamma={round(gamma, 3)}, alpha={alpha}, sigma={sigma}")
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel(f"Average of the last {H} rewards")
    plt.title(f"CACLA on LQR learning curve")
    plt.savefig(f"results/cacla/LQR/2_gamma={round(gamma, 3)}_alpha={alpha}_sigma={sigma}.png")
    # plt.show()
    plt.close()

''' Graph plotting '''
plt.semilogx(alphas, distance)
plt.xlabel("Backpropagation step size")
plt.ylabel("||F - optimal_F||_2")
plt.grid()
plt.title(f"LQR: distance of the policy after {n_iter} iterations to the optimal one, for different backprop step size")
plt.show()
plt.close()
