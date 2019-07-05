import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TwoLayersNet(nn.Module):
    def __init__(self, n_in, H1, H2, n_out):
        super(TwoLayersNet, self).__init__()
        self.linear1 = nn.Linear(n_in, H1).double()
        self.linear2 = nn.Linear(H1, H2).double()
        self.linear3 = nn.Linear(H2, n_out).double()

    def forward(self, x):
        x = x.double()
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return F.relu(self.linear3(x))


class ActorFA():
    def __init__(self, n_obs, H1, H2, n_ac):
        self.networks = nn.ModuleList([TwoLayersNet(n_obs, H1, H2, 1) for i in range(n_ac)])

    def approximate_action(self, state):
        x = torch.tensor(state)
        seq = [network(x) for network in self.networks]
        return torch.cat(seq).detach().numpy()

    def update_weights(self, alpha, state, action):
        x = torch.tensor(state)
        for i, network_i in enumerate(self.networks):
            FA_act_i = network_i(x)
            network_i.zero_grad()
            FA_act_i.backward()
            with torch.no_grad():
                for param in network_i.parameters():
                    param += alpha * (action[i] - FA_act_i) * param.grad


class CriticFA():
    def __init__(self, n_obs, H1, H2):
        super(CriticFA, self).__init__()
        self.network = TwoLayersNet(n_obs, H1, H2, 1)

    def approximate_value(self, state):
        x = torch.tensor(state)
        return self.network(x)

    def update_weigths(self, alpha, delta, state):
        x = torch.tensor(state)
        FA_val = self.network(x)
        self.network.zero_grad()
        FA_val.backward()
        with torch.no_grad():
            for param in self.network.parameters():
                param += alpha * delta * param.grad


class CACLA_agent:

    def __init__(self, gamma, alpha, sigma, H=1000):
        # Parameters
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma
        self.H = H

    def run(self, env, n_iter, train=True):
        # Function approximator
        n_obs = env.observation_space.shape[0]
        n_ac = env.action_space.shape[0]
        actor = ActorFA(n_obs, 10, 10, n_ac)
        critic = CriticFA(n_obs, 10, 10)

        save = []
        self.rewards = [] # Save rewards for every H steps
        state = env.reset() # Initialization
        for i in range(n_iter):
            if not train:
                with torch.no_grad():
                    FA_act = actor.approximate_action(state)  # Actor function approximation
                    action = np.random.multivariate_normal(FA_act,
                                                           self.sigma * np.identity(len(FA_act)))  # Gaussian policy
                    new_state, reward, done, info = env.step(action)  # Environment step
            else:
                FA_act = actor.approximate_action(state)  # Actor function approximation
                action = np.random.multivariate_normal(FA_act, self.sigma * np.identity(len(FA_act)))  # Gaussian policy
                # print(action)
                new_state, reward, done, info = env.step(action)  # Environment step
                temp_diff = reward + self.gamma * critic.approximate_value(new_state) - critic.approximate_value(state) # Temporal difference
                critic.update_weigths(self.alpha, temp_diff, state)  # Update critic FA
                if temp_diff > 0:
                    actor.update_weights(self.alpha, state, action)  # CACLA
                state = new_state

            self.rewards.append(reward)
            if done:
                break

            if i % self.H == 0:
                save.append(np.sum(self.rewards))
                # print(f"Iteration {i}: reward={save[-1]}")
                self.rewards = []

        return save