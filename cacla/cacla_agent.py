import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TwoLayersNet(nn.Module):
    def __init__(self, n_in, H1):
        super(TwoLayersNet, self).__init__()
        self.linear1 = nn.Linear(n_in, H1).double()
        self.linear2 = nn.Linear(H1, 1).double()

    def forward(self, x):
        x = x.double()
        x = F.leaky_relu(self.linear1(x))
        return self.linear2(x)

    def update_weights(self, step, x):
        self.zero_grad()
        target = self.forward(x)
        target.backward(torch.ones(1).double())
        with torch.no_grad():
            for param in self.parameters():
                # print(param)
                # print(f"Gradient: {param.grad.data.sum()}")
                param += step * param.grad



class ActorFA():
    def __init__(self, n_obs, H1, n_ac):
        self.network = [TwoLayersNet(n_obs, H1) for i in range(n_ac)]
        self.n_ac = n_ac

    def approximate_action(self, state):
        x = torch.tensor(state)
        seq = [self.network[i](x) for i in range(self.n_ac)]
        out = torch.cat(seq)
        return out.detach().numpy()

    def update_weights(self, alpha, state, action, FA_act):
        x = torch.tensor(state)
        for i in range(self.n_ac):
            step = alpha * (action[i] - FA_act[i])
            self.network[i].update_weights(step, x)


class CriticFA():
    def __init__(self, n_obs, H1):
        self.network = TwoLayersNet(n_obs, H1)

    def approximate_value(self, state):
        x = torch.tensor(state)
        return self.network(x).item()

    def update_weigths(self, alpha, delta, state):
        x = torch.tensor(state)
        step = alpha * delta
        self.network.update_weights(step, x)


class CACLA_agent:

    def __init__(self, gamma, alpha, sigma):
        # Parameters
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma

    def run(self, env, n_iter, H=1000, train=True, render=False):
        # Function approximator
        n_obs = env.observation_space.shape[0]
        n_ac = env.action_space.shape[0]
        actor = ActorFA(n_obs, 12, n_ac)
        critic = CriticFA(n_obs, 12)

        rewards = []
        state = env.reset() # Initialization
        for i in range(n_iter):
            if render:
                env.render() # Rendering
            if not train:
                with torch.no_grad():
                    FA_act = actor.approximate_action(state)  # Actor function approximation
                    action = np.random.multivariate_normal(FA_act,
                                                           self.sigma * np.identity(len(FA_act)))  # Gaussian policy
                    new_state, reward, done, info = env.step(action)  # Environment step
            else:
                FA_act = actor.approximate_action(state)  # Actor function approximation
                # print(f"FA_act = {FA_act}")
                action = np.random.multivariate_normal(FA_act, self.sigma * np.identity(len(FA_act)))  # Gaussian policy
                # print(action)
                new_state, reward, done, info = env.step(action)  # Environment step
                # print(f"Value: {critic.approximate_value(state)}")
                temp_diff = reward + self.gamma * critic.approximate_value(new_state) - critic.approximate_value(state) # Temporal difference
                critic.update_weigths(self.alpha, temp_diff, state)  # Update critic FA
                # print(f"Temporal difference: {temp_diff}")
                if temp_diff > 0:
                    actor.update_weights(self.alpha, state, action, FA_act)  # CACLA
                state = new_state

            rewards.append(reward)
            # if done:
            #     break
            if i%H == 0 and i > 0:
                print(f"Iteration {i}/{n_iter}: reward: {reward}")

        return rewards
