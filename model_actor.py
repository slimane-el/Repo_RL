from torch import nn
import torch
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_dim, n_actions, layer_size=256, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.layer1 = nn.Linear(input_dim, layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)
        self.relu = nn.ReLU()
        self.layer3 = nn.Linear(layer_size, layer_size)
        self.mean = nn.Linear(layer_size, n_actions)
        self.log_std = nn.Linear(layer_size, n_actions)

    def forward(self, state):
        out = self.relu(self.layer1(state))
        out = self.relu(self.layer2(out))
        out = self.relu(self.layer3(out))
        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # Pour actions discrètes
        #action_discrete = torch.distributions.Categorical(logits=out)

        return mean, log_std

    def sample_action(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state) # Forward le state => mean, log_std of action
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z) 
        log_pi = normal.log_prob(
            z) - torch.sum(torch.log(1 - torch.square(action) + epsilon), axis=0, keepdims=True) # e
        return action, log_pi
