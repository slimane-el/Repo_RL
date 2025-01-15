from torch import nn
import torch
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, inputs_dim, layer_size=256):
        super(Critic, self).__init__()
        self.state_layer_1 = nn.Linear(inputs_dim[0], 32)
        self.state_layer_2 = nn.Linear(32, 32)
        self.action_layer = nn.Linear(inputs_dim[1], 32)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.fc3 = nn.Linear(layer_size, layer_size)
        self.q_value = nn.Linear(layer_size, 1)

    def forward(self, state, action):
        state_out = self.relu(self.state_layer_1(state))
        state_out = self.relu(self.state_layer_2(state_out))
        action_out = self.relu(self.action_layer(action))
        concat = torch.cat([state_out, action_out], dim=1)
        out = self.relu(self.fc1(concat))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        return self.q_value(out)
