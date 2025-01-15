from Agent_env import Agent
import torch

# usegpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Create an agent
agent = Agent()
agent.run()
