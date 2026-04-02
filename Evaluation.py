import torch
import gymnasium as gym
import numpy as np

# Define model (same as training)
class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


# Load model
device = torch.device("cpu")
model = DQN()
model.load_state_dict(torch.load("dqn_cartpole.pth", map_location=device))
model.eval()

# Environment
env = gym.make("CartPole-v1")

episodes = 5

for ep in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item()

        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        if truncated:
            break

    print(f"Episode {ep+1}: Reward = {total_reward}")

env.close()