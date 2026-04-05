---
license: mit
language: en
tags:
  - deep-q-network
  - reinforcement-learning
  - cartpole-v1
model-index:
  - name: cartpole-dqn-converged
    results:
      - task:
          type: reinforcement-learning
        dataset:
          type: cartpole-v1
          name: CartPole-v1
        metrics:
        - name: average_reward
          type: reward
          value: 437.235
        - name: max_reward
          type: reward
          value: 500.0
        - name: min_reward
          type: reward
          value: 12.0
        - name: success_rate
          type: success_rate
          value: 0.87
        - name: reward_variance
          type: variance
          value: 21881.059
        - name: reward_std_dev
          type: std_dev
          value: 147.922
        - name: final_epsilon
          type: epsilon
          value: 0.01
        - name: td_error
          type: loss
          value: 0.929
        - name: final_loss
          type: loss
          value: 2.860
        - name: total_regret
          type: regret
          value: 12553.0
        - name: convergence_episode
          type: episode
          value: 0
---

# DQN CartPole Optimal Agent

This repository contains a Deep Q-Network (DQN) agent trained to solve the CartPole-v1 environment using reinforcement learning.

---

## Overview

The agent learns to balance a pole on a moving cart by interacting with the environment and maximizing cumulative reward. The model is trained using experience replay and a target network for stable learning.

---

## Environment

- **Environment**: CartPole-v1 (Gymnasium)
- **State Space**: 4-dimensional continuous vector  
- **Action Space**: 2 discrete actions (left, right)

---

## Model Architecture

- Fully connected neural network
- Two hidden layers (128 neurons each)
- ReLU activation
- Output: Q-values for each action

---

## Training Details

- **Algorithm**: Deep Q-Network (DQN)
- **Episodes**: 2000
- **Discount** Factor (γ): 0.99
- **Batch Size**: 64
- **Learning Rate**: 5e-4
- **Replay Buffer Size**: 10,000
- **Target Network Update**: Every 20 episodes
- **Epsilon Decay Strategy** for *exploration*

---

## Files

- `dqn_cartpole.pth` – **PyTorch model** weights  
- `dqn_cartpole.onnx` – **ONNX model** for deployment  
- `dqn_cartpole.onnx.data` – **ONNX weights**  
- `config.json` – **Model configuration**  
- `evaluation.py` – Script to **run and evaluate the model**   

---

## Usage

### Load and Run (PyTorch)

```python
import torch
import gymnasium as gym

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

model = DQN()
model.load_state_dict(torch.load("dqn_cartpole.pth", map_location="cpu"))
model.eval()

env = gym.make("CartPole-v1")

state, _ = env.reset()
done = False

while not done:
    state_tensor = torch.FloatTensor(state)
    with torch.no_grad():
        action = torch.argmax(model(state_tensor)).item()

    state, reward, done, truncated, _ = env.step(action)

    if truncated:
        break

env.close()
```

## Evaluation

The *model evaluation* over *2000 episodes*.

- Average Reward: 181.372
- Max Reward: 500.0
- Min Reward: 9.0
- Success Rate: 0.264
- Reward Variance: 24288.796
- Reward Standard Deviation: 155.849
- Final epsilon: 0.01
- TD Error: 0.519
- Final Loss: 2.860
- Total Regret: 637256.0
- Convergence Episode: 678


The *model evaluation* over *200 episodes*.

- Average Reward: 437.235
- Max Reward: 500.0
- Min Reward: 12.0
- Success Rate: 0.87
- Reward Variance: 21881.059
- Reward Standard Deviation: 147.922
- Final epsilon: 0.01
- TD Error: 0.929
- Final Loss: 2.860
- Total Regret: 12553.0
- Convergence Episode: 0

## Author

[**Bindupautra Jyotibrat**](https://github.com/Jyotibrat)