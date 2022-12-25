import matplotlib.pyplot as plt
import torch

from src.environments import CustomPendulumEnv
from src.agents import GaussianAgent
from src.algorithms import actor_critic


if __name__ == "__main__":
    env = CustomPendulumEnv()

    actor = torch.nn.Sequential(
        torch.nn.Linear(2, 124),
        torch.nn.ReLU(),
        torch.nn.Linear(124, 124),
        torch.nn.ReLU(),
        torch.nn.Linear(124, 1)
    ).double()

    critic = torch.nn.Sequential(
        torch.nn.Linear(2, 124),
        torch.nn.ReLU(),
        torch.nn.Linear(124, 124),
        torch.nn.ReLU(),
        torch.nn.Linear(124, 1)
    ).double()

    agent = GaussianAgent(
        actor,
        critic,
        lr_actor=1e-5,
        lr_critic=1e-5,
        gamma=0.99
    )

    _, totals, _ = actor_critic(env, agent, epochs=5_000, T=100)

    fig = plt.figure(figsize=[8, 7])
    plt.grid()
    plt.plot(totals)
    plt.xlim(0, 5_000)
    plt.ylim(0, 110)
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.tight_layout()
    fig.savefig('figures/fig_results.jpg', dpi=300)
