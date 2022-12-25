import numpy as np
import torch


class GaussianAgent:
    def __init__(
        self,
        actor: torch.nn.Sequential,
        critic: torch.nn.Sequential,
        lr_actor: float=1e-2,
        lr_critic: float=1e-2,
        gamma: float=.99
    ) -> None:

        self.actor = actor
        self.critic = critic
        self.log_sigma = torch.ones(1, dtype=torch.double, requires_grad=True)

        self.opt_actor = torch.optim.Adam(list(self.actor.parameters()) + [self.log_sigma], lr=lr_actor) 
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma

    def pi(self, s_t: np.ndarray) -> torch.distributions.MultivariateNormal:
        s_t = torch.as_tensor(s_t).double()
        mu = self.actor(s_t)

        log_sigma = self.log_sigma
        sigma = torch.exp(log_sigma)
        
        pi = torch.distributions.MultivariateNormal(mu, torch.diag(sigma))
        return pi

    def evaluate(self, s_t: torch.Tensor) -> torch.Tensor:
        v = self.critic(s_t)
        return v

    def act(self, s_t: np.ndarray) -> np.ndarray:
        a_t = self.pi(s_t).sample()
        return a_t

    def learn(
        self,
        s_t: np.ndarray,
        s_t_next: np.ndarray,
        a_t: np.ndarray,
        r_t: float
    ) -> None:

        s_t = torch.as_tensor(s_t).double()
        s_t_next = torch.as_tensor(s_t_next).double()
        a_t = torch.as_tensor(a_t).double()
        r_t = torch.as_tensor(r_t).double()

        v_t = self.evaluate(s_t)
        v_t_next = self.evaluate(s_t_next)
        v_t_next.detach()
                
        delta = r_t + v_t_next - v_t
        
        # Critic
        loss_critic = delta.mean()
        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()
        
        # Actor
        log_prob = self.pi(s_t).log_prob(a_t)
        loss_action = -log_prob*delta.detach().item()
        self.opt_actor.zero_grad()
        loss_action.backward()
        self.opt_actor.step()
