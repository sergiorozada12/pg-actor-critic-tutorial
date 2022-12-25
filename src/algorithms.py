from src.environments import CustomPendulumEnv
from src.agents import GaussianAgent


def actor_critic(
    env: CustomPendulumEnv,
    agent: GaussianAgent,
    epochs: int=100,
    T: int=1000
):

    totals, timesteps = [], []
    for epoch in range(epochs):
        R = 0
        s_t = env.reset()
        for t in range(T):
            a_t = agent.act(s_t)
            s_t_next, r_t, d_t, _ = env.step(a_t.numpy())
            
            R += r_t
            
            agent.learn(s_t, s_t_next, a_t, r_t)

            if d_t:
                break 
                
            s_t = s_t_next

        totals.append(R)
        timesteps.append(t)
        print(f'{epoch}/{epochs}: {totals[-1]} - {t} \r', end='')
    return agent, totals, timesteps
