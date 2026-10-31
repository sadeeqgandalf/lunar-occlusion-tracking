"""CVAE policy: the latent-variable idea behind ACT (Zhao et al., RSS 2023, the ALOHA robot's policy).

Training: an encoder reads the expert's chunk and squeezes what the observation cannot explain (here: which side the
expert took) into a small random code z. A decoder rebuilds the chunk from the observation and z. A KL term keeps the
codes close to N(0, I), so at drive time a z drawn from N(0, I) means "some side", not "the middle".

    python policy/cvae.py      # trains on seed 0's demonstrations and scores closed loop
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Normalizer
from sim import H, from_unit, to_unit

def mlp(i: int, hidden: int, o: int):
    return nn.Sequential(nn.Linear(i, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, o))


class CVAEPolicy(nn.Module):
    def __init__(self, z: int=4, hidden: int=128, beta: float = 0.01, obs_dim: int=6):
        super().__init__()
        self.z, self.beta = z, beta
        self.norm = Normalizer(obs_dim)
        self.encoder = mlp(obs_dim + 2 * H, hidden, 2* z)
        self.decoder = mlp(obs_dim + z, hidden, 2* H)


    def decode(self, obs, z):
        return self.decoder(torch.cat([self.norm(obs), z], -1)).view(-1, H, 2)
        
        
    
    def loss(self, obs, unit_actions, gen: torch.Generator = None):
        mu, logvar = self.encoder(torch.cat([self.norm(obs), unit_actions.flatten(1)], -1)).chunk(2, -1)
        z = mu + (0.5 * logvar).exp() * torch.randn(mu.shape, generator=gen)
        kl = 0.5 * (mu ** 2 + logvar.exp() -1-logvar).sum(-1).mean()
        return F.mse_loss(self.decode(obs, z), unit_actions) + self.beta * kl


    @torch.no_grad()
    def sample(self, obs, gen: torch.Generator=None):
        return from_unit(self.decode(obs, torch.randn(len(obs), self.z, generator = gen)))


if __name__ == "__main__":
    import dataset
    from train_eval import EPOCHS, score, test_suites

    torch.manual_seed(0)
    data = dataset.build(seed=0)
    obs, unit = data["obs"], to_unit(data["actions"])
    policy = CVAEPolicy()
    policy.norm.fit(obs)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, EPOCHS+1):
        for idx in torch.randperm(len(obs)).split(256):
            loss = policy.loss(obs[idx], unit[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 ==0:
            print (f"epoch {epoch:2d} loss {loss.item():.4f}")
        

    g = torch.Generator().manual_seed(11)
    for suite, r in score(lambda: lambda o: policy.sample(o, g), test_suites()).items():
        print(f"{suite:10s}  crash {r['crash']:4.1f}%   goal {r['goal']:5.1f}%   passed left {r['left']}%")


