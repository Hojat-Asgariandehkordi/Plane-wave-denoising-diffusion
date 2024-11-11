import torch
import torch.nn.functional as F

@torch.no_grad()
def sample_timestep(model, x, t):
    y = model(x, t)
    return y

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_loss(model, x_0, y_0, t, sigma_ratio, device="cpu"):
    x_noisy, noise = forward_diffusion_sample(x_0, y_0, t, sigma_ratio, device)
    pred = model(x_noisy, t.view(4))
    return F.l1_loss(noise, pred)

def forward_diffusion_sample(x_0, noise, t, sigma_ratio, device="cpu"):
    normal = torch.randn_like(x_0)
    x_noisy = x_0 + ((t / sigma_ratio) * noise)
    return x_noisy, noise / sigma_ratio
