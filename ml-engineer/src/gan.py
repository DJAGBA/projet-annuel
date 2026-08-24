"""
Architectures partagées DCGAN / WGAN-GP.

DCGAN: Radford et al. 2015 (https://arxiv.org/abs/1511.06434)
WGAN-GP: Gulrajani et al. 2017 (https://arxiv.org/abs/1704.00028)

Les deux variantes réutilisent la même architecture convolutionnelle de base ;
seules la fonction de perte et la normalisation du discriminateur/critic diffèrent
(BatchNorm interdit dans le critic WGAN-GP à cause du gradient penalty par échantillon).
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, channels: int = 1, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_dim) -> (B, latent_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """Utilisé tel quel pour DCGAN (sortie sigmoid) et comme critic pour WGAN-GP
    (sortie linéaire, pas de BatchNorm -> voir use_batchnorm)."""

    def __init__(self, channels: int = 1, feature_maps: int = 64, use_batchnorm: bool = True,
                 use_sigmoid: bool = True):
        super().__init__()
        norm = (lambda c: nn.BatchNorm2d(c)) if use_batchnorm else (lambda c: nn.InstanceNorm2d(c, affine=True))

        layers = [
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            norm(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            norm(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
        ]
        self.features = nn.Sequential(*layers)
        self.use_sigmoid = use_sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x).view(-1)
        return torch.sigmoid(out) if self.use_sigmoid else out


def weights_init(m: nn.Module) -> None:
    """Initialisation recommandée par le papier DCGAN (N(0, 0.02))."""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
    """Gradient penalty de WGAN-GP (Gulrajani et al., éq. 3)."""
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)

    critic_out = critic(interpolated)
    grads = torch.autograd.grad(
        outputs=critic_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp
