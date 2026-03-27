from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MixStyle(nn.Module):
    """
    Feature-level MixStyle.
    - p: probability to apply MixStyle during training
    - alpha: Beta distribution parameter for mixing coefficient (alpha>0)
    Implementation follows common feature-mix approaches:
      - compute per-sample channel mean and std over spatial dims
      - normalize features, then mix per-sample mean/std with permuted batch
      - denormalize with mixed statistics
    """
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super(MixStyle, self).__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        """
        x: tensor B x C x H x W
        """
        if not self.training or random.random() > self.p:
            return x

        B, C, H, W = x.size()
        if B == 1:
            return x

        # flatten spatial dims
        x_ = x.view(B, C, -1)

        # per-sample mean & std (B x C x 1 x 1)
        mu = x_.mean(dim=2).view(B, C, 1, 1)
        var = x_.var(dim=2, unbiased=False).view(B, C, 1, 1)
        sigma = (var + self.eps).sqrt()

        # normalize
        x_normed = (x - mu) / sigma

        # sample mixing coefficient lambda from Beta(alpha, alpha)
        # use torch.distributions to keep on device
        try:
            m = torch.distributions.Beta(self.alpha, self.alpha)
            lam = m.sample((B,)).to(x.device)  # (B,)
        except Exception:
            # fallback to numpy if distribution not available (should not be needed)
            import numpy as np
            lam = torch.tensor(np.random.beta(self.alpha, self.alpha, size=(B,)), device=x.device, dtype=x.dtype)

        lam = lam.view(B, 1, 1, 1)

        # random permutation for pairing
        perm = torch.randperm(B).to(x.device)

        mu2 = mu[perm]
        sigma2 = sigma[perm]

        # mix statistics
        mu_mix = mu * lam + mu2 * (1 - lam)
        sigma_mix = sigma * lam + sigma2 * (1 - lam)

        # denormalize
        out = x_normed * sigma_mix + mu_mix
        return out


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 mixstyle=True, mix_p=0.5, mix_alpha=0.1):
        """
        UNET with optional feature-level MixStyle.

        Parameters:
        - in_channels, out_channels, init_features: same as before
        - mixstyle: bool - whether to enable MixStyle
        - mix_p: probability to apply MixStyle during training
        - mix_alpha: beta distribution alpha for MixStyle
        """
        super(UNET, self).__init__()
        features = init_features
        self.encoder1 = UNET._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNET._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNET._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNET._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNET._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNET._block((features * 8) * 2, features * 8, name="dec4")


        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNET._block((features * 4) * 2, features * 4, name="dec3")


        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNET._block((features * 2) * 2, features * 2, name="dec2")


        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNET._block(features * 2, features, name="dec1")


        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # MixStyle setup: create module(s) and flags
        self.use_mixstyle = bool(mixstyle)
        if self.use_mixstyle:
            # create MixStyle modules; you can create one and reuse or multiple placed modules.
            # Here we instantiate two independent MixStyle modules (after enc1 and enc2).
            self.mixstyle1 = MixStyle(p=mix_p, alpha=mix_alpha)
            self.mixstyle2 = MixStyle(p=mix_p, alpha=mix_alpha)

    def forward(self, x):
        enc1 = self.encoder1(x)
        # apply MixStyle after encoder1 if enabled
        if self.use_mixstyle:
            enc1 = self.mixstyle1(enc1)

        enc2 = self.encoder2(self.pool1(enc1))
        # apply MixStyle after encoder2 if enabled
        if self.use_mixstyle:
            enc2 = self.mixstyle2(enc2)

        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1) #64
        dec1 = self.decoder1(dec1)     #64 inchannels, #32 outchannels
        return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
