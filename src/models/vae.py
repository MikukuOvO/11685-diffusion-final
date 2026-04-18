import torch
import torch.nn as nn

from .vae_modules import Encoder, Decoder
from .vae_distributions import DiagonalGaussianDistribution


class VAE(nn.Module):
    # NOTE: do not change anything in __init__ function
    def __init__(self,
                 ### Encoder Decoder Related
                 double_z=True,
                 z_channels=3,
                 embed_dim=3,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[1,2,4],  # num_down = len(ch_mult)-1
                 num_res_blocks=2):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(in_channels=in_channels, ch=ch, out_ch=out_ch, num_res_blocks=num_res_blocks, z_channels=z_channels, ch_mult=ch_mult, resolution=resolution, double_z=double_z, attn_resolutions=[])
        self.decoder = Decoder(in_channels=in_channels, ch=ch, out_ch=out_ch, num_res_blocks=num_res_blocks, z_channels=z_channels, ch_mult=ch_mult, resolution=resolution, double_z=double_z, attn_resolutions=[])
        self.quant_conv = torch.nn.Conv2d(2*z_channels, 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)

    def encode_to_posterior(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode_latents(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    @torch.no_grad()
    def encode(self, x):
        posterior = self.encode_to_posterior(x)
        return posterior.sample()

    @torch.no_grad()
    def decode(self, z):
        return self.decode_latents(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode_to_posterior(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode_latents(z)
        return dec, posterior

    @torch.no_grad()
    def reconstruct(self, x, sample_posterior=False):
        dec, _ = self.forward(x, sample_posterior=sample_posterior)
        return dec

    @torch.no_grad()
    def sample(self, batch_size, device=None, generator=None):
        if device is None:
            device = next(self.parameters()).device
        else:
            device = torch.device(device)
        latent_h, latent_w = self.decoder.z_shape[-2:]
        latent_ch = self.post_quant_conv.in_channels
        z = torch.randn(
            (batch_size, latent_ch, latent_h, latent_w),
            generator=generator,
            device=device,
        )
        return self.decode(z)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        print(keys)
