import unittest

import torch

from models.vae import VAE
from models.vae_distributions import DiagonalGaussianDistribution


class VAETests(unittest.TestCase):
    def test_forward_returns_reconstruction_and_posterior(self):
        model = VAE(resolution=32, ch=32, ch_mult=[1, 2], num_res_blocks=1)
        inputs = torch.randn(2, 3, 32, 32)

        reconstructions, posterior = model(inputs)

        self.assertEqual(reconstructions.shape, inputs.shape)
        self.assertIsInstance(posterior, DiagonalGaussianDistribution)
        self.assertEqual(posterior.mean.shape[0], inputs.shape[0])

    def test_encode_decode_and_sample_shapes(self):
        model = VAE(resolution=32, ch=32, ch_mult=[1, 2], num_res_blocks=1)
        inputs = torch.randn(2, 3, 32, 32)

        latents = model.encode(inputs)
        decoded = model.decode(latents)
        samples = model.sample(batch_size=3, device="cpu")

        self.assertEqual(latents.shape[0], 2)
        self.assertEqual(decoded.shape, inputs.shape)
        self.assertEqual(samples.shape, (3, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
