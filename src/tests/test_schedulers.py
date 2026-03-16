import unittest

import torch

from schedulers.scheduling_ddim import DDIMScheduler
from schedulers.scheduling_ddpm import DDPMScheduler


class SchedulerTests(unittest.TestCase):
    def test_ddpm_schedule_and_timesteps(self):
        scheduler = DDPMScheduler(
            num_train_timesteps=10,
            beta_start=0.0001,
            beta_end=0.02,
        )

        self.assertEqual(scheduler.betas.shape, (10,))
        self.assertTrue(torch.equal(scheduler.timesteps, torch.arange(9, -1, -1)))

        scheduler.set_timesteps(4)
        self.assertTrue(torch.equal(scheduler.timesteps.cpu(), torch.tensor([9, 6, 3, 0])))
        self.assertEqual(scheduler.previous_timestep(6), 3)
        self.assertEqual(scheduler.previous_timestep(0), -1)

    def test_ddpm_add_noise_matches_closed_form(self):
        scheduler = DDPMScheduler(num_train_timesteps=10)
        original = torch.ones(2, 3, 4, 4)
        noise = torch.zeros_like(original)
        timesteps = torch.tensor([0, 9], dtype=torch.long)

        noisy = scheduler.add_noise(original, noise, timesteps)
        expected_scales = scheduler.alphas_cumprod[timesteps].sqrt().view(2, 1, 1, 1)
        expected = expected_scales * original

        torch.testing.assert_close(noisy, expected)

    def test_ddpm_step_recovers_x0_at_t0_without_variance(self):
        scheduler = DDPMScheduler(num_train_timesteps=10, clip_sample=False)
        x0 = torch.randn(2, 3, 4, 4)
        alpha0 = scheduler.alphas_cumprod[0].sqrt()
        xt = alpha0 * x0
        model_output = torch.zeros_like(x0)

        prev_sample = scheduler.step(model_output=model_output, timestep=0, sample=xt)

        torch.testing.assert_close(prev_sample, x0, atol=1e-5, rtol=1e-5)

    def test_ddim_eta_zero_is_deterministic(self):
        scheduler = DDIMScheduler(
            num_train_timesteps=10,
            num_inference_steps=4,
            clip_sample=False,
        )
        x0 = torch.randn(2, 3, 4, 4)
        t = int(scheduler.timesteps[0].item())
        prev_t = scheduler.previous_timestep(t)
        alpha_t = scheduler.alphas_cumprod[t].sqrt()
        alpha_prev = scheduler.alphas_cumprod[prev_t].sqrt()

        xt = alpha_t * x0
        model_output = torch.zeros_like(x0)
        prev_sample = scheduler.step(model_output=model_output, timestep=t, sample=xt, eta=0.0)

        expected = alpha_prev * x0
        torch.testing.assert_close(prev_sample, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
