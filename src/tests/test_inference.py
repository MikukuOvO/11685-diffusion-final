import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from inference import build_scheduler, generate_unconditional_batches
from schedulers.scheduling_ddim import DDIMScheduler
from schedulers.scheduling_ddpm import DDPMScheduler


class DummyPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, batch_size, num_inference_steps, generator, device):
        self.calls.append(
            {
                "batch_size": batch_size,
                "num_inference_steps": num_inference_steps,
                "device": torch.device(device),
            }
        )
        images = []
        for idx in range(batch_size):
            value = len(self.calls) * 10 + idx
            array = np.full((8, 8, 3), value, dtype=np.uint8)
            images.append(Image.fromarray(array))
        return images


class InferenceTests(unittest.TestCase):
    def _make_args(self, use_ddim):
        return SimpleNamespace(
            use_ddim=use_ddim,
            num_train_timesteps=10,
            num_inference_steps=4,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            variance_type="fixed_small",
            prediction_type="epsilon",
            clip_sample=True,
            clip_sample_range=1.0,
        )

    def test_build_scheduler_selects_ddpm_or_ddim(self):
        ddpm_scheduler = build_scheduler(self._make_args(use_ddim=False), torch.device("cpu"))
        ddim_scheduler = build_scheduler(self._make_args(use_ddim=True), torch.device("cpu"))

        self.assertIsInstance(ddpm_scheduler, DDPMScheduler)
        self.assertIsInstance(ddim_scheduler, DDIMScheduler)
        self.assertEqual(ddim_scheduler.num_inference_steps, 4)

    def test_generate_unconditional_batches_stacks_and_saves(self):
        pipeline = DummyPipeline()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            images = generate_unconditional_batches(
                pipeline=pipeline,
                total_images=5,
                batch_size=2,
                num_inference_steps=3,
                generator=generator,
                device="cpu",
                save_dir=tmpdir,
            )

            self.assertEqual(images.shape, (5, 3, 8, 8))
            self.assertEqual([call["batch_size"] for call in pipeline.calls], [2, 2, 1])
            self.assertEqual(sorted(os.listdir(tmpdir)), [f"{idx:05d}.png" for idx in range(5)])
            self.assertTrue(torch.all((images >= 0.0) & (images <= 1.0)))


if __name__ == "__main__":
    unittest.main()
