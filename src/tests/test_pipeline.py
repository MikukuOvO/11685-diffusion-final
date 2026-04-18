import unittest

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from models.class_embedder import ClassEmbedder
from pipelines.ddpm import DDPMPipeline
from schedulers.scheduling_ddim import DDIMScheduler
from schedulers.scheduling_ddpm import DDPMScheduler


class ZeroUNet(nn.Module):
    def __init__(self, input_size=8, input_ch=3):
        super().__init__()
        self.input_size = input_size
        self.input_ch = input_ch

    def forward(self, x, t, c=None):
        return torch.zeros_like(x)


class PipelineTests(unittest.TestCase):
    def test_ddpm_pipeline_returns_pil_images(self):
        pipeline = DDPMPipeline(
            unet=ZeroUNet(),
            scheduler=DDPMScheduler(num_train_timesteps=10, clip_sample=False),
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(123)
        images = pipeline(batch_size=2, num_inference_steps=4, generator=generator, device="cpu")

        self.assertEqual(len(images), 2)
        self.assertTrue(all(isinstance(image, Image.Image) for image in images))
        self.assertTrue(all(image.size == (8, 8) for image in images))

    def test_pipeline_uses_requested_num_inference_steps(self):
        scheduler = DDPMScheduler(num_train_timesteps=10, clip_sample=False)
        pipeline = DDPMPipeline(unet=ZeroUNet(), scheduler=scheduler)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        pipeline(batch_size=1, num_inference_steps=3, generator=generator, device="cpu")

        self.assertEqual(scheduler.num_inference_steps, 3)
        self.assertEqual(len(scheduler.timesteps), 3)

    def test_pipeline_is_deterministic_for_same_seed(self):
        scheduler = DDPMScheduler(num_train_timesteps=10, clip_sample=False)
        pipeline = DDPMPipeline(unet=ZeroUNet(), scheduler=scheduler)

        generator_1 = torch.Generator(device="cpu")
        generator_1.manual_seed(7)
        images_1 = pipeline(batch_size=1, num_inference_steps=4, generator=generator_1, device="cpu")

        generator_2 = torch.Generator(device="cpu")
        generator_2.manual_seed(7)
        images_2 = pipeline(batch_size=1, num_inference_steps=4, generator=generator_2, device="cpu")

        np.testing.assert_array_equal(np.array(images_1[0]), np.array(images_2[0]))

    def test_ddim_pipeline_runs_with_same_interface(self):
        scheduler = DDIMScheduler(
            num_train_timesteps=10,
            num_inference_steps=4,
            clip_sample=False,
        )
        pipeline = DDPMPipeline(unet=ZeroUNet(), scheduler=scheduler)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(11)
        images = pipeline(batch_size=1, num_inference_steps=4, generator=generator, device="cpu")

        self.assertEqual(len(images), 1)
        self.assertIsInstance(images[0], Image.Image)

    def test_cfg_pipeline_accepts_class_labels(self):
        scheduler = DDPMScheduler(num_train_timesteps=10, clip_sample=False)
        class_embedder = ClassEmbedder(embed_dim=4, n_classes=3, cond_drop_rate=0.0)
        pipeline = DDPMPipeline(
            unet=ZeroUNet(),
            scheduler=scheduler,
            class_embedder=class_embedder,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(5)
        images = pipeline(
            batch_size=2,
            num_inference_steps=2,
            classes=[0, 1],
            guidance_scale=2.0,
            generator=generator,
            device="cpu",
        )

        self.assertEqual(len(images), 2)
        self.assertTrue(all(isinstance(image, Image.Image) for image in images))

    def test_conditional_pipeline_requires_classes(self):
        scheduler = DDPMScheduler(num_train_timesteps=10, clip_sample=False)
        class_embedder = ClassEmbedder(embed_dim=4, n_classes=3, cond_drop_rate=0.0)
        pipeline = DDPMPipeline(
            unet=ZeroUNet(),
            scheduler=scheduler,
            class_embedder=class_embedder,
        )

        with self.assertRaises(ValueError):
            pipeline(batch_size=1, num_inference_steps=2, device="cpu")


if __name__ == "__main__":
    unittest.main()
