import unittest
from types import SimpleNamespace

import torch
from train import resolve_training_schedule
from train import build_lr_scheduler


class TrainUtilsTests(unittest.TestCase):
    def test_uses_full_epoch_schedule_by_default(self):
        num_epochs, max_train_steps = resolve_training_schedule(
            num_epochs=3,
            num_update_steps_per_epoch=10,
            max_train_steps=None,
        )

        self.assertEqual(num_epochs, 3)
        self.assertEqual(max_train_steps, 30)

    def test_respects_smaller_requested_max_train_steps(self):
        num_epochs, max_train_steps = resolve_training_schedule(
            num_epochs=1,
            num_update_steps_per_epoch=100,
            max_train_steps=25,
        )

        self.assertEqual(num_epochs, 1)
        self.assertEqual(max_train_steps, 25)

    def test_expands_epochs_when_requested_steps_exceed_full_schedule(self):
        num_epochs, max_train_steps = resolve_training_schedule(
            num_epochs=1,
            num_update_steps_per_epoch=100,
            max_train_steps=250,
        )

        self.assertEqual(num_epochs, 3)
        self.assertEqual(max_train_steps, 250)

    def test_cosine_lr_scheduler_warms_up_and_decays(self):
        param = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([param], lr=0.01)
        args = SimpleNamespace(
            lr_scheduler="cosine",
            lr_warmup_steps=2,
            max_train_steps=10,
            min_lr=0.001,
            learning_rate=0.01,
        )
        scheduler = build_lr_scheduler(optimizer, args)

        lrs = []
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        self.assertGreaterEqual(max(lrs[:3]), lrs[0])
        self.assertGreater(lrs[2], lrs[-1])
        self.assertAlmostEqual(lrs[-1], 0.001, delta=0.001)


if __name__ == "__main__":
    unittest.main()
