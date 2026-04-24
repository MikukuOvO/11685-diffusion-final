import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from train import resolve_training_schedule
from train import build_lr_scheduler
from train import resolve_output_paths
from train import sync_lr_scheduler_to_step


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

    def test_sync_lr_scheduler_to_step_matches_direct_stepping(self):
        param_a = torch.nn.Parameter(torch.tensor(1.0))
        optimizer_a = torch.optim.AdamW([param_a], lr=0.01)
        param_b = torch.nn.Parameter(torch.tensor(1.0))
        optimizer_b = torch.optim.AdamW([param_b], lr=0.01)
        args = SimpleNamespace(
            lr_scheduler="cosine",
            lr_warmup_steps=2,
            max_train_steps=10,
            min_lr=0.001,
            learning_rate=0.01,
        )
        scheduler_a = build_lr_scheduler(optimizer_a, args)
        scheduler_b = build_lr_scheduler(optimizer_b, args)

        for _ in range(6):
            optimizer_a.step()
            scheduler_a.step()

        sync_lr_scheduler_to_step(scheduler_b, current_step=6)

        self.assertAlmostEqual(
            optimizer_a.param_groups[0]["lr"],
            optimizer_b.param_groups[0]["lr"],
            delta=1e-8,
        )
        self.assertEqual(scheduler_b.last_epoch, 6)

    def test_resolve_output_paths_indexes_non_exact_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "exp-0-old"))

            run_name, output_dir = resolve_output_paths(
                tmpdir,
                run_name="demo",
                exact_output_dir=False,
            )

            self.assertEqual(run_name, "exp-1-demo")
            self.assertEqual(output_dir, os.path.join(tmpdir, "exp-1-demo"))

    def test_resolve_output_paths_keeps_exact_output_dir_stable(self):
        run_name, output_dir = resolve_output_paths(
            "/tmp/my-modal-run",
            run_name="modal-demo",
            exact_output_dir=True,
        )

        self.assertEqual(run_name, "modal-demo")
        self.assertEqual(output_dir, "/tmp/my-modal-run")


if __name__ == "__main__":
    unittest.main()
