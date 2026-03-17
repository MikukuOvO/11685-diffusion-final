import unittest

from train import resolve_training_schedule


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


if __name__ == "__main__":
    unittest.main()
