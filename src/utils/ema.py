import copy

import torch


class EMAModel:
    def __init__(self, model, decay=0.9999, update_after_step=0, update_every=1):
        if not 0.0 < decay < 1.0:
            raise ValueError("`decay` must be between 0 and 1.")
        if update_after_step < 0:
            raise ValueError("`update_after_step` must be non-negative.")
        if update_every <= 0:
            raise ValueError("`update_every` must be positive.")

        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.num_updates = 0

        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def to(self, device):
        self.ema_model = self.ema_model.to(device)
        return self

    def state_dict(self):
        return {
            "model": self.ema_model.state_dict(),
            "num_updates": self.num_updates,
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
        }

    def load_state_dict(self, state_dict):
        if "model" in state_dict:
            self.ema_model.load_state_dict(state_dict["model"])
            self.num_updates = state_dict.get("num_updates", self.num_updates)
            self.decay = state_dict.get("decay", self.decay)
            self.update_after_step = state_dict.get("update_after_step", self.update_after_step)
            self.update_every = state_dict.get("update_every", self.update_every)
        else:
            self.ema_model.load_state_dict(state_dict)

    def _current_decay(self):
        # Warm up the EMA to avoid over-smoothing early training.
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model, step):
        if step < self.update_after_step:
            self.ema_model.load_state_dict(model.state_dict())
            return False
        if step % self.update_every != 0:
            return False

        self.num_updates += 1
        decay = self._current_decay()
        ema_state_dict = self.ema_model.state_dict()
        model_state_dict = model.state_dict()

        for key, value in model_state_dict.items():
            ema_value = ema_state_dict[key]
            if not torch.is_floating_point(value):
                ema_value.copy_(value)
            else:
                ema_value.lerp_(value, 1.0 - decay)
        return True
