import torch
import torch.nn as nn

class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, cond_drop_rate=0.1):
        super().__init__()
        if embed_dim is None:
            raise ValueError("`embed_dim` must be provided for ClassEmbedder.")

        # One extra token is reserved for the unconditional branch used by CFG.
        self.embedding = nn.Embedding(n_classes + 1, embed_dim)
        self.cond_drop_rate = cond_drop_rate
        self.num_classes = n_classes
        self.uncond_class = n_classes

    def forward(self, x):
        if x is None:
            raise ValueError("Class labels must be provided for conditional sampling/training.")

        x = x.long()
        
        if self.cond_drop_rate > 0 and self.training:
            drop_mask = torch.rand(x.shape[0], device=x.device) < self.cond_drop_rate
            x = x.clone()
            x[drop_mask] = self.uncond_class
        
        return self.embedding(x)

    def unconditional_embedding(self, batch_size, device):
        labels = torch.full(
            (batch_size,),
            self.uncond_class,
            dtype=torch.long,
            device=device,
        )
        return self.embedding(labels)
