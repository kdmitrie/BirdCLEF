import numpy as np
import torch
import timm
from typing import Optional, Callable
from ..bird.config import CFG


class AttnBlock(torch.nn.Module):
    def __init__(self, n=512, nheads=8):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(n, nheads)
        self.norm = torch.nn.LayerNorm(n)
        self.drop = torch.nn.Dropout(0.2)

    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], shape[1], -1).permute(2, 0, 1)
        x = self.norm(self.drop(self.attn(x, x, x)[0]) + x)
        x = x.permute(1, 2, 0).reshape(shape)
        return x


class BirdTransformer(torch.nn.Module):
    augmentations: Optional[Callable] = None

    def __init__(self, backbone='resnet18', num_classes=1, task='classification'):
        super().__init__()

        backbone = timm.create_model(backbone, pretrained=True, in_chans=1)
        nh = 768

        enc_features = list(backbone.children())[-1].in_features
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-2])

        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(enc_features, nh, (CFG.sg['mel_bins'] // 32, 1)),
            AttnBlock(nh),
            AttnBlock(nh),
            torch.nn.Conv2d(nh, num_classes, 1),
        )

        self.seg_head = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 1),
            torch.nn.Sigmoid()
        )

        self.set_task(task)

    @staticmethod
    def sec2sample(s):
        return np.round(s * CFG.FS / 2 ** 14).astype(int)

    def set_task(self, task):
        if task == 'classification':
            self.forward = self.forward_classification
        else:
            self.forward = self.forward_segmentation

    def segmentation(self, x):
        if self.augmentations and self.training:
            x = self.augmentations(x)
        x = x.to(CFG.device)
        y = self.encoder(x)
        y = self.head(y)
        return y

    def forward_segmentation(self, x):
        logits = self.segmentation(x)

        elogits = torch.exp(logits)
        cselogits = torch.cumsum(elogits, dim=-1)

        logsumexplogits = []
        for start in range(x.shape[-1] * 512 // CFG.FS - CFG.DURATION):
            a, b = self.sec2sample(start), self.sec2sample(start + CFG.DURATION)
            lselogits = torch.log((cselogits[..., 0, b] - cselogits[..., 0, a]))
            logsumexplogits.append(lselogits)
        result = torch.stack(logsumexplogits, dim=-2)
        result = self.seg_head(result)
        return result

    def forward_classification(self, x):
        y = self.segmentation(x)
        y = torch.logsumexp(y, -1).squeeze(-1) - torch.Tensor([x.shape[-1]]).to(y.device).log()
        return y
