import numpy as np
import torch
import timm
from typing import Optional, Callable
from ..bird.config import CFG


class AttnBlock(torch.nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.drop = torch.nn.Dropout(0.2)

    def forward(self, x):
        shape = x.shape
        x = x.view(shape[0], shape[1], -1).permute(2, 0, 1)
        x = self.norm(self.drop(self.attn(x, x, x)[0]) + x)
        x = x.permute(1, 2, 0).reshape(shape)
        return x


class BirdTransformer(torch.nn.Module):
    augmentations: Optional[Callable] = None
    predict_last: bool = False

    def __init__(self, backbone='resnet18', num_classes=1, task='classification',
                 embed_dim=768, num_heads=8, att_blocks=2, pretrained=True):
        super().__init__()

        backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=1)

        enc_features = list(backbone.children())[-1].in_features
        self.encoder = torch.nn.Sequential(*list(backbone.children())[:-2])

        att_blocks = [AttnBlock(embed_dim, num_heads) for _ in range(att_blocks)]
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(enc_features, embed_dim, (CFG.sg['mel_bins'] // 32, 1)),
            *att_blocks,
            torch.nn.Conv2d(embed_dim, num_classes, 1),
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
        if self.augmentations:
            x = self.augmentations(x, self)
        x = x.to(CFG.device)
        y = self.encoder(x)
        y = self.head(y)
        return y

    def forward_segmentation(self, x):
        logits = self.segmentation(x)
        elogits = torch.exp(logits)

        cselogits = torch.cumsum(elogits, dim=-1)
        zeros = torch.zeros((*cselogits.shape[:-1], 1)).to(cselogits.device)
        cselogits = torch.cat((zeros, cselogits), dim=-1)

        logsumexplogits = []
        for start in range(x.shape[-1] * 512 // CFG.FS - CFG.DURATION + 1):
            a, b = self.sec2sample(start), self.sec2sample(start + CFG.DURATION)
            lselogits = torch.log((cselogits[..., 0, b] - cselogits[..., 0, a]))
            logsumexplogits.append(lselogits)
        result = torch.stack(logsumexplogits, dim=-2)
        if not self.predict_last:
            result = result[..., :-1, :]
        result = self.seg_head(result)
        return result

    def forward_classification(self, x):
        y = self.segmentation(x)
        y = torch.logsumexp(y, -1).squeeze(-1) - torch.Tensor([x.shape[-1]]).to(y.device).log()
        return y
