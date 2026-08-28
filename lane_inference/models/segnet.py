"""Segmentation network for IDD-Lite."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import config as cfg


_TAPS = {1: (16, 2), 3: (24, 4), 6: (40, 8), 12: (112, 16), 16: (960, 32)}


def _conv_bn_act(cin, cout, k=3, s=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def _sep_conv(cin, cout):
    """Depthwise-separable 3x3 convolution."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, 1, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class LRASPP(nn.Module):
    """Lite Reduced Atrous Spatial Pyramid Pooling (Howard et al., 2019)."""

    def __init__(self, cin: int, cout: int = 128):
        super().__init__()
        self.local = _conv_bn_act(cin, cout, k=1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.local(x) * self.gate(x)


class MobileNetV3UNet(nn.Module):
    """MobileNetV3-Large encoder with an FPN decoder."""

    def __init__(self, num_classes: int = cfg.NUM_CLASSES,
                 pretrained: bool = True, decoder_channels: int = 64,
                 lane_head: bool = False):
        super().__init__()
        self.has_lane_head = lane_head
        weights = "IMAGENET1K_V2" if pretrained else None
        self.encoder = torchvision.models.mobilenet_v3_large(weights=weights).features
        self.tap_indices = sorted(_TAPS)

        c1, c2, c3, c4, c5 = (_TAPS[i][0] for i in self.tap_indices)
        d = decoder_channels

        self.context = LRASPP(c5, 128)
        self.lat5 = nn.Conv2d(128, d, 1, bias=False)
        self.lat4 = nn.Conv2d(c4, d, 1, bias=False)
        self.lat3 = nn.Conv2d(c3, d, 1, bias=False)
        self.lat2 = nn.Conv2d(c2, d, 1, bias=False)

        self.smooth4 = _sep_conv(d, d)
        self.smooth3 = _sep_conv(d, d)
        self.smooth2 = _sep_conv(d, d)

        self.head = nn.Sequential(_sep_conv(d, d), nn.Conv2d(d, num_classes, 1))
        self.aux_head = nn.Sequential(_conv_bn_act(c4, d, k=3),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(d, num_classes, 1))


        self.lane_head = nn.Sequential(
            _sep_conv(d, d), nn.Conv2d(d, 1, 1)) if lane_head else None

    def _encode(self, x):
        feats = {}
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in _TAPS:
                feats[i] = x
        return [feats[i] for i in self.tap_indices]

    def forward(self, x):
        size = x.shape[-2:]
        _, c2, c3, c4, c5 = self._encode(x)

        p5 = self.lat5(self.context(c5))
        p4 = self.smooth4(self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:],
                                                        mode="bilinear", align_corners=False))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:],
                                                        mode="bilinear", align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:],
                                                        mode="bilinear", align_corners=False))

        out = F.interpolate(self.head(p2), size=size, mode="bilinear", align_corners=False)
        lane = None
        if self.lane_head is not None:
            lane = F.interpolate(self.lane_head(p2), size=size,
                                 mode="bilinear", align_corners=False)

        if self.training:
            aux = F.interpolate(self.aux_head(c4), size=size,
                                mode="bilinear", align_corners=False)
            return (out, aux, lane) if lane is not None else (out, aux)
        return (out, lane) if lane is not None else out


class DiceLoss(nn.Module):
    """Soft multi-class Dice, ignoring IGNORE_INDEX pixels."""

    def __init__(self, num_classes: int = cfg.NUM_CLASSES,
                 ignore_index: int = cfg.IGNORE_INDEX, eps: float = 1.0):
        super().__init__()
        self.n, self.ignore, self.eps = num_classes, ignore_index, eps

    def forward(self, logits, target):
        valid = (target != self.ignore)
        target = torch.where(valid, target, torch.zeros_like(target))
        onehot = F.one_hot(target, self.n).permute(0, 3, 1, 2).float()
        probs = logits.softmax(1) * valid.unsqueeze(1)
        onehot = onehot * valid.unsqueeze(1)

        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        card = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (card + self.eps)
        return 1.0 - dice.mean()


class SegLoss(nn.Module):
    """Weighted cross-entropy + Dice, with optional auxiliary and lane supervision."""

    def __init__(self, class_weights=None, dice_weight: float = 0.5,
                 aux_weight: float = 0.4, label_smoothing: float = 0.0,
                 lane_weight: float = 0.5, lane_pos_weight: float = 20.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights,
                                      ignore_index=cfg.IGNORE_INDEX,
                                      label_smoothing=label_smoothing)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.aux_weight = aux_weight
        self.lane_weight = lane_weight
        self.register_buffer("lane_pos_weight", torch.tensor(lane_pos_weight))

    @staticmethod
    def _binary_dice(logits, target, eps: float = 1.0):
        prob = torch.sigmoid(logits)
        inter = (prob * target).sum()
        card = prob.sum() + target.sum()
        return 1.0 - (2 * inter + eps) / (card + eps)

    def forward(self, outputs, target, lane_target=None):
        lane = None
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 3:
                main, aux, lane = outputs
            else:
                main, aux = outputs
        else:
            main, aux = outputs, None

        loss = self.ce(main, target) + self.dice_weight * self.dice(main, target)
        if aux is not None:
            loss = loss + self.aux_weight * self.ce(aux, target)

        if lane is not None and lane_target is not None:
            t = lane_target.unsqueeze(1).float()
            bce = F.binary_cross_entropy_with_logits(
                lane, t, pos_weight=self.lane_pos_weight)
            loss = loss + self.lane_weight * (bce + self._binary_dice(lane, t))
        return loss


def build_model(arch: str = "mobilenetv3_unet", num_classes: int = cfg.NUM_CLASSES,
                pretrained: bool = True, lane_head: bool = False) -> nn.Module:
    if arch == "mobilenetv3_unet":
        return MobileNetV3UNet(num_classes=num_classes, pretrained=pretrained,
                               lane_head=lane_head)
    raise ValueError(f"Unknown architecture: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
