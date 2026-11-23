"""The 10 selected architectures.

Each builder returns an nn.Module mapping (B,3,H,W) -> logits (B, num_classes).
Feature+classical-ML models (P6, P11) also expose ``.features(x)``.
Provenance/adaptation notes are in reports/model_selection.md.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def n_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


# ============================================================ 1. P4 EfficientNetB3 + SE
class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(c, c // r), nn.Linear(c // r, c)

    def forward(self, x):
        s = x.mean((2, 3))
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class EffB3SE(nn.Module):
    """EfficientNet-B3 -> extra SE block on final feature map -> GAP -> dropout 0.5 -> FC (P4).
    SE reduction ratio (16) is our choice; P4 does not state it."""

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b3.ra2_in1k", pretrained=pretrained,
                                          num_classes=0, global_pool="")
        c = self.backbone.num_features
        self.se = SEBlock(c, 16)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x):
        f = self.se(self.backbone(x))
        return self.fc(self.drop(f.mean((2, 3))))


# ============================================================ generic timm classifier
class TimmClassifier(nn.Module):
    def __init__(self, name: str, num_classes: int, pretrained: bool = True, drop: float = 0.2, **kw):
        super().__init__()
        self.net = timm.create_model(name, pretrained=pretrained, num_classes=num_classes,
                                     drop_rate=drop, **kw)

    def forward(self, x):
        return self.net(x)

    def features(self, x):
        return self.net.forward_head(self.net.forward_features(x), pre_logits=True)


# ============================================================ 5. P6 GoogleNet + ResNet18 hybrid features
class HybridGoogLeNetResNet18(nn.Module):
    """Frozen ImageNet GoogLeNet and ResNet-18; 1000-d FC outputs of each concatenated
    to a 2000-d vector (P6). Classification is done by an SVM outside torch."""

    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        self.g = tvm.googlenet(weights=tvm.GoogLeNet_Weights.IMAGENET1K_V1).eval()
        self.r = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1).eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def features(self, x):
        return torch.cat([self.g(x), self.r(x)], 1)

    def forward(self, x):
        return self.features(x)


# ============================================================ 6. P10 lightweight two-stage CNN
def _cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class LightweightCNN(nn.Module):
    """Adapted P10 network: plain conv/BN/ReLU/maxpool trunk with one parallel branch,
    GAP, dropout, FC. ~1.56M parameters, 227x227 input (P10: 37 layers, 1.3M params).
    Exact layer list of P10 is only available as a figure, so widths are our choice
    matched to the reported parameter budget."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(_cbr(3, 32, 5, 2), nn.MaxPool2d(2))           # 57
        self.b1 = nn.Sequential(_cbr(32, 64), _cbr(64, 64), nn.MaxPool2d(2))      # 28
        self.b2 = nn.Sequential(_cbr(64, 128), _cbr(128, 128), nn.MaxPool2d(2))  # 14
        # parallel branch: 3x3 path and 1x1->5x5 path, concatenated
        self.pa = nn.Sequential(_cbr(128, 128), _cbr(128, 128))
        self.pb = nn.Sequential(_cbr(128, 64, 1), _cbr(64, 128, 5))
        self.b3 = nn.Sequential(nn.MaxPool2d(2), _cbr(256, 192), _cbr(192, 192))  # 7
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(192, num_classes)

    def embed(self, x):
        x = self.b2(self.b1(self.stem(x)))
        x = torch.cat([self.pa(x), self.pb(x)], 1)
        return self.b3(x).mean((2, 3))

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))

    def reset_head(self, num_classes: int, freeze_body: bool = True):
        """P10 stage 2: freeze trained body, replace & retrain the classification head."""
        if freeze_body:
            for p in self.parameters():
                p.requires_grad = False
        self.fc = nn.Linear(192, num_classes)


# ============================================================ 7. P12 attention-MIL ResNet-18
class AttentionMIL(nn.Module):
    """Papadopoulos et al. 2021. Bag of 64x64 patches -> ResNet-18 -> 128-d (M) ->
    additive attention (L=32): a_k = softmax(w^T tanh(V h_k)) -> z = sum a_k h_k -> FC.

    Input: full 512x512 preprocessed image. Patches are sampled inside the model:
      train: K random patches with >=50% retina; eval: regular grid, overlap 0.75.
    """

    def __init__(self, num_classes: int, patch: int = 64, k_train: int = 50,
                 overlap_eval: float = 0.75, M: int = 128, L: int = 32, pretrained: bool = True):
        super().__init__()
        self.patch, self.k_train, self.overlap_eval = patch, k_train, overlap_eval
        self.phi = timm.create_model("resnet18.tv_in1k", pretrained=pretrained, num_classes=0)
        self.proj = nn.Linear(self.phi.num_features, M)
        self.V = nn.Linear(M, L)
        self.w = nn.Linear(L, 1, bias=False)
        self.rho = nn.Linear(M, num_classes)
        self.last_attention = None

    def _grid(self, x, stride):
        p = self.patch
        patches = x.unfold(2, p, stride).unfold(3, p, stride)  # B,C,nh,nw,p,p
        B, C, nh, nw = patches.shape[:4]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, nh * nw, C, p, p)
        return patches, nh, nw

    def _retina_frac(self, patches):
        # background is black (normalised to large negative); retina = any channel non-trivial
        # we compute on un-normalised proxy: pixel considered background if all channels < -1.5
        bg = (patches < -1.5).all(2)  # B,N,p,p
        return 1.0 - bg.float().mean((2, 3))

    def forward(self, x):
        B = x.shape[0]
        if self.training:
            patches, _, _ = self._grid(x, self.patch // 4)
            frac = self._retina_frac(patches)
            bags, masks = [], []
            for b in range(B):
                ok = torch.nonzero(frac[b] >= 0.5).squeeze(1)
                if ok.numel() == 0:
                    ok = torch.arange(patches.shape[1], device=x.device)
                sel = ok[torch.randint(len(ok), (self.k_train,), device=x.device)]
                bags.append(patches[b, sel])
            bag = torch.stack(bags)  # B,K,C,p,p
            mask = torch.ones(bag.shape[:2], dtype=torch.bool, device=x.device)
        else:
            stride = max(1, int(round(self.patch * (1 - self.overlap_eval))))
            bag, nh, nw = self._grid(x, stride)
            mask = self._retina_frac(bag) >= 0.5
        K = bag.shape[1]
        h = []
        flat = bag.reshape(B * K, *bag.shape[2:])
        for s in range(0, flat.shape[0], 2048):  # chunk for memory at eval
            h.append(self.proj(self.phi(flat[s: s + 2048])))
        h = torch.cat(h).reshape(B, K, -1)
        e = self.w(torch.tanh(self.V(h))).squeeze(-1)
        e = e.masked_fill(~mask, float("-inf"))
        a = torch.softmax(e, 1)
        self.last_attention = a.detach()
        z = (a.unsqueeze(-1) * h).sum(1)
        return self.rho(z)


# ============================================================ 9. P15 multi-scale CNN + LSTM + attention
class MultiScaleCNNLSTM(nn.Module):
    """TAHDL as described in the survey: three parallel conv paths (3x3, 5x5, 7x7) ->
    concat + maxpool -> 2-layer LSTM (256, 128) -> softmax attention -> FC.

    Adaptation (documented): P15 does not say how a single image becomes a sequence.
    We read the final feature map in raster order (each spatial position = one time step).
    """

    def __init__(self, num_classes: int, width: int = 48):
        super().__init__()

        def path(k):
            return nn.Sequential(
                _cbr(3, width, k, 2), nn.MaxPool2d(2),            # 56
                _cbr(width, width * 2, k), nn.MaxPool2d(2),       # 28
                _cbr(width * 2, width * 2, k), nn.MaxPool2d(2),   # 14
            )
        self.paths = nn.ModuleList([path(3), path(5), path(7)])
        self.pool = nn.MaxPool2d(2)                               # 7x7 = 49 steps
        c = 3 * width * 2
        self.lstm1 = nn.LSTM(c, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True)
        self.att = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        f = self.pool(torch.cat([p(x) for p in self.paths], 1))  # B,C,h,w
        seq = f.flatten(2).transpose(1, 2)                         # B,T,C
        s, _ = self.lstm1(seq)
        s, _ = self.lstm2(self.drop(s))
        a = torch.softmax(self.att(s).squeeze(-1), 1)
        return self.fc(self.drop((a.unsqueeze(-1) * s).sum(1)))


# ============================================================ registry
# name -> (builder(num_classes), default input size, enhancement, paper)
REGISTRY = {
    "effb3_se":      (lambda n: EffB3SE(n), 384, "ben_graham", "P4"),
    "inception_v3":  (lambda n: TimmClassifier("inception_v3.tv_in1k", n, drop=0.2), 384, "raw", "P7"),
    "densenet169":   (lambda n: TimmClassifier("densenet169.tv_in1k", n, drop=0.2), 384, "raw", "P14"),
    "resnet50_rf":   (lambda n: TimmClassifier("resnet50.tv_in1k", n, drop=0.2), 384, "raw", "P11"),
    "hybrid_svm":    (lambda n: HybridGoogLeNetResNet18(), 224, "raw", "P6"),
    "light2stage":   (lambda n: LightweightCNN(n), 227, "raw", "P10"),
    "mil_resnet18":  (lambda n: AttentionMIL(n), 512, "graham_rim", "P12"),
    "p8_vgg16":      (lambda n: TimmClassifier("vgg16_bn.tv_in1k", n, drop=0.5), 384, "agc_p8", "P8"),
    "mscnn_lstm":    (lambda n: MultiScaleCNNLSTM(n), 224, "clahe", "P15"),
    "vit_b16":       (lambda n: TimmClassifier("vit_base_patch16_224.augreg2_in21k_ft_in1k", n,
                                               drop=0.1), 224, "raw", "P14"),
}


def build(name: str, num_classes: int) -> nn.Module:
    return REGISTRY[name][0](num_classes)
