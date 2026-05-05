"""
ManTraNet (CVPR 2019) -- PyTorch GPU implementation.
Loads weights from the original ManTraNet_Ptrain4.h5 (Keras/HDF5 format).

Architecture (create_manTraNet_model in modelCore.py):
  Featex (shared VGG backbone, identical to SPAN) ->
  outlierTrans (256->64, 1x1 conv, unit-norm, no bias) ->
  BatchNorm2d (no affine, just running stats) ->
  NestedWindowAvgDev ([7,15,31] + global window, minus_original=True) ->
  GlobalStd normalisation  ->
  |dev / std|  ->
  ConvLSTM2D (hidden=8, kernel=7x7, tanh + hard-sigmoid) ->
  Conv2d(1, 7x7, sigmoid)

The feature extractor runs at the original image resolution; the output
prediction map is returned at the same resolution and must be resized
externally (e.g. to 224x224 to match ground-truth masks).

Reference: Wu et al., ManTra-Net: Manipulation Tracing Network For
Detection And Localization of Image Forgeries With Anomalous Features,
CVPR 2019.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py


# ── shared building blocks (identical to SPAN's feature extractor) ────

def _srm_kernels():
    srm1 = np.zeros([5, 5], dtype='float32')
    srm1[1:-1, 1:-1] = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
    srm1 /= 4.
    srm2 = np.array([[-1, 2,-2, 2,-1],[2,-6, 8,-6, 2],[-2, 8,-12, 8,-2],
                      [2,-6, 8,-6, 2],[-1, 2,-2, 2,-1]], dtype='float32') / 12.
    srm3 = np.zeros([5, 5], dtype='float32')
    srm3[2, 1:-1] = np.array([1, -2, 1])
    srm3 /= 2.
    kernels = []
    for srm in [srm1, srm2, srm3]:
        for ch in range(3):
            k = np.zeros([5, 5, 3], dtype='float32')
            k[:, :, ch] = srm
            kernels.append(k)
    arr = np.stack(kernels, axis=-1)          # (5,5,3,9)
    return torch.from_numpy(arr.transpose(3, 2, 0, 1))  # (9,3,5,5)


class CombinedConv2D(nn.Module):
    """b1c1: regular(4) + SRM-fixed(9) + Bayar(3) = 16 output channels."""
    def __init__(self):
        super().__init__()
        self.regular = nn.Parameter(torch.empty(4, 3, 5, 5))
        self.bayar   = nn.Parameter(torch.empty(3, 3, 5, 5))
        self.register_buffer('srm', _srm_kernels())
        nn.init.xavier_uniform_(self.regular)
        nn.init.xavier_uniform_(self.bayar)

    def forward(self, x):
        w = torch.cat([self.regular, self.srm, self.bayar], dim=0)  # (16,3,5,5)
        return F.relu(F.conv2d(F.pad(x, [2,2,2,2], 'reflect'), w))


class SymConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, activation='relu', bias=True):
        super().__init__()
        self.pad  = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=0, bias=bias)
        self.act  = activation

    def forward(self, x):
        x = F.pad(x, [self.pad]*4, 'reflect')
        x = self.conv(x)
        if self.act == 'relu':
            x = F.relu(x)
        return x


# ── ManTraNet-specific detection layers ───────────────────────────────

class GlobalStd2D(nn.Module):
    """Per-channel std over spatial dims, clamped by a learnable minimum."""
    def __init__(self):
        super().__init__()
        # min_std shape: (1, 64, 1, 1) in NCHW → loaded from H5 (1,1,1,64) NHWC
        self.min_std = nn.Parameter(torch.full((1, 64, 1, 1), 1e-5))

    def forward(self, x):
        # Keras uses population std (unbiased=False), keepdim over H,W
        sigma = x.std(dim=(2, 3), unbiased=False, keepdim=True)
        min_val = 1e-6 + self.min_std          # matches Keras: min_std_val/10 + min_std
        return sigma.clamp(min=0.) + min_val.abs()


class _HardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (0.2 * x + 0.5).clamp(0., 1.)
    @staticmethod
    def backward(ctx, g):
        return g  # straight-through

def hard_sigmoid(x):
    return _HardSigmoid.apply(x)


class ConvLSTM2DCell(nn.Module):
    """Single ConvLSTM2D cell (Keras gate order: i, f, c, o)."""
    def __init__(self, in_ch, hidden, kernel=7):
        super().__init__()
        pad = kernel // 2
        # input→gates (kernel weight from H5: (kH,kW,in_ch,4*hidden))
        self.W = nn.Conv2d(in_ch,    4 * hidden, kernel, padding=pad, bias=True)
        # hidden→gates (recurrent weight)
        self.U = nn.Conv2d(hidden,   4 * hidden, kernel, padding=pad, bias=False)
        self.hidden = hidden

    def forward(self, x, h, c):
        gates = self.W(x) + self.U(h)          # (B, 4*hidden, H, W)
        H = self.hidden
        gi, gf, gc, go = gates[:,:H], gates[:,H:2*H], gates[:,2*H:3*H], gates[:,3*H:]
        i = hard_sigmoid(gi)
        f = hard_sigmoid(gf)
        z = torch.tanh(gc)
        o = hard_sigmoid(go)
        c_new = f * c + i * z
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


# ── Full ManTraNet model ──────────────────────────────────────────────

class ManTraNet(nn.Module):
    WINDOW_SIZES = [7, 15, 31]   # ManTraNet_Ptrain4 setting

    def __init__(self):
        super().__init__()
        # Feature extractor (identical VGG backbone to SPAN)
        self.b1c1      = CombinedConv2D()
        self.b1c2      = SymConv2D(16,  32)
        self.b2c1      = SymConv2D(32,  64)
        self.b2c2      = SymConv2D(64,  64)
        self.b3c1      = SymConv2D(64,  128)
        self.b3c2      = SymConv2D(128, 128)
        self.b3c3      = SymConv2D(128, 128)
        self.b4c1      = SymConv2D(128, 256)
        self.b4c2      = SymConv2D(256, 256)
        self.b4c3      = SymConv2D(256, 256)
        self.b5c1      = SymConv2D(256, 256)
        self.b5c2      = SymConv2D(256, 256)
        self.transform = SymConv2D(256, 256, activation=None)

        # Detection head
        self.outlierTrans = nn.Conv2d(256, 64, 1, bias=False)
        self.bnorm        = nn.BatchNorm2d(64, affine=False)
        self.glbStd       = GlobalStd2D()
        self.lstm_cell    = ConvLSTM2DCell(in_ch=64, hidden=8, kernel=7)
        self.pred         = nn.Conv2d(8, 1, kernel_size=7, padding=3)

    def _featex(self, x):
        """Run VGG feature extractor at original resolution."""
        x = self.b1c1(x)
        x = self.b1c2(x)
        x = self.b2c1(x)
        x = self.b2c2(x)
        x = self.b3c1(x)
        x = self.b3c2(x)
        x = self.b3c3(x)
        x = self.b4c1(x)
        x = self.b4c2(x)
        x = self.b4c3(x)
        x = self.b5c1(x)
        x = self.b5c2(x)
        x = self.transform(x)
        return F.normalize(x, p=2, dim=1)      # L2 normalise channels

    def _nested_avg_dev(self, x):
        """
        Compute (local_avg - x) for each window in WINDOW_SIZES plus global.
        Returns list of 4 tensors each (B, 64, H, W).
        """
        devs = []
        for w in self.WINDOW_SIZES:
            # zero-pad to keep spatial size, then avg_pool (count_include_pad=False
            # gives correct per-position counts, matching the Keras integral-image impl)
            avg = F.avg_pool2d(x, kernel_size=w, stride=1,
                               padding=w // 2, count_include_pad=False)
            devs.append(avg - x)
        # global average
        mu = x.mean(dim=(2, 3), keepdim=True).expand_as(x)
        devs.append(mu - x)
        return devs   # 4 elements

    def forward(self, x):
        # 1. feature extraction at original resolution
        f = self._featex(x)                    # (B, 256, H, W)

        # 2. outlierTrans + BN
        f = self.outlierTrans(f)               # (B, 64, H, W)
        f = self.bnorm(f)

        # 3. nested window deviations
        devs = self._nested_avg_dev(f)         # 4 × (B, 64, H, W)

        # 4. normalise by global std
        sigma = self.glbStd(f)                 # (B, 64, 1, 1)
        devs  = [torch.abs(d / sigma) for d in devs]

        # 5. ConvLSTM2D over 4 time steps (h0=c0=0)
        B, C, H, W = devs[0].shape
        h = torch.zeros(B, 8, H, W, device=x.device)
        c = torch.zeros(B, 8, H, W, device=x.device)
        for t in range(4):
            h, c = self.lstm_cell(devs[t], h, c)

        # 6. prediction
        return torch.sigmoid(self.pred(h))     # (B, 1, H, W)


# ── Weight loader ─────────────────────────────────────────────────────

def _k2pt_conv(arr):
    """Keras (kH,kW,in,out) → PyTorch (out,in,kH,kW)."""
    return torch.from_numpy(arr.transpose(3, 2, 0, 1).copy())

def _k2pt_1x1(arr):
    return torch.from_numpy(arr.transpose(3, 2, 0, 1).copy())

def _b(arr):
    return torch.from_numpy(arr.copy())


def load_mantranet_weights(model, h5_path):
    with h5py.File(h5_path, 'r') as f:
        fe = f['Featex']

        # ── Feature extractor ──
        def fw(group_name, wname):
            return fe[f'{group_name}/{wname}:0'][()]

        model.b1c1.regular.data = _k2pt_conv(fw('b1c1_6', 'regular_kernel'))
        model.b1c1.bayar.data   = _k2pt_conv(fw('b1c1_6', 'bayar_kernel'))

        for attr, ln in [('b1c2','b1c2_6'), ('b2c1','b2c1_6'), ('b2c2','b2c2_6'),
                          ('b3c1','b3c1_6'), ('b3c2','b3c2_6'), ('b3c3','b3c3_6'),
                          ('b4c1','b4c1_6'), ('b4c2','b4c2_6'), ('b4c3','b4c3_6'),
                          ('b5c1','b5c1_6'), ('b5c2','b5c2_6'), ('transform','transform_6')]:
            layer = getattr(model, attr).conv
            layer.weight.data = _k2pt_conv(fw(ln, 'kernel'))
            layer.bias.data   = _b(fw(ln, 'bias'))

        # ── outlierTrans ──
        model.outlierTrans.weight.data = _k2pt_1x1(
            f['outlierTrans']['outlierTrans_6']['kernel:0'][()])

        # ── BatchNorm (no affine, only running stats) ──
        model.bnorm.running_mean.copy_(
            _b(f['bnorm']['bnorm_6']['moving_mean:0'][()]))
        model.bnorm.running_var.copy_(
            _b(f['bnorm']['bnorm_6']['moving_variance:0'][()]))

        # ── GlobalStd2D min_std (NHWC → NCHW) ──
        ms = f['glbStd']['glbStd_6']['min_std:0'][()]   # (1,1,1,64) NHWC
        model.glbStd.min_std.data = torch.from_numpy(
            ms.transpose(0, 3, 1, 2).copy())             # (1,64,1,1)

        # ── ConvLSTM2D ──
        # Keras kernel:           (7,7, in_ch,   4*hidden) — gates: i,f,c,o
        # Keras recurrent_kernel: (7,7, hidden,  4*hidden)
        # Keras bias:             (4*hidden,)
        # We split along last axis into 4 equal parts [i,f,c,o]
        lstm_kern = f['cLSTM']['cLSTM_6']['kernel:0'][()]            # (7,7,64,32)
        lstm_rec  = f['cLSTM']['cLSTM_6']['recurrent_kernel:0'][()]  # (7,7,8,32)
        lstm_bias = f['cLSTM']['cLSTM_6']['bias:0'][()]              # (32,)

        # PyTorch needs (out, in, kH, kW) — split gate channels (dim 0 after transpose)
        W = torch.from_numpy(lstm_kern.transpose(3, 2, 0, 1).copy())  # (32,64,7,7)
        U = torch.from_numpy(lstm_rec.transpose(3, 2, 0, 1).copy())   # (32,8,7,7)
        b = torch.from_numpy(lstm_bias.copy())                         # (32,)

        model.lstm_cell.W.weight.data = W
        model.lstm_cell.W.bias.data   = b
        model.lstm_cell.U.weight.data = U

        # ── Prediction layer ──
        model.pred.weight.data = _k2pt_conv(
            f['pred']['pred_6']['kernel:0'][()])    # (7,7,8,1) → (1,8,7,7)
        model.pred.bias.data   = _b(f['pred']['pred_6']['bias:0'][()])

    print(f'[ManTraNet-PyTorch] Weights loaded from {h5_path}')
    return model


def build_mantranet(h5_path, device='cuda'):
    model = ManTraNet()
    model = load_mantranet_weights(model, h5_path)
    model = model.to(device)
    model.eval()
    return model


# ── Inference helper ──────────────────────────────────────────────────

@torch.no_grad()
def predict(model, rgb_uint8, device='cuda'):
    """
    rgb_uint8 : H x W x 3 numpy uint8 at ORIGINAL resolution.
    ManTraNet runs at full resolution (no internal resize).
    Returns : 224 x 224 float32 numpy prediction map in [0, 1].
    """
    import cv2
    x = rgb_uint8.astype('float32') / 255.0 * 2.0 - 1.0    # [-1, 1]
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    y = model(x)[0, 0].cpu().numpy()                         # (H, W)
    return cv2.resize(y, (224, 224), interpolation=cv2.INTER_LINEAR)
