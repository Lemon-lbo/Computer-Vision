"""
SPAN (Spatial Pyramid Attention Network) — PyTorch GPU implementation.
Loads weights directly from the original PixelAttention32.h5 (Keras format).

Architecture matches get_model_1010_resize from the original IRIS0-SPAN repo:
  CombinedConv2D → 11x Conv2DSymPadding → L2norm → resize(224) →
  outlierTrans → 5x PixelAttention(shifts=1,3,9,27,81) → 5x Conv2D decoder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py


# ──────────────────────────────────────────────────────────────────────
# SRM kernels (fixed, not in H5)
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Layers
# ──────────────────────────────────────────────────────────────────────

class CombinedConv2D(nn.Module):
    """b1c1: regular(4) + SRM-fixed(9) + Bayar(3) = 16 output channels."""
    def __init__(self):
        super().__init__()
        self.regular = nn.Parameter(torch.empty(4, 3, 5, 5))
        self.bayar    = nn.Parameter(torch.empty(3, 3, 5, 5))
        srm = _srm_kernels()                 # (9,3,5,5)
        self.register_buffer('srm', srm)
        nn.init.xavier_uniform_(self.regular)
        nn.init.xavier_uniform_(self.bayar)

    def forward(self, x):
        w = torch.cat([self.regular, self.srm, self.bayar], dim=0)  # (16,3,5,5)
        x = F.pad(x, [2, 2, 2, 2], mode='reflect')
        return F.relu(F.conv2d(x, w, padding=0))


class SymConv2D(nn.Module):
    """Conv2D with symmetric (reflect) padding. Matches Conv2DSymPadding."""
    def __init__(self, in_ch, out_ch, kernel=3, activation='relu', bias=True):
        super().__init__()
        self.pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=0, bias=bias)
        self.act  = activation

    def forward(self, x):
        x = F.pad(x, [self.pad]*4, mode='reflect')
        x = self.conv(x)
        if self.act == 'relu':
            x = F.relu(x)
        return x


class PixelAttention(nn.Module):
    """
    Single SPAN attention block with 3×3 neighbourhood at dilation `shift`.
    Matches layers/PixelAttention.py exactly.
    """
    def __init__(self, D, shift=1):
        super().__init__()
        self.shift = shift
        self.D     = D
        n_p = 9  # 3×3
        # 1×1 conv weights stored as (out, in, 1, 1)
        self.K_P = nn.Parameter(torch.empty(D * n_p, D, 1, 1))
        self.V_P = nn.Parameter(torch.empty(D * n_p, D, 1, 1))
        self.Q_P = nn.Parameter(torch.empty(D,       D, 1, 1))
        # Feed-forward network
        self.ff1_k = nn.Parameter(torch.empty(D,     D,   3, 3))
        self.ff1_b = nn.Parameter(torch.zeros(D))
        self.ff2_k = nn.Parameter(torch.empty(2*D,   D,   3, 3))
        self.ff2_b = nn.Parameter(torch.zeros(2*D))
        self.ff3_k = nn.Parameter(torch.empty(D,   2*D,   3, 3))
        self.ff3_b = nn.Parameter(torch.zeros(D))

    def forward(self, x):
        B, D, H, W = x.shape
        shift = self.shift

        x_k = F.conv2d(x, self.K_P, padding=0)   # (B, D*9, H, W)
        x_v = F.conv2d(x, self.V_P, padding=0)
        x_q = F.conv2d(x, self.Q_P, padding=0)   # (B, D, H, W)

        # Pad for dilated neighbourhood extraction
        pad = shift
        x_k_pad = F.pad(x_k, [pad]*4, mode='constant', value=0)
        x_v_pad = F.pad(x_v, [pad]*4, mode='constant', value=0)
        mask     = torch.ones(B, 1, H, W, device=x.device)
        mask_pad = F.pad(mask, [pad]*4, mode='constant', value=0)

        k_list, v_list, m_list = [], [], []
        for ch_idx, (di, dj) in enumerate(
                [(i, j) for i in range(-1, 2) for j in range(-1, 2)]):
            iy = pad + di * shift
            ix = pad + dj * shift
            k_list.append(x_k_pad[:, ch_idx*D:(ch_idx+1)*D, iy:iy+H, ix:ix+W])
            v_list.append(x_v_pad[:, ch_idx*D:(ch_idx+1)*D, iy:iy+H, ix:ix+W])
            m_list.append(mask_pad[:,              :,         iy:iy+H, ix:ix+W])

        # Stack: (B, 9, D, H, W) → (B*H*W, 9, D)
        k = torch.stack(k_list, dim=1).permute(0,3,4,1,2).reshape(B*H*W, 9, D)
        v = torch.stack(v_list, dim=1).permute(0,3,4,1,2).reshape(B*H*W, 9, D)
        m = torch.stack(m_list, dim=1).permute(0,3,4,1,2).reshape(B*H*W, 9, 1)
        q = x_q.permute(0,2,3,1).reshape(B*H*W, 1, D)

        # Attention
        score = torch.bmm(k, q.transpose(1,2)) * m / 8.0  # (BHW, 9, 1)
        alpha = torch.softmax(score, dim=1)
        agg   = torch.bmm(alpha.transpose(1,2), v)         # (BHW, 1, D)
        _res  = agg.reshape(B, H, W, D).permute(0,3,1,2)   # (B, D, H, W)

        t  = x + _res   # residual
        _t = t

        # Feed-forward (symmetric padding)
        t = F.relu(F.conv2d(F.pad(t, [1]*4, 'reflect'), self.ff1_k) + self.ff1_b[:, None, None])
        t = F.relu(F.conv2d(F.pad(t, [1]*4, 'reflect'), self.ff2_k) + self.ff2_b[:, None, None])
        t = F.relu(F.conv2d(F.pad(t, [1]*4, 'reflect'), self.ff3_k) + self.ff3_b[:, None, None])
        t = _t + t       # second residual
        return t


class SPAN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor
        self.b1c1 = CombinedConv2D()               # 3→16
        self.b1c2 = SymConv2D(16,  32)
        self.b2c1 = SymConv2D(32,  64)
        self.b2c2 = SymConv2D(64,  64)
        self.b3c1 = SymConv2D(64,  128)
        self.b3c2 = SymConv2D(128, 128)
        self.b3c3 = SymConv2D(128, 128)
        self.b4c1 = SymConv2D(128, 256)
        self.b4c2 = SymConv2D(256, 256)
        self.b4c3 = SymConv2D(256, 256)
        self.b5c1 = SymConv2D(256, 256)
        self.b5c2 = SymConv2D(256, 256)
        self.transform = SymConv2D(256, 256, activation=None)

        # outlierTrans: 1×1 conv, no bias
        self.outlierTrans = nn.Conv2d(256, 32, 1, bias=False)

        # Pyramid attention (shifts: 1, 3, 9, 27, 81)
        self.pa1 = PixelAttention(32, shift=1)
        self.pa2 = PixelAttention(32, shift=3)
        self.pa3 = PixelAttention(32, shift=9)
        self.pa4 = PixelAttention(32, shift=27)
        self.pa5 = PixelAttention(32, shift=81)

        # Decoder
        self.final1 = nn.Conv2d(32, 32, 5, padding=2)
        self.final2 = nn.Conv2d(32, 16, 5, padding=2)
        self.final3 = nn.Conv2d(16,  8, 5, padding=2)
        self.final4 = nn.Conv2d( 8,  4, 5, padding=2)
        self.final5 = nn.Conv2d( 4,  1, 5, padding=2)

    def forward(self, x):
        # Feature extraction
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
        x = F.normalize(x, p=2, dim=1)            # L2 normalise channels

        # Resize to 224×224 (bilinear, matching tf.image.resize)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # outlierTrans
        x = self.outlierTrans(x)

        # Pyramid attention
        x = self.pa1(x)
        x = self.pa2(x)
        x = self.pa3(x)
        x = self.pa4(x)
        x = self.pa5(x)

        # Decoder
        x = F.relu(self.final1(x))
        x = F.relu(self.final2(x))
        x = F.relu(self.final3(x))
        x = F.relu(self.final4(x))
        x = torch.sigmoid(self.final5(x))
        return x


# ──────────────────────────────────────────────────────────────────────
# Weight loader: H5 (Keras NHWC) → PyTorch (NCHW)
# ──────────────────────────────────────────────────────────────────────

def _k2pt_conv(arr):
    """Keras conv weight (kH, kW, in, out) → PyTorch (out, in, kH, kW)."""
    return torch.from_numpy(arr.transpose(3, 2, 0, 1).copy())

def _k2pt_1x1(arr):
    """Keras 1×1 conv (1,1,in,out) → PyTorch (out,in,1,1)."""
    return torch.from_numpy(arr.transpose(3, 2, 0, 1).copy())

def _b(arr):
    return torch.from_numpy(arr.copy())


def load_weights(model, h5_path):
    with h5py.File(h5_path, 'r') as f:
        g = f['model_weights']['sigNet']

        def w(layer, name):
            return g[layer][f'{name}:0'][()]

        # ── b1c1 ──
        model.b1c1.regular.data = _k2pt_conv(w('b1c1', 'regular_kernel'))
        model.b1c1.bayar.data   = _k2pt_conv(w('b1c1', 'bayar_kernel'))

        # ── SymConv2D blocks ──
        for attr, ln in [('b1c2','b1c2'),('b2c1','b2c1'),('b2c2','b2c2'),
                          ('b3c1','b3c1'),('b3c2','b3c2'),('b3c3','b3c3'),
                          ('b4c1','b4c1'),('b4c2','b4c2'),('b4c3','b4c3'),
                          ('b5c1','b5c1'),('b5c2','b5c2'),('transform','transform')]:
            layer = getattr(model, attr).conv
            layer.weight.data = _k2pt_conv(w(ln, 'kernel'))
            layer.bias.data   = _b(w(ln, 'bias'))

        # ── outlierTrans (no bias) ──
        model.outlierTrans.weight.data = _k2pt_1x1(w('outlierTrans_new', 'kernel'))

        # ── PixelAttention layers ──
        for i, pa in enumerate([model.pa1, model.pa2, model.pa3,
                                 model.pa4, model.pa5], start=1):
            ln = f'pixel_attention_{i}'
            pa.K_P.data   = _k2pt_1x1(w(ln, 'K_P'))
            pa.V_P.data   = _k2pt_1x1(w(ln, 'V_P'))
            pa.Q_P.data   = _k2pt_1x1(w(ln, 'Q_P'))
            pa.ff1_k.data = _k2pt_conv(w(ln, 'ff1_kernel'))
            pa.ff1_b.data = _b(w(ln, 'ff1_bias'))
            pa.ff2_k.data = _k2pt_conv(w(ln, 'ff2_kernel'))
            pa.ff2_b.data = _b(w(ln, 'ff2_bias'))
            pa.ff3_k.data = _k2pt_conv(w(ln, 'ff3_kernel'))
            pa.ff3_b.data = _b(w(ln, 'ff3_bias'))

        # ── Decoder ──
        for i, layer in enumerate([model.final1, model.final2, model.final3,
                                    model.final4, model.final5], start=1):
            ln = f'final_{i}'
            layer.weight.data = _k2pt_conv(w(ln, 'kernel'))
            layer.bias.data   = _b(w(ln, 'bias'))

    print(f'[SPAN-PyTorch] Weights loaded from {h5_path}')
    return model


def build_span(h5_path, device='cuda'):
    model = SPAN()
    model = load_weights(model, h5_path)   # load on CPU first
    model = model.to(device)               # then move everything to GPU
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, rgb_uint8, device='cuda'):
    """
    rgb_uint8: HxWx3 numpy uint8 at ORIGINAL resolution.
    The model's feature extractor runs at full resolution; the feature tensor
    is resized to 224x224 internally (matching the paper's inference protocol).
    Returns: 224x224 float32 numpy mask in [0,1]
    """
    x = rgb_uint8.astype('float32') / 255.0 * 2.0 - 1.0    # [-1, 1]
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)   # (1,3,H,W)
    x = x.to(device)
    y = model(x)[0, 0].cpu().numpy()                         # (224,224)
    return y
