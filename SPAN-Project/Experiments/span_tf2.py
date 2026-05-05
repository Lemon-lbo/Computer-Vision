"""
TF2-compatible SPAN (Spatial Pyramid Attention Network) model.
Loads pretrained weights from PixelAttention32.h5 (original TF1/Keras format).

Architecture:
  Feature Extractor: CombinedConv2D → 5x Conv2DSymPadding → transform → L2 norm
  SPAN head: resize(224x224) → outlierTrans → 5x PixelAttention → 5x Conv2D decoder
"""

import numpy as np
import tensorflow as tf
import h5py


# ============================================================
# SRM (Spatial Rich Model) kernel construction
# ============================================================

def _build_srm_kernel_numpy():
    """Build fixed 5x5 SRM kernels: 9 total = 3 SRM bases × 3 colour channels."""
    srm1 = np.zeros([5, 5], dtype='float32')
    srm1[1:-1, 1:-1] = np.array([[-1, 2, -1],
                                   [ 2,-4,  2],
                                   [-1, 2, -1]])
    srm1 /= 4.

    srm2 = np.array([[-1,  2, -2,  2, -1],
                      [ 2, -6,  8, -6,  2],
                      [-2,  8,-12,  8, -2],
                      [ 2, -6,  8, -6,  2],
                      [-1,  2, -2,  2, -1]], dtype='float32')
    srm2 /= 12.

    srm3 = np.zeros([5, 5], dtype='float32')
    srm3[2, 1:-1] = np.array([1, -2, 1])
    srm3 /= 2.

    kernel = []
    for srm in [srm1, srm2, srm3]:
        for ch in range(3):
            this_ch_kernel = np.zeros([5, 5, 3], dtype='float32')
            this_ch_kernel[:, :, ch] = srm
            kernel.append(this_ch_kernel)
    return np.stack(kernel, axis=-1)   # (5, 5, 3, 9)


# ============================================================
# Custom Keras Layers
# ============================================================

class CombinedConv2DLayer(tf.keras.layers.Layer):
    """
    b1c1: first layer combining regular + SRM (fixed) + Bayar kernels.
    Applies 5x5 symmetric-padded conv with 16 output channels (4+9+3).
    """
    def __init__(self, n_regular=4, **kwargs):
        super().__init__(**kwargs)
        self.n_regular = n_regular

    def build(self, input_shape):
        # trainable kernels
        self.regular_kernel = self.add_weight(
            name='regular_kernel', shape=(5, 5, 3, self.n_regular),
            initializer='glorot_uniform', trainable=True)
        self.bayar_kernel = self.add_weight(
            name='bayar_kernel', shape=(5, 5, 3, 3),
            initializer='glorot_uniform', trainable=True)
        # fixed SRM kernel – not stored in H5, recomputed
        srm_np = _build_srm_kernel_numpy()          # (5, 5, 3, 9)
        self.srm_kernel = tf.constant(srm_np, dtype=tf.float32)
        self.built = True

    def call(self, x):
        kernel = tf.concat([self.regular_kernel,
                             self.srm_kernel,
                             self.bayar_kernel], axis=-1)   # (5,5,3,16)
        # symmetric padding for 5x5 kernel (pad=2)
        padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='SYMMETRIC')
        out = tf.nn.conv2d(padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
        return tf.nn.relu(out)

    def get_config(self):
        cfg = super().get_config()
        cfg['n_regular'] = self.n_regular
        return cfg


class Conv2DSymPaddingLayer(tf.keras.layers.Layer):
    """
    Conv2D with symmetric (mirror) padding instead of zero-padding.
    Matches the original Conv2DSymPadding from modelCore.py.
    """
    def __init__(self, filters, kernel_size, activation=None,
                 use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = (kernel_size, kernel_size) \
            if isinstance(kernel_size, int) else tuple(kernel_size)
        self.pad_h = self.kernel_size[0] // 2
        self.pad_w = self.kernel_size[1] // 2
        self.activation_fn = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        in_ch = int(input_shape[-1])
        self.kernel = self.add_weight(
            name='kernel',
            shape=(*self.kernel_size, in_ch, self.filters),
            initializer='glorot_uniform', trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias', shape=(self.filters,),
                initializer='zeros', trainable=True)
        self.built = True

    def call(self, x):
        padded = tf.pad(x,
                        [[0, 0],
                         [self.pad_h, self.pad_h],
                         [self.pad_w, self.pad_w],
                         [0, 0]],
                        mode='SYMMETRIC')
        out = tf.nn.conv2d(padded, self.kernel,
                           strides=[1, 1, 1, 1], padding='VALID')
        if self.use_bias:
            out = out + self.bias
        if self.activation_fn is not None:
            out = self.activation_fn(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(filters=self.filters, kernel_size=self.kernel_size,
                        use_bias=self.use_bias))
        return cfg


class PixelAttentionLayer(tf.keras.layers.Layer):
    """
    SPAN's core attention module with a 3x3 spatial neighbourhood
    sampled at dilation `shift`.  Includes a 3-layer feed-forward net.
    Matches PixelAttention.py from the original repo.
    """
    def __init__(self, shift=1, useBN=False, useRes=True, **kwargs):
        super().__init__(**kwargs)
        self.shift  = shift
        self.useBN  = useBN
        self.useRes = useRes

    def build(self, input_shape):
        D   = int(input_shape[-1])
        n_p = 9                           # 3×3 kernel_range
        # Q / K / V projections  (1×1 conv weights)
        self.K_P = self.add_weight(name='K_P',      shape=(1, 1, D, D * n_p),
                                   initializer='glorot_uniform', trainable=True)
        self.V_P = self.add_weight(name='V_P',      shape=(1, 1, D, D * n_p),
                                   initializer='glorot_uniform', trainable=True)
        self.Q_P = self.add_weight(name='Q_P',      shape=(1, 1, D, D),
                                   initializer='glorot_uniform', trainable=True)
        # Feed-forward network weights (matching original naming exactly)
        self.ff1_kernel = self.add_weight(name='ff1_kernel', shape=(3, 3, D, D),
                                          initializer='glorot_uniform', trainable=True)
        self.ff1_bais   = self.add_weight(name='ff1_bias',   shape=(D,),
                                          initializer='zeros',          trainable=True)
        self.ff2_kernel = self.add_weight(name='ff2_kernel', shape=(3, 3, D, 2 * D),
                                          initializer='glorot_uniform', trainable=True)
        self.ff2_bais   = self.add_weight(name='ff2_bias',   shape=(2 * D,),
                                          initializer='zeros',          trainable=True)
        self.ff3_kernel = self.add_weight(name='ff3_kernel', shape=(3, 3, 2 * D, D),
                                          initializer='glorot_uniform', trainable=True)
        self.ff3_bais   = self.add_weight(name='ff3_bias',   shape=(D,),
                                          initializer='zeros',          trainable=True)
        self.D = D
        self.built = True

    def call(self, x):
        h_half = 1          # kernel_range = [3, 3]
        w_half = 1
        D      = self.D
        s      = tf.shape(x)   # [B, H, W, D] – runtime shape

        # Project keys / values / query
        x_k = tf.nn.conv2d(x, self.K_P, strides=[1, 1, 1, 1], padding='SAME')
        x_v = tf.nn.conv2d(x, self.V_P, strides=[1, 1, 1, 1], padding='SAME')
        x_q = tf.nn.conv2d(x, self.Q_P, strides=[1, 1, 1, 1], padding='SAME')

        # Pad to allow shifted neighbourhood extraction
        pad_amt = h_half * self.shift
        paddings = [[0, 0], [pad_amt, pad_amt], [pad_amt, pad_amt], [0, 0]]
        x_k_pad = tf.pad(x_k, paddings, 'CONSTANT')
        x_v_pad = tf.pad(x_v, paddings, 'CONSTANT')

        # Validity mask (1 inside image, 0 in padding)
        mask_x   = tf.ones_like(x[..., :1])             # (B, H, W, 1)
        mask_pad = tf.pad(mask_x, paddings, 'CONSTANT') # (B, H+2p, W+2p, 1)

        c = pad_amt   # centre offset after padding
        k_ls, v_ls, m_ls = [], [], []
        ch_idx = 0
        for i in range(-h_half, h_half + 1):
            for j in range(-w_half, w_half + 1):
                iy = c + i * self.shift
                ix = c + j * self.shift
                k_t = x_k_pad[:, iy:iy + s[1], ix:ix + s[2],
                               ch_idx * D:(ch_idx + 1) * D]
                v_t = x_v_pad[:, iy:iy + s[1], ix:ix + s[2],
                               ch_idx * D:(ch_idx + 1) * D]
                m_t = mask_pad[:, iy:iy + s[1], ix:ix + s[2], :]
                k_ls.append(k_t)
                v_ls.append(v_t)
                m_ls.append(m_t)
                ch_idx += 1

        # Stack neighbours: (B, H, W, 9, D)
        k_stack = tf.stack(k_ls, axis=3)
        v_stack = tf.stack(v_ls, axis=3)
        m_stack = tf.stack(m_ls, axis=3)   # (B, H, W, 9, 1)

        BHW = s[0] * s[1] * s[2]
        k     = tf.reshape(k_stack, [BHW,  9, D])
        v     = tf.reshape(v_stack, [BHW,  9, D])
        m_vec = tf.reshape(m_stack, [BHW,  9, 1])
        q     = tf.reshape(x_q,     [BHW,  1, D])

        # Attention score (masked, scaled by 1/8)
        score = tf.matmul(k, q, transpose_b=True) * m_vec / 8.0   # (BHW, 9, 1)
        alpha = tf.nn.softmax(score, axis=1)                        # (BHW, 9, 1)

        # Weighted aggregation
        agg  = tf.matmul(alpha, v, transpose_a=True)               # (BHW, 1, D)
        _res = tf.reshape(agg, [s[0], s[1], s[2], D])

        t  = (x + _res) if self.useRes else _res
        _t = t

        # Feed-forward network
        t = tf.nn.relu(tf.nn.conv2d(t, self.ff1_kernel,
                                    strides=[1,1,1,1], padding='SAME') + self.ff1_bais)
        t = tf.nn.relu(tf.nn.conv2d(t, self.ff2_kernel,
                                    strides=[1,1,1,1], padding='SAME') + self.ff2_bais)
        t = tf.nn.relu(tf.nn.conv2d(t, self.ff3_kernel,
                                    strides=[1,1,1,1], padding='SAME') + self.ff3_bais)
        if self.useRes:
            t = _t + t
        return t

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(shift=self.shift, useBN=self.useBN, useRes=self.useRes))
        return cfg


# ============================================================
# Model builder
# ============================================================

def build_span_model():
    """
    Build the SPAN model (get_model_1010_resize variant).
    All layer names match those in PixelAttention32.h5 exactly.
    """
    img_in = tf.keras.Input(shape=(None, None, 3), name='img_in')

    # ---- Feature extractor ----
    x = CombinedConv2DLayer(n_regular=4, name='b1c1')(img_in)       # (H,W,16)
    x = Conv2DSymPaddingLayer(32,  (3,3), activation='relu',
                               name='b1c2')(x)                        # (H,W,32)
    x = Conv2DSymPaddingLayer(64,  (3,3), activation='relu',
                               name='b2c1')(x)                        # (H,W,64)
    x = Conv2DSymPaddingLayer(64,  (3,3), activation='relu',
                               name='b2c2')(x)                        # (H,W,64)
    x = Conv2DSymPaddingLayer(128, (3,3), activation='relu',
                               name='b3c1')(x)                        # (H,W,128)
    x = Conv2DSymPaddingLayer(128, (3,3), activation='relu',
                               name='b3c2')(x)                        # (H,W,128)
    x = Conv2DSymPaddingLayer(128, (3,3), activation='relu',
                               name='b3c3')(x)                        # (H,W,128)
    x = Conv2DSymPaddingLayer(256, (3,3), activation='relu',
                               name='b4c1')(x)                        # (H,W,256)
    x = Conv2DSymPaddingLayer(256, (3,3), activation='relu',
                               name='b4c2')(x)                        # (H,W,256)
    x = Conv2DSymPaddingLayer(256, (3,3), activation='relu',
                               name='b4c3')(x)                        # (H,W,256)
    x = Conv2DSymPaddingLayer(256, (3,3), activation='relu',
                               name='b5c1')(x)                        # (H,W,256)
    x = Conv2DSymPaddingLayer(256, (3,3), activation='relu',
                               name='b5c2')(x)                        # (H,W,256)

    # transform conv (no activation) + L2 normalise
    x = Conv2DSymPaddingLayer(256, (3,3), activation=None,
                               name='transform')(x)                   # (H,W,256)
    rf = tf.keras.layers.Lambda(
        lambda t: tf.linalg.l2_normalize(t, axis=-1), name='L2')(x)

    # ---- Resize feature map to 224×224 ----
    rf = tf.keras.layers.Lambda(
        lambda t: tf.image.resize(t, (224, 224)), name='resize')(rf)

    # ---- outlierTrans: 1×1 conv, no bias, unit-norm constraint ----
    rf = tf.keras.layers.Conv2D(
        32, (1, 1), use_bias=False,
        kernel_constraint=tf.keras.constraints.unit_norm(axis=-2),
        padding='same', name='outlierTrans_new')(rf)

    # ---- Pyramid of PixelAttention layers (shifts: 1,3,9,27,81) ----
    t = rf
    for i, step in enumerate([1, 3, 9, 27, 81], start=1):
        t = PixelAttentionLayer(shift=step, useBN=False, useRes=True,
                                name=f'pixel_attention_{i}')(t)

    # ---- Decoder (standard Conv2D with zero-padding) ----
    t = tf.keras.layers.Conv2D(32, (5,5), activation='relu',
                                padding='same', name='final_1')(t)
    t = tf.keras.layers.Conv2D(16, (5,5), activation='relu',
                                padding='same', name='final_2')(t)
    t = tf.keras.layers.Conv2D( 8, (5,5), activation='relu',
                                padding='same', name='final_3')(t)
    t = tf.keras.layers.Conv2D( 4, (5,5), activation='relu',
                                padding='same', name='final_4')(t)
    pred = tf.keras.layers.Conv2D(1, (5,5), activation='sigmoid',
                                   padding='same', name='final_5')(t)

    model = tf.keras.Model(inputs=img_in, outputs=pred, name='sigNet')
    return model


# ============================================================
# Weight loader (manual H5 → layer.set_weights)
# ============================================================

def load_span_weights(model, h5_path):
    """
    Load pretrained weights from the original TF1/Keras H5 file.
    Uses explicit weight mapping to avoid TF1 vs TF2 H5 schema differences.
    """
    with h5py.File(h5_path, 'r') as f:
        wg = f['model_weights']['sigNet']

        def _w(layer_name, weight_name):
            """Read a single weight tensor from the H5 group."""
            return wg[layer_name][f'{weight_name}:0'][()]

        # b1c1  (CombinedConv2DLayer: regular_kernel, bayar_kernel – in build order)
        model.get_layer('b1c1').set_weights([
            _w('b1c1', 'regular_kernel'),
            _w('b1c1', 'bayar_kernel'),
        ])

        # Conv2DSymPaddingLayer blocks  (kernel, bias in build order)
        for ln in ['b1c2', 'b2c1', 'b2c2',
                   'b3c1', 'b3c2', 'b3c3',
                   'b4c1', 'b4c2', 'b4c3',
                   'b5c1', 'b5c2',
                   'transform']:
            model.get_layer(ln).set_weights([
                _w(ln, 'kernel'),
                _w(ln, 'bias'),
            ])

        # outlierTrans_new  (Conv2D, no bias)
        model.get_layer('outlierTrans_new').set_weights([
            _w('outlierTrans_new', 'kernel'),
        ])

        # PixelAttention layers  (build order: K_P, V_P, Q_P, ff1_k, ff1_b, ff2_k, ff2_b, ff3_k, ff3_b)
        for i in range(1, 6):
            ln = f'pixel_attention_{i}'
            model.get_layer(ln).set_weights([
                _w(ln, 'K_P'),
                _w(ln, 'V_P'),
                _w(ln, 'Q_P'),
                _w(ln, 'ff1_kernel'),
                _w(ln, 'ff1_bias'),
                _w(ln, 'ff2_kernel'),
                _w(ln, 'ff2_bias'),
                _w(ln, 'ff3_kernel'),
                _w(ln, 'ff3_bias'),
            ])

        # Decoder Conv2D layers  (kernel, bias)
        for i in range(1, 6):
            ln = f'final_{i}'
            model.get_layer(ln).set_weights([
                _w(ln, 'kernel'),
                _w(ln, 'bias'),
            ])

    print(f"[SPAN] Weights loaded from {h5_path}")
    return model


# ============================================================
# Inference helper
# ============================================================

def predict_mask(model, rgb_image):
    """
    Run SPAN on a uint8 RGB image (H×W×3 numpy array).
    Returns a float32 prediction mask of shape (H, W) in [0, 1].
    """
    # Normalise to [-1, 1]
    x = (rgb_image.astype('float32') / 255.0) * 2.0 - 1.0
    x = x[np.newaxis]                          # (1, H, W, 3)
    y = model.predict(x, verbose=0)[0, ..., 0] # (224, 224)
    return y


def postprocess(mask, target_shape):
    """
    Resize the model output (224×224) back to target_shape (H, W).
    Uses nearest-neighbour to match original repo postprocess.
    """
    import cv2
    nh, nw = target_shape
    return cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
