"""
SPAN + ManTraNet evaluation on Columbia dataset -- matches paper Table 5
(Robustness Analysis).

Speed optimisations:
  1. Both models run on the same image in one loop (single disk read per image)
  2. All images pre-loaded into RAM; perturbation runs cost zero I/O
  3. Shared preprocessing tensor sent to GPU once per image

Metrics (per paper):
  Pixel AUC : pixel-level ROC-AUC per spliced image, averaged
  F1 (%)    : pixel-level F1 at 0.5 threshold, best polarity, averaged
"""

import os, sys, glob, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from span_pytorch      import build_span,      SPAN
from mantranet_pytorch import build_mantranet,  ManTraNet

DATASET_DIR  = './columbia_dataset'
SPAN_WEIGHTS = './IRIS0-SPAN/PixelAttention32.h5'
MT_WEIGHTS   = './ManTraNet_Ptrain4.h5'
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'


# -- perturbations (Table 5 order) ------------------------------------

def apply_resize(img, scale):
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    return cv2.resize(cv2.resize(img, (nw, nh), cv2.INTER_LINEAR), (w, h), cv2.INTER_LINEAR)

def apply_blur(img, k):
    kk = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(img, (kk, kk), 0)

def apply_noise(img, sigma):
    noise = np.random.randn(*img.shape).astype('float32') * sigma
    return np.clip(img.astype('float32') + noise, 0, 255).astype('uint8')

def apply_jpeg(img, quality):
    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

PERTURBATIONS = [
    ('No manipulation',               None),
    ('Resize (0.78x)',                lambda x: apply_resize(x, 0.78)),
    ('Resize (0.25x)',                lambda x: apply_resize(x, 0.25)),
    ('No manipulation',               None),
    ('GaussianBlur (kernal size=3)',  lambda x: apply_blur(x, 3)),
    ('GaussianBlur (kernal size=15)', lambda x: apply_blur(x, 15)),
    ('No manipulation',               None),
    ('GaussianNoise (sigma=3)',       lambda x: apply_noise(x, 3)),
    ('GaussianNoise (sigma=15)',      lambda x: apply_noise(x, 15)),
    ('No manipulation',               None),
    ('JPEGCompress (quality=100)',    lambda x: apply_jpeg(x, 100)),
    ('JPEGCompress (quality=50)',     lambda x: apply_jpeg(x, 50)),
]


# -- metrics ----------------------------------------------------------

def _roc_auc(y_true, y_score):
    y_true  = np.asarray(y_true,  'float32').ravel()
    y_score = np.asarray(y_score, 'float32').ravel()
    idx  = np.argsort(-y_score)
    yt, ys = y_true[idx], y_score[idx]
    dist = np.concatenate([np.where(np.diff(ys))[0], [len(yt) - 1]])
    tps  = np.concatenate([[0], np.cumsum(yt)[dist]])
    fps  = np.concatenate([[0], dist + 1 - np.cumsum(yt)[dist]])
    fpr  = fps / (fps[-1] + 1e-10)
    tpr  = tps / (tps[-1] + 1e-10)
    _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return float(_trapz(tpr, fpr))

def _f1(y_true, y_pred):
    yt = np.asarray(y_true).ravel().astype(bool)
    yp = np.asarray(y_pred).ravel().astype(bool)
    tp = np.logical_and(yt,  yp).sum()
    fp = np.logical_and(~yt, yp).sum()
    fn = np.logical_and(yt, ~yp).sum()
    d  = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else 0.0

def compute_metrics(gt_list, pred_list):
    auc_px, f1_px = [], []
    for gt, pred in zip(gt_list, pred_list):
        gt_b = (gt > 0.5).astype('float32').ravel()
        npos = gt_b.sum()
        if npos == 0 or npos == len(gt_b):
            continue
        pf = pred.ravel().astype('float32')
        auc_px.append(max(_roc_auc(gt_b, pf), _roc_auc(gt_b, 1 - pf)))
        f1_px.append(max(_f1(gt_b, pf > 0.5), _f1(gt_b, 1 - pf > 0.5)))
    pixel_auc = float(np.mean(auc_px)) * 100 if auc_px else float('nan')
    f1        = float(np.mean(f1_px))  * 100 if f1_px  else float('nan')
    return pixel_auc, f1


# -- dataset ----------------------------------------------------------

def load_mask(path):
    """Load mask, return float32 binary at 224×224 (nearest-neighbour)."""
    m = cv2.imread(path, cv2.IMREAD_COLOR)
    if m is None:
        return None
    gt = (m[..., 1] > 0).astype('float32')
    return cv2.resize(gt, (224, 224), interpolation=cv2.INTER_NEAREST)

def collect_spliced_pairs(dataset_dir):
    splc_dir = os.path.join(dataset_dir, '4cam_splc')
    mask_dir = os.path.join(splc_dir, 'edgemask')
    pairs = []
    for ext in ('*.tif', '*.TIF', '*.bmp', '*.BMP'):
        for p in sorted(glob.glob(os.path.join(splc_dir, ext))):
            bname = os.path.splitext(os.path.basename(p))[0]
            mp = None
            for mext in ('.bmp', '.jpg', '.png'):
                for sfx in ('_edgemask', '_edgemask_3'):
                    cand = os.path.join(mask_dir, bname + sfx + mext)
                    if os.path.isfile(cand):
                        mp = cand
                        break
                if mp:
                    break
            if mp:
                pairs.append((p, mp))
    return pairs

def preload_dataset(pairs):
    """Load all images and masks into RAM once. Returns list of (rgb_uint8, gt_224)."""
    data = []
    for img_path, mask_path in pairs:
        raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw is None:
            continue
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        gt  = load_mask(mask_path)
        if gt is None:
            continue
        data.append((rgb, gt))
    return data


# -- inference helpers ------------------------------------------------

@torch.no_grad()
def infer_span(model, rgb_uint8):
    """SPAN: feature extractor at original size, resize feature tensor to 224x224."""
    x = torch.from_numpy(
            rgb_uint8.astype('float32') / 255.0 * 2.0 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast(device_type='cuda'):
        y = model(x)[0, 0]
    return y.float().cpu().numpy()         # (224,224)

@torch.no_grad()
def infer_mt(model, rgb_uint8):
    """ManTraNet: full original-size forward pass, resize output to 224x224."""
    x = torch.from_numpy(
            rgb_uint8.astype('float32') / 255.0 * 2.0 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.amp.autocast(device_type='cuda'):
        y = model(x)[0, 0]
    y = y.float().cpu().numpy()            # (H, W)
    return cv2.resize(y, (224, 224), interpolation=cv2.INTER_LINEAR)


# -- evaluation loop (both models, one image pass) --------------------

def run_one(span, mt, data, pert_fn, label):
    """Run both models on all images under pert_fn in a single image pass."""
    gt_list       = []
    span_preds    = []
    mt_preds      = []
    n = len(data)
    for done, (rgb, gt) in enumerate(data, 1):
        img = pert_fn(rgb) if pert_fn else rgb
        span_preds.append(infer_span(span, img))
        mt_preds.append(infer_mt(mt,   img))
        gt_list.append(gt)
        if done % 50 == 0:
            torch.cuda.empty_cache()
        print(f'\r  {label}: {done}/{n}', end='', flush=True)
    print()
    pa_s, f1_s = compute_metrics(gt_list, span_preds)
    pa_m, f1_m = compute_metrics(gt_list, mt_preds)
    return pa_s, f1_s, pa_m, f1_m


# -- main -------------------------------------------------------------

def main():
    print(f'Device: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    print('Loading SPAN ...')
    span = build_span(SPAN_WEIGHTS, device=DEVICE)

    print('Loading ManTraNet ...')
    mt = build_mantranet(MT_WEIGHTS, device=DEVICE)

    print('Pre-loading dataset into RAM ...')
    pairs = collect_spliced_pairs(DATASET_DIR)
    data  = preload_dataset(pairs)
    print(f'Loaded {len(data)} spliced images.')

    # Warmup: run all unique image sizes through both models so CUDA compiles
    # all kernels upfront — prevents mid-run slowdowns on new image dimensions.
    unique_sizes = {}
    for rgb, _ in data:
        key = rgb.shape[:2]
        if key not in unique_sizes:
            unique_sizes[key] = rgb
    print(f'Warming up {len(unique_sizes)} unique image sizes ...', end='', flush=True)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        for rgb in unique_sizes.values():
            x = torch.from_numpy(rgb.astype('float32')/255.0*2.0-1.0).permute(2,0,1).unsqueeze(0).to(DEVICE)
            span(x); mt(x)
    torch.cuda.synchronize()
    print(' done.\n')

    W = 42
    sep = '-' * (W + 36)
    hdr = f"{'Perturbation':<{W}} {'SPAN AUC':>9} {'MT AUC':>9} {'SPAN F1':>8} {'MT F1':>8}"
    print(sep); print(hdr); print(sep)

    results       = []
    cache_nomanip = None

    for name, fn in PERTURBATIONS:
        if name == 'No manipulation' and cache_nomanip is not None:
            pa_s, f1_s, pa_m, f1_m = cache_nomanip
        else:
            t0 = time.time()
            pa_s, f1_s, pa_m, f1_m = run_one(span, mt, data, fn, name)
            elapsed = time.time() - t0
            print(f'  -> done in {elapsed:.0f}s')
            if name == 'No manipulation':
                cache_nomanip = (pa_s, f1_s, pa_m, f1_m)

        print(f'{name:<{W}} {pa_s:>9.2f} {pa_m:>9.2f} {f1_s:>8.2f} {f1_m:>8.2f}')
        results.append((name, pa_s, f1_s, pa_m, f1_m))

    print(sep)

    csv = 'columbia_results.csv'
    with open(csv, 'w') as f:
        f.write('Perturbation,SPAN_Pixel_AUC,SPAN_F1,MT_Pixel_AUC,MT_F1\n')
        for r in results:
            f.write(f'{r[0]},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f}\n')
    print(f'\nSaved -> {csv}')


if __name__ == '__main__':
    main()