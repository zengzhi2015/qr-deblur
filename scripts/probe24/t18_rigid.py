import os
import cv2
import numpy as np
from _common import init_env
init_env()
NET = 256
GAMMA = 2.2

def gamma_encode(lin):
    return np.power(np.clip(lin, 0.0, 1.0), 1.0 / GAMMA)

def gamma_decode(srgb):
    return np.power(np.clip(srgb, 0.0, 1.0), GAMMA)

def u8_to_linear(u8):
    return gamma_decode(np.asarray(u8, np.float32) / 255.0)
_TORCH = {}

def _torch(device=None):
    import torch
    if 'dev' not in _TORCH:
        env = os.environ.get('T18_DEVICE')
        _TORCH['dev'] = device or env or ('cuda' if torch.cuda.is_available() else 'cpu')
    return (torch, _TORCH['dev'])

def _base_grid(torch, dev, H, Wd):
    key = ('grid', H, Wd, str(dev))
    if key not in _TORCH:
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(Wd, device=dev, dtype=torch.float32), indexing='ij')
        g = torch.stack([xs, ys, torch.ones_like(xs)], -1)
        if sum((1 for k in _TORCH if isinstance(k, tuple) and k[0] == 'grid')) >= 4:
            return g
        _TORCH[key] = g
    return _TORCH[key]

def make_grids(Hs, src_hw, out_hw):
    torch, dev = _torch()
    h, w = src_hw
    H, Wd = out_hw
    base = _base_grid(torch, dev, H, Wd).reshape(-1, 3)
    Hinv = torch.from_numpy(np.linalg.inv(np.asarray(Hs, np.float64)).astype(np.float32)).to(dev)
    p = base @ Hinv.transpose(1, 2)
    z = p[..., 2:3]
    z = torch.where(z.abs() < 1e-08, torch.full_like(z, 1e-08), z)
    xy = p[..., :2] / z
    gx = (xy[..., 0] + 0.5) / w * 2 - 1
    gy = (xy[..., 1] + 0.5) / h * 2 - 1
    return torch.stack([gx, gy], -1).reshape(-1, H, Wd, 2)

def warp_mean_grids(src, grids, padding='zeros', chunk=32):
    torch, _ = _torch()
    N = grids.shape[0]
    acc = None
    for s in range(0, N, chunk):
        g = grids[s:s + chunk]
        out = torch.nn.functional.grid_sample(src[None].expand(g.shape[0], -1, -1, -1), g, mode='bilinear', padding_mode=padding, align_corners=False).sum(0)
        acc = out if acc is None else acc + out
    return acc / N

def homography_pts(src_pts, dst_pts):
    return cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts)).astype(np.float64)
WIN_FACTOR = 1.5
WIN_MARGIN = 0.15
NET_B = 384

def window_for_quad(quad):
    q = np.asarray(quad, float)
    s_q = quad_side(q)
    lo_xy, hi_xy = (q.min(0), q.max(0))
    c = (lo_xy + hi_xy) / 2
    side = max(WIN_FACTOR * s_q, float((hi_xy - lo_xy).max()) / (1 - 2 * WIN_MARGIN))
    side = int(np.ceil(side))
    x0, y0 = (int(round(c[0] - side / 2)), int(round(c[1] - side / 2)))
    return (x0, y0, side)

def crop_window(img, win):
    x0, y0, side = win
    H, W_ = img.shape[:2]
    xa, ya, xb, yb = (max(x0, 0), max(y0, 0), min(x0 + side, W_), min(y0 + side, H))
    if xb <= xa or yb <= ya:
        return np.full((side, side), int(np.median(img)), img.dtype)
    core = img[ya:yb, xa:xb]
    return cv2.copyMakeBorder(core, ya - y0, y0 + side - yb, xa - x0, x0 + side - xb, cv2.BORDER_REPLICATE)

def order_quad(P):
    P = np.asarray(P, float)
    c = P.mean(0)
    ang = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
    perm = np.argsort(ang)
    k = int(np.argmin(P[perm, 0] + P[perm, 1]))
    return np.roll(perm, -k)

def path_from_PDA(Pm, D, A, n):
    ss = np.linspace(-1, 1, n)
    return np.stack([Pm + s * D + 0.5 * s * s * A for s in ss])

def decode_quad(v, W):
    return np.asarray(v, float).reshape(4, 2) * W

def quad_side(q):
    q = np.asarray(q, float)
    return float(np.mean([np.hypot(*q[(k + 1) % 4] - q[k]) for k in range(4)]))

def canvas_geometry(path, margin_frac=0.12):
    n = path.shape[0]
    Pm = path[n // 2]
    S_core = quad_side(Pm)
    sweep = np.linalg.norm(path - Pm[None], axis=-1).max()
    m = int(np.ceil(max(margin_frac * S_core, 1.5 * sweep) + 8))
    S = int(np.ceil(S_core)) + 2 * m
    canvas_core = np.array([[m, m], [m + S_core, m], [m + S_core, m + S_core], [m, m + S_core]], float)
    return dict(S=S, m=m, S_core=S_core, canvas_core=canvas_core)

def path_homographies(path, geo):
    return np.stack([homography_pts(geo['canvas_core'], P) for P in path])

def crop_box(Hs, geo, photo_hw, pad=4):
    S = geo['S']
    cc = np.array([[-0.5, -0.5, 1], [S - 0.5, -0.5, 1], [S - 0.5, S - 0.5, 1], [-0.5, S - 0.5, 1]], float)
    pts = []
    for H in Hs:
        p = cc @ H.T
        pts.append(p[:, :2] / p[:, 2:3])
    pts = np.concatenate(pts, 0)
    x0 = int(np.clip(np.floor(pts[:, 0].min()) - pad, 0, photo_hw[1] - 1))
    x1 = int(np.clip(np.ceil(pts[:, 0].max()) + pad, x0 + 1, photo_hw[1]))
    y0 = int(np.clip(np.floor(pts[:, 1].min()) - pad, 0, photo_hw[0] - 1))
    y1 = int(np.clip(np.ceil(pts[:, 1].max()) + pad, y0 + 1, photo_hw[0]))
    return (x0, y0, x1, y1)

class PathOperator:

    def __init__(self, Hs, geo, box):
        torch, dev = _torch()
        self.torch, self.dev = (torch, dev)
        x0, y0, x1, y1 = box
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
        self.Hs = np.stack([T @ H for H in Hs])
        self.S = geo['S']
        self.hw = (y1 - y0, x1 - x0)
        self.grids = make_grids(self.Hs, (self.S, self.S), self.hw)
        ones = torch.ones((1, self.S, self.S), device=dev)
        self.alpha = warp_mean_grids(ones, self.grids, padding='zeros')[0]
        self.mask = (self.alpha > 0.999).float()

    def B(self, x):
        return warp_mean_grids(x[None], self.grids, padding='zeros')[0] * self.mask

    def Bt(self, r):
        torch = self.torch
        x = torch.zeros((self.S, self.S), device=self.dev, requires_grad=True)
        y = self.B(x)
        g, = torch.autograd.grad((y * r).sum(), x)
        return g.detach()

    def solve(self, y, lam=0.001, iters=40, level=None, free=None):
        torch = self.torch
        yt = torch.from_numpy(np.asarray(y, np.float32)).to(self.dev) * self.mask
        if level is not None:
            ones = torch.ones((self.S, self.S), device=self.dev)
            yt = yt - self.B(ones * float(level))
            M = torch.ones((self.S, self.S), device=self.dev) if free is None else torch.from_numpy(np.asarray(free, np.float32)).to(self.dev)
        else:
            M = None

        def A(v):
            if M is None:
                return self.Bt(self.B(v)) + lam * v
            return M * self.Bt(self.B(M * v)) + lam * v
        b = self.Bt(yt) if M is None else M * self.Bt(yt)
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs = (r * r).sum()
        for _ in range(iters):
            Ap = A(p)
            alpha = rs / torch.clamp((p * Ap).sum(), min=1e-20)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r * r).sum()
            if rs_new.sqrt() < 1e-07 * max(1.0, float(b.norm())):
                break
            p = r + rs_new / rs * p
            rs = rs_new
        if M is not None:
            x = float(level) + M * x
        return x.cpu().numpy()
PAPER_BAND_FRAC = 0.12

def paper_prior(photo_u8, path, geo, band_frac=PAPER_BAND_FRAC):
    rect = rectify_mid(photo_u8, path, geo)
    S, m, Sc = (geo['S'], geo['m'], geo['S_core'])
    band = band_frac * Sc
    a = int(max(m - band, 0))
    b = int(min(m + Sc + band, S))
    free = np.zeros((S, S), np.float32)
    free[a:b, a:b] = 1.0
    outside = rect[free == 0]
    level = float(np.median(outside)) if outside.size >= 64 else float(np.median(rect))
    return (level, free)

def invert_photo(photo_u8, path, lam=0.001, iters=40, paper=True):
    y_lin = u8_to_linear(photo_u8)
    geo = canvas_geometry(path)
    Hs = path_homographies(path, geo)
    box = crop_box(Hs, geo, y_lin.shape)
    op = PathOperator(Hs, geo, box)
    level, free = paper_prior(photo_u8, path, geo) if paper else (None, None)
    x = op.solve(y_lin[box[1]:box[3], box[0]:box[2]], lam=lam, iters=iters, level=level, free=free)
    return (x, geo, op)
CG_SOLVERS = (('toward_black', False), ('toward_paper', True))

def invert_decode_dual(photo_u8, path, lam=0.001, iters=40):
    import qr_decoders as Q
    results = {}
    best = None
    for name, paper in CG_SOLVERS:
        x, geo, _ = invert_photo(photo_u8, path, lam=lam, iters=iters, paper=paper)
        r = Q.standard_decode(x, geo)
        r['x'] = x
        r['geo'] = geo
        results[name] = r
        if r['payload'] and (best is None or r['n_agree'] > results[best]['n_agree']):
            best = name
    if best is None:
        best = CG_SOLVERS[0][0]
    rb = results[best]
    return dict(payload=rb['payload'], which=best if rb['payload'] else '', results=results, x=rb['x'], geo=rb['geo'])

def n_inverse_samples(path, step_px=0.75, lo=8, hi=96):
    seg = np.linalg.norm(np.diff(path, axis=0), axis=-1).sum(0).max()
    return int(np.clip(np.ceil(seg / step_px), lo, hi))

def canvas_core_view(x, geo, quiet_modules=4):
    m, S_core = (geo['m'], geo['S_core'])
    pad = int(min(m, quiet_modules * (S_core / 25)))
    a = int(max(m - pad, 0))
    b = int(min(m + S_core + pad, x.shape[0]))
    return x[a:b, a:b]
_FINDER = np.ones((7, 7), bool)
_FINDER[1:6, 1:6] = False
_FINDER[2:5, 2:5] = True
N_CORE_LADDER = (25, 29, 33, 37, 41, 45, 49, 53, 57)

def finder_levels(v, S_core, n_core=None):
    off = (v.shape[0] - S_core) / 2.0
    best = None
    for n in (n_core,) if n_core else N_CORE_LADDER:
        mpx = S_core / n
        w = 7 * mpx
        corners = {'tl': (off, off), 'tr': (off + S_core - w, off), 'bl': (off, off + S_core - w), 'br': (off + S_core - w, off + S_core - w)}
        stats = []
        for name, (x0, y0) in corners.items():
            src = np.float32([[x0, y0], [x0 + w, y0], [x0 + w, y0 + w], [x0, y0 + w]])
            H = cv2.getPerspectiveTransform(src, np.float32([[0, 0], [56, 0], [56, 56], [0, 56]]))
            cells = cv2.warpPerspective(v.astype(np.float32), H, (56, 56), flags=cv2.INTER_LINEAR).reshape(7, 8, 7, 8).mean((1, 3))
            blk, wht = (float(cells[_FINDER].mean()), float(cells[~_FINDER].mean()))
            c = np.corrcoef(cells.ravel(), (~_FINDER).astype(float).ravel())[0, 1] if cells.std() > 1e-09 else 0.0
            stats.append((float(c), name, blk, wht))
        stats.sort(reverse=True)
        top3 = stats[:3]
        score = float(np.mean([s[0] for s in top3]))
        if best is None or score > best['score']:
            best = dict(score=score, n_core=n, corners=[s[1] for s in top3], black=float(np.mean([s[2] for s in top3])), white=float(np.mean([s[3] for s in top3])))
    return best

def decode_canvas_info_levels(x, geo):
    v = canvas_core_view(x, geo)
    lv = finder_levels(v, geo['S_core'])
    vn = np.clip((v - lv['black']) / max(lv['white'] - lv['black'], 1e-06), 0, 1)
    return dict(black=lv['black'], white=lv['white'], n_core=lv['n_core'], finder_score=lv['score'], vn=vn)

def ladder_candidates(vn, geo):
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu
    S = geo['S_core']
    sigmas = (1.0, S / 57 / 4, S / 41 / 4, S / 33 / 4, S / 25 / 4, S / 25 / 3)
    cands = [('raw', vn)]
    for sig in sigmas:
        cands.append((f'gauss sigma={max(sig, 0.5):.2f}', np.clip(gaussian_filter(vn, max(sig, 0.5)), 0, 1)))
    g = cands[3][1]
    if g.max() - g.min() > 1e-06:
        cands.append(('otsu', (g > threshold_otsu(g)).astype(np.float64)))
    cands.append(('gamma', gamma_encode(vn)))
    cands.append(('gamma+gauss', gamma_encode(cands[3][1])))
    if vn.shape[0] > 300:
        cands.append(('half', cv2.resize(vn, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)))
    return cands

def rectify_mid(photo_u8, path, geo):
    y_lin = u8_to_linear(photo_u8)
    Hmid = homography_pts(geo['canvas_core'], path[path.shape[0] // 2])
    return cv2.warpPerspective(y_lin, np.linalg.inv(Hmid), (geo['S'], geo['S']), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
