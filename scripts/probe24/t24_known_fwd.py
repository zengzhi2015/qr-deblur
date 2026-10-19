from __future__ import annotations
import numpy as np
import t18_rigid as R
from t21_fit_diag import Model, rot_canvas
from t23_known_modules import known_patterns
QUIET = 4
Q_RING = 2
F_THR = 0.6
HUB_DELTA = 0.05
N_EFF_MIN = 400.0
N_SEG = 32

def canvas_mp(side_px, n):
    mp_frame = float(side_px) / max(int(n), 1)
    return (int(np.clip(np.ceil(mp_frame * 1.5), 4, 24)), mp_frame)

def known_canvases(n, rot, mp):
    n = int(n)
    mp = int(mp)
    rot = int(rot) % 4
    black, known = known_patterns(n)
    pad = QUIET + 2
    S = (n + 2 * pad) * mp
    canvas = np.ones((S, S), np.float32)
    core = np.full((n, n), 0.5, np.float32)
    core[known & black] = 0.0
    core[known & ~black] = 1.0
    sl = slice(pad * mp, (pad + n) * mp)
    canvas[sl, sl] = np.kron(core, np.ones((mp, mp), np.float32))
    kc = np.zeros((S, S), np.float32)
    kc[sl, sl] = np.kron(known.astype(np.float32), np.ones((mp, mp), np.float32))
    r0 = (pad - Q_RING) * mp
    r1 = (pad + n + Q_RING) * mp
    ring_m = np.zeros((S, S), bool)
    ring_m[r0:r1, r0:r1] = True
    ring_m[sl, sl] = False
    kc[ring_m] = 1.0
    if rot:
        canvas = rot_canvas(canvas, rot)
        kc = rot_canvas(kc, rot)
    cmask = np.zeros((S, S), np.float32)
    q0, q1 = ((pad - QUIET) * mp, (pad + n + QUIET) * mp)
    cmask[q0:q1, q0:q1] = 1.0
    core_c = np.array([[pad * mp, pad * mp], [(pad + n) * mp, pad * mp], [(pad + n) * mp, (pad + n) * mp], [pad * mp, (pad + n) * mp]], float) - 0.5
    return (canvas, kc, cmask, core_c)

def crop_box_for(P, shape_hw):
    H, W = (int(shape_hw[0]), int(shape_hw[1]))
    x0, y0, side = R.window_for_quad(np.asarray(P, float))
    return (max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H))

def _as_chw(arr, torch, dev):
    t = torch.as_tensor(np.asarray(arr, np.float32), device=dev)
    if t.ndim == 2:
        t = t[None, None]
    return t

def build_model_ctx(torch, dev, u8, P, D, A, n, rot):
    P = np.asarray(P, float)
    D = np.asarray(D, float)
    A = np.asarray(A, float)
    u8 = np.asarray(u8)
    H, W = u8.shape[:2]
    mp, _ = canvas_mp(R.quad_side(P), n)
    canvas, kc, cmask, core_c = known_canvases(n, rot, mp)
    box = crop_box_for(P, (H, W))
    x0, y0, x1, y1 = box
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise RuntimeError(f'crop too small: {box}')
    photo_crop = (u8.astype(np.float32) / 255.0)[y0:y1, x0:x1]
    M = Model(torch, dev, canvas, cmask, core_c, box, photo_crop, P, D, A, n_seg=N_SEG)
    M.wl.requires_grad_(False)
    M.wl.zero_()
    return dict(M=M, known_t=_as_chw(canvas, torch, dev), kc_t=_as_chw(kc, torch, dev))

def _set_pda(M, P, D, A):
    torch = M.torch
    with torch.no_grad():
        M.P.copy_(torch.as_tensor(np.asarray(P, np.float32), device=M.dev))
        M.D.copy_(torch.as_tensor(np.asarray(D, np.float32), device=M.dev))
        M.A.copy_(torch.as_tensor(np.asarray(A, np.float32), device=M.dev))
        M.wl.zero_()

def _closed_levels(w, c, y_lin):
    oc = 1.0 - c
    A11 = (w * oc * oc).sum()
    A12 = (w * oc * c).sum()
    A22 = (w * c * c).sum()
    b1 = (w * oc * y_lin).sum()
    b2 = (w * c * y_lin).sum()
    ridge = A11.new_tensor(1e-08)
    A11 = A11 + ridge
    A22 = A22 + ridge
    det = A11 * A22 - A12 * A12
    b = (A22 * b1 - A12 * b2) / det
    wh = (A11 * b2 - A12 * b1) / det
    return (b, wh)

def e1_score(ctx, P, D, A):
    M = ctx['M']
    torch = M.torch
    with torch.no_grad():
        _set_pda(M, P, D, A)
        c = M.render(canvas=ctx['known_t'])
        f = M.render(canvas=ctx['kc_t'])
        w = (f >= F_THR).to(c.dtype) * (f * f)
        if float(w.sum()) < N_EFF_MIN:
            return float('inf')
        y_disp = M.y
        y_lin = torch.pow(torch.clamp(y_disp, 0.0, 1.0), R.GAMMA)
        b, wh = _closed_levels(w, c, y_lin)
        lin = torch.clamp(b + (wh - b) * c, 0.0, 1.0)
        pred = torch.pow(lin + 1e-06, 1.0 / R.GAMMA)
        r = pred - y_disp
        hub = torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction='none', delta=HUB_DELTA)
        return float((w * hub).sum() / w.sum().clamp(min=1e-12))

def pick_rot(torch, dev, u8, P, D, A, n):
    best = None
    for rot in range(4):
        ctx = build_model_ctx(torch, dev, u8, P, D, A, n, rot)
        e1 = e1_score(ctx, P, D, A)
        if best is None or e1 < best[0]:
            best = (e1, rot)
    return best[1]
