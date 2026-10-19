import cv2
import numpy as np
import t18_rigid as R
_FINDER = np.ones((7, 7), bool)
_FINDER[1:6, 1:6] = False
_FINDER[2:5, 2:5] = True
_ALIGN = np.array([[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], bool)
ALIGN_POS = {25: (18,), 29: (22,), 33: (26,), 37: (30,), 41: (34,), 45: (22, 38), 49: (24, 42), 53: (26, 46), 57: (28, 50)}

def known_patterns(n):
    black = np.zeros((n, n), bool)
    known = np.zeros((n, n), bool)
    for r0, c0 in ((0, 0), (0, n - 7), (n - 7, 0)):
        known[r0:r0 + 7, c0:c0 + 7] = True
        black[r0:r0 + 7, c0:c0 + 7] = _FINDER
    for r0, c0 in ((0, 0), (0, n - 8), (n - 8, 0)):
        known[max(r0, 0):min(r0 + 8, n), max(c0, 0):min(c0 + 8, n)] = True
    for i in range(8, n - 8):
        known[6, i] = True
        black[6, i] = i % 2 == 0
        known[i, 6] = True
        black[i, 6] = i % 2 == 0
    for r in ALIGN_POS.get(n, ()):
        for c in ALIGN_POS.get(n, ()):
            if r < 10 and c < 10 or (r < 10 and c > n - 11) or (r > n - 11 and c < 10):
                continue
            r0, c0 = (r - 2, c - 2)
            if r0 < 0 or c0 < 0 or r0 + 5 > n or (c0 + 5 > n):
                continue
            known[r0:r0 + 5, c0:c0 + 5] = True
            black[r0:r0 + 5, c0:c0 + 5] = _ALIGN
    return (black, known)

def module_means_quad(x, quad, n, up=4):
    src = np.float32(quad)
    dst = np.float32([[0, 0], [n * up, 0], [n * up, n * up], [0, n * up]])
    w = cv2.warpPerspective(np.asarray(x, np.float32), cv2.getPerspectiveTransform(src, dst), (n * up, n * up), flags=cv2.INTER_LINEAR)
    return w.reshape(n, up, n, up).mean((1, 3))

def e_known_and_sep(means, black, known):
    mb = means[known & black]
    mw = means[known & ~black]
    if mb.size < 4 or mw.size < 4:
        return (float('nan'), float('nan'))
    thr = 0.5 * (float(mb.mean()) + float(mw.mean()))
    pred_black = means < thr
    e = float((pred_black[known] != black[known]).mean())
    sb = float(mb.std()) + 1e-09
    sw = float(mw.std()) + 1e-09
    sep = float((mw.mean() - mb.mean()) / (sb + sw))
    return (e, sep)

def finder_sep(means, n):
    best = -1000000000.0
    best_rot = 0
    for rot in range(4):
        b, k = known_patterns(n)
        if rot:
            b, k = (np.rot90(b, rot), np.rot90(k, rot))
        fin = np.zeros((n, n), bool)
        for r0, c0 in ((0, 0), (0, n - 7), (n - 7, 0)):
            fin[r0:r0 + 7, c0:c0 + 7] = True
        fin = np.rot90(fin, rot) if rot else fin
        e, sep = e_known_and_sep(means, b, fin)
        if np.isfinite(sep) and sep > best:
            best, best_rot = (sep, rot)
    return (best, best_rot)

def apply_H(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:3]

def match_order(P_ref, P_hat):
    ir, ih = (R.order_quad(P_ref), R.order_quad(P_hat))
    out = np.empty_like(P_hat, float)
    for k in range(4):
        out[ir[k]] = P_hat[ih[k]]
    return out

def net_quad_on_canvas(P_net, P_fit, geo):
    Pn = match_order(P_fit, np.asarray(P_net, float))
    H = R.homography_pts(np.asarray(P_fit, float), geo['canvas_core'])
    return apply_H(H, Pn)

def shift_search(x, quad0, n, mp, half=0.5, step=0.125):
    best = None
    for dy in np.arange(-half, half + 1e-09, step):
        for dx in np.arange(-half, half + 1e-09, step):
            q = quad0 + np.array([dx * mp, dy * mp], float)
            means = module_means_quad(x, q, n)
            sep, rot = finder_sep(means, n)
            if best is None or sep > best[0]:
                b, k = known_patterns(n)
                if rot:
                    b, k = (np.rot90(b, rot), np.rot90(k, rot))
                e, sep2 = e_known_and_sep(means, b, k)
                best = (sep, e, sep2, rot, float(dx), float(dy))
    return dict(e=best[1], sep=best[2], rot=best[3], dx=best[4], dy=best[5])
