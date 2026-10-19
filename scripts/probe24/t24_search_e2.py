from __future__ import annotations
import time
import numpy as np
import t18_rigid as R
from t18_eval import n_iters, path_of
import t23_known_modules as KM
LAM = 0.001
E_TIE = 0.005
KS_I = tuple((round(0.7 + i * 0.05, 2) for i in range(14)))
DPHIS_II = (-10.0, -5.0, 5.0, 10.0)
DK_III = (0.05, 0.1)

def rotate_vecs(V, deg):
    t = np.deg2rad(float(deg))
    c, s = (np.cos(t), np.sin(t))
    return np.asarray(V, float) @ np.array([[c, -s], [s, c]], float).T

def apply_kd(P, D, A, k, dphi):
    P = np.asarray(P, float)
    D2 = float(k) * rotate_vecs(D, dphi)
    A2 = rotate_vecs(A, dphi)
    return (P, D2, A2)

def dist_origin(k, dphi):
    return ((float(k) - 1.0) / 0.05) ** 2 + (float(dphi) / 5.0) ** 2

def e3_from_means(means):
    v = np.asarray(means, float).ravel()
    tv = float(v.var())
    if tv < 1e-12 or v.size < 4:
        return 0.0
    import cv2
    u = np.clip((v - v.min()) / max(v.max() - v.min(), 1e-06) * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = (v[u <= t], v[u > t])
    if lo.size == 0 or hi.size == 0:
        return 0.0
    mu = float(v.mean())
    w0, w1 = (lo.size / v.size, hi.size / v.size)
    bcv = w0 * (float(lo.mean()) - mu) ** 2 + w1 * (float(hi.mean()) - mu) ** 2
    return float(bcv / tv)

def pick_best(cands):
    ok = [c for c in cands if np.isfinite(c.get('e', float('nan')))]
    if not ok:
        return None
    emin = min((c['e'] for c in ok))
    ties = [c for c in ok if c['e'] - emin < E_TIE]
    ties.sort(key=lambda c: (dist_origin(c['k'], c['dphi']), -float(c.get('E3') or 0.0)))
    return ties[0]

def _score_xy(x, geo, P, quadA, n_lock, k, dphi):
    q = KM.net_quad_on_canvas(quadA, P, geo)
    mp = geo['S_core'] / max(int(n_lock), 1)
    rec = KM.shift_search(x, q, int(n_lock), mp)
    e = float(rec['e']) if rec['e'] == rec['e'] else 1.0
    q2 = q + np.array([rec['dx'] * mp, rec['dy'] * mp], float)
    means = KM.module_means_quad(x, q2, int(n_lock))
    return dict(k=float(k), dphi=float(dphi), e=e, E3=e3_from_means(means))

def _score_invert(u8, P, D, A, k, dphi, quadA, n_lock, n_path):
    P, D2, A2 = apply_kd(P, D, A, k, dphi)
    path = R.path_from_PDA(P, D2, A2, n_path)
    x, geo, _ = R.invert_photo(u8, path, lam=LAM, iters=n_iters(path), paper=False)
    return _score_xy(x, geo, P, quadA, n_lock, k, dphi)

def search_e2(u8, quadA, P, D, A):
    t0 = time.time()
    P = np.asarray(P, float)
    D = np.asarray(D, float)
    A = np.asarray(A, float)
    quadA = np.asarray(quadA, float)
    u8 = np.asarray(u8)
    Wn = float(max(u8.shape))
    path0 = path_of((P, D, A), Wn)
    if path0 is None:
        path0 = R.path_from_PDA(P, D, A, 41)
    n_path = int(path0.shape[0])
    path1 = R.path_from_PDA(P, D, A, n_path)
    x1, geo1, _ = R.invert_photo(u8, path1, lam=LAM, iters=n_iters(path1), paper=False)
    n_lock = int(R.finder_levels(R.canvas_core_view(x1, geo1), geo1['S_core'])['n_core'])
    rec1 = _score_xy(x1, geo1, P, quadA, n_lock, 1.0, 0.0)
    rec1['stage'] = 'i'
    scan = [rec1]
    cache = {(1.0, 0.0): rec1}

    def eval_pt(k, dphi, stage):
        key = (round(float(k), 4), round(float(dphi), 4))
        if key in cache:
            row = dict(cache[key])
            row['stage'] = stage
            return row
        row = _score_invert(u8, P, D, A, k, dphi, quadA, n_lock, n_path)
        row['stage'] = stage
        cache[key] = row
        return row
    for k in KS_I:
        if abs(k - 1.0) < 1e-09:
            continue
        scan.append(eval_pt(k, 0.0, 'i'))
    stage_i = [r for r in scan if r['stage'] == 'i']
    best_i = pick_best(stage_i)
    k1 = float(best_i['k']) if best_i else 1.0
    for dphi in DPHIS_II:
        scan.append(eval_pt(k1, dphi, 'ii'))
    stage_ii = [r for r in scan if r['stage'] in ('i', 'ii') and abs(r['k'] - k1) < 1e-09]
    if not any((abs(r['dphi']) < 1e-09 and abs(r['k'] - k1) < 1e-09 for r in stage_ii)):
        stage_ii.append(eval_pt(k1, 0.0, 'i'))
    best_ii = pick_best(stage_ii) or best_i
    k_ii = float(best_ii['k'])
    dphi1 = float(best_ii['dphi'])
    for dk in DK_III:
        for sign in (-1.0, 1.0):
            k = round(k1 + sign * dk, 4)
            if k < 0.5 or k > 1.5:
                continue
            scan.append(eval_pt(k, dphi1, 'iii'))
    pool_iii = [r for r in scan if r.get('stage') == 'iii' or (abs(r['k'] - k_ii) < 1e-09 and abs(r['dphi'] - dphi1) < 1e-09)]
    best = pick_best(pool_iii) or best_ii or rec1
    k_star = float(best['k'])
    dphi_star = float(best['dphi'])
    return dict(theta2=apply_kd(P, D, A, k_star, dphi_star), k=k_star, dphi=dphi_star, e2_star=float(best['e']), sec=time.time() - t0)
