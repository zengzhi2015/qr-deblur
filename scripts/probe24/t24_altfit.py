from __future__ import annotations
import time
import cv2
import numpy as np
import t18_rigid as R
from t18_eval import n_iters, path_of
from t21_fit_diag import FIT_SIDE, ITERS_GEO, ITERS_PRE, Model, N_SEG
import t23_known_modules as KM
import qr_decoders as Q
from t24_known_fwd import pick_rot
from t24_search_e2 import e3_from_means
LAM = 0.001
QUIET = 4
E_TIE = 0.005

def canvas_from_bits(bits_black, quiet=QUIET, mp=8):
    n = int(bits_black.shape[0])
    pad = int(quiet) + 2
    S = (n + 2 * pad) * int(mp)
    canvas = np.ones((S, S), np.float32)
    core = np.kron((~np.asarray(bits_black, bool)).astype(np.float32), np.ones((mp, mp), np.float32))
    canvas[pad * mp:(pad + n) * mp, pad * mp:(pad + n) * mp] = core
    mask = np.zeros((S, S), np.float32)
    q0, q1 = ((pad - quiet) * mp, (pad + n + quiet) * mp)
    mask[q0:q1, q0:q1] = 1.0
    core_c = np.array([[pad * mp, pad * mp], [(pad + n) * mp, pad * mp], [(pad + n) * mp, (pad + n) * mp], [pad * mp, (pad + n) * mp]], float) - 0.5
    return (canvas, mask, core_c)

def otsu_bits(means):
    v = np.asarray(means, float)
    u = np.clip((v - v.min()) / max(float(v.max() - v.min()), 1e-06) * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return u <= t

def force_function(bits, n, rot):
    black, known = KM.known_patterns(int(n))
    if rot:
        black = np.rot90(black, int(rot) % 4)
        known = np.rot90(known, int(rot) % 4)
    out = np.asarray(bits, bool).copy()
    out[known] = black[known]
    return out

def _invert(u8, P, D, A, Wn):
    path = path_of((np.asarray(P, float), np.asarray(D, float), np.asarray(A, float)), Wn)
    if path is None:
        path = R.path_from_PDA(P, D, A, 41)
    x, geo, _ = R.invert_photo(u8, path, lam=LAM, iters=n_iters(path), paper=False)
    return (x, geo)

def score_invert(x, geo, P, quadA, n):
    q = KM.net_quad_on_canvas(quadA, P, geo)
    mp = geo['S_core'] / max(int(n), 1)
    ss = KM.shift_search(x, q, int(n), mp)
    e = float(ss['e']) if ss['e'] == ss['e'] else 1.0
    q2 = q + np.array([ss['dx'] * mp, ss['dy'] * mp], float)
    means = KM.module_means_quad(x, q2, int(n))
    return dict(e2=e, e3=e3_from_means(means), means=means)

def _down_for_fit(u8, P, D, A, n_mod):
    side = float(R.quad_side(P))
    f_extra = int(max(1, round(side / FIT_SIDE)))
    if f_extra > 1:
        h, w = u8.shape[:2]
        u8f = cv2.resize(u8, (max(w // f_extra, 8), max(h // f_extra, 8)), interpolation=cv2.INTER_AREA)
        u8f = np.clip(u8f, 0, 255).astype(np.uint8)
        P_f = (np.asarray(P, float) + 0.5) / f_extra - 0.5
        D_f = np.asarray(D, float) / f_extra
        A_f = np.asarray(A, float) / f_extra
    else:
        u8f = u8
        P_f, D_f, A_f = (np.asarray(P, float), np.asarray(D, float), np.asarray(A, float))
    mp_frame = side / max(int(n_mod), 1)
    mp_fit = mp_frame / f_extra
    return (u8f, P_f, D_f, A_f, f_extra, mp_fit)

def fit_once(torch, dev, u8, P, D, A, canvas, cmask, core_c, n_mod, geo_div=1):
    u8f, P_f, D_f, A_f, f_extra, mp_fit = _down_for_fit(u8, P, D, A, n_mod)
    x0, y0, side = R.window_for_quad(P_f)
    Hh, Ww = u8f.shape[:2]
    box = (max(x0, 0), max(y0, 0), min(x0 + side, Ww), min(y0 + side, Hh))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        raise RuntimeError(f'crop too small {box}')
    yc = (u8f.astype(np.float32) / 255.0)[box[1]:box[3], box[0]:box[2]]
    M = Model(torch, dev, canvas, cmask, core_c, box, yc, P_f, D_f, A_f, n_seg=N_SEG)
    M.wl.requires_grad_(False)
    M.wl.zero_()
    mask = (M.render(mask_only=True).detach() > 0.999).float()
    if float(mask.sum()) < 500:
        raise RuntimeError(f'mask too small {float(mask.sum())}')
    sig_levels = [mp_fit * 1.0, mp_fit * 0.5, mp_fit * 0.25, 0.0]
    pre = max(4, ITERS_PRE // (2 * geo_div))
    pre_p = max(8, ITERS_PRE // geo_div)
    geo = max(8, ITERS_GEO // 2 // geo_div)
    M.fit([M.lev, M.ill], pre, [0.02, 0.02], mask, sigma=sig_levels[1])
    M.fit([M.P, M.lev, M.ill], pre_p, [0.1, 0.01, 0.01], mask, sigma=sig_levels[1])
    mask = (M.render(mask_only=True).detach() > 0.999).float()
    M.side0 = float(R.quad_side(M.P.detach().cpu().numpy()))
    for sg in sig_levels[1:]:
        M.fit([M.P, M.D, M.A, M.lev, M.ill], geo, [0.05, 0.05, 0.05, 0.01, 0.01], mask, sigma=sg)
        mask = (M.render(mask_only=True).detach() > 0.999).float()
    P1 = M.P.detach().cpu().numpy()
    D1 = M.D.detach().cpu().numpy()
    A1 = M.A.detach().cpu().numpy()
    if f_extra > 1:
        P1 = (P1 + 0.5) * f_extra - 0.5
        D1 = D1 * f_extra
        A1 = A1 * f_extra
    return (P1, D1, A1)

def pick_theta(recs):
    ok = [r for r in recs if np.isfinite(r.get('e2', float('nan')))]
    if not ok:
        return recs[0]
    emin = min((r['e2'] for r in ok))
    ties = [r for r in ok if r['e2'] - emin < E_TIE]
    ties.sort(key=lambda r: (r['t'], -float(r.get('e3') or 0.0)))
    return ties[0]

def decode_round(u8, P, D, A, x_tb=None, geo_tb=None, lam_ladder=None):
    path = path_of((np.asarray(P, float), np.asarray(D, float), np.asarray(A, float)), float(max(u8.shape)))
    if path is None:
        return dict(payload='', which='', n_agree=0, x=None, geo=None, lam_used=LAM, ladder=[])
    it = n_iters(path)
    if x_tb is None or geo_tb is None:
        d = R.invert_decode_dual(u8, path, lam=LAM, iters=it)
        results = d['results']
        payload = d.get('payload') or ''
        which = d.get('which') or ''
        x, geo = (d.get('x'), d.get('geo'))
        n_agree = 0
        if which and which in results:
            n_agree = int(results[which].get('n_agree') or 0)
    else:
        results = {}
        r0 = dict(Q.standard_decode(x_tb, geo_tb))
        r0['x'], r0['geo'] = (x_tb, geo_tb)
        results['toward_black'] = r0
        x1, geo1, _ = R.invert_photo(u8, path, lam=LAM, iters=it, paper=True)
        r1 = dict(Q.standard_decode(x1, geo1))
        r1['x'], r1['geo'] = (x1, geo1)
        results['toward_paper'] = r1
        best = None
        for name, r in results.items():
            if r.get('payload') and (best is None or r['n_agree'] > results[best]['n_agree']):
                best = name
        if best is None:
            best = 'toward_black'
        rb = results[best]
        payload = rb.get('payload') or ''
        which = best if payload else ''
        x, geo = (rb['x'], rb['geo'])
        n_agree = int(rb.get('n_agree') or 0)
    lam_used = LAM
    ladder = [dict(lam=LAM, payload=payload, n_agree=n_agree)]
    if lam_ladder:
        assert abs(float(lam_ladder[0]) - LAM) < 1e-12, 'lam_ladder rung 0 must be the legacy LAM'
        for lam in lam_ladder[1:]:
            if n_agree >= 2:
                break
            try:
                d2 = R.invert_decode_dual(u8, path, lam=float(lam), iters=it)
            except Exception:
                ladder.append(dict(lam=float(lam), payload='', n_agree=0, err=True))
                continue
            p2 = d2.get('payload') or ''
            w2 = d2.get('which') or ''
            n2 = int(d2['results'][w2].get('n_agree') or 0) if w2 and w2 in d2.get('results', {}) else 0
            ladder.append(dict(lam=float(lam), payload=p2, n_agree=n2))
            if p2 and n2 > n_agree:
                payload, which, n_agree, lam_used = (p2, w2, n2, float(lam))
                x, geo = (d2.get('x'), d2.get('geo'))
    return dict(payload=payload, which=which, n_agree=n_agree, x=x, geo=geo, lam_used=lam_used, ladder=ladder)

def pick_consensus_strict(round_decs, e2_t, min_agree=2):
    el = [(int(d['t']), d['payload'], int(d.get('n_agree') or 0)) for d in round_decs if d.get('payload') and int(d.get('n_agree') or 0) >= min_agree]
    if not el:
        return (int(e2_t), 'none')
    cnt = {}
    for _, p, _ in el:
        cnt[p] = cnt.get(p, 0) + 1
    multi = [p for p, c in cnt.items() if c >= 2]
    if multi:
        best_p = max(multi, key=lambda p: (cnt[p], max((n for _, q, n in el if q == p))))
        n_best, neg_t = max(((n, -t) for t, q, n in el if q == best_p))
        return (int(-neg_t), 'consensus')
    n_best, neg_t, _ = max(((n, -t, p) for t, p, n in el))
    return (int(-neg_t), 'max_agree')

def altfit(u8, quadA, P0, D0, A0, rounds, torch, device, geo_div, lam_ladder, early_stop):
    t0 = time.time()
    P = np.asarray(P0, float)
    D = np.asarray(D0, float)
    A = np.asarray(A0, float)
    quadA = np.asarray(quadA, float)
    u8 = np.asarray(u8)
    Wn = float(max(u8.shape))
    x, geo = _invert(u8, P, D, A, Wn)
    n_lock = int(R.finder_levels(R.canvas_core_view(x, geo), geo['S_core'])['n_core'])
    rot = pick_rot(torch, device, u8, P, D, A, n_lock)
    thetas = [(P.copy(), D.copy(), A.copy())]
    recs = []
    views = []
    decs = []

    def record(t, P_, D_, A_, x_, geo_, sc):
        rec = dict(t=t, e2=sc['e2'], e3=sc['e3'], P=np.asarray(P_, float), D=np.asarray(D_, float), A=np.asarray(A_, float))
        recs.append(rec)
        views.append(dict(x=x_, geo=geo_))

    def decode_rec(rec, view):
        d = decode_round(u8, rec['P'], rec['D'], rec['A'], view.get('x'), view.get('geo'), lam_ladder=lam_ladder)
        decs.append({'t': rec['t'], 'payload': d['payload'], 'which': d['which'], 'n_agree': d['n_agree'], 'lam_used': d['lam_used'], 'ladder': d['ladder']})

    def stop_now():
        return bool(early_stop and decs and (int(decs[-1].get('n_agree') or 0) >= 2))
    sc0 = score_invert(x, geo, P, quadA, n_lock)
    bits = force_function(otsu_bits(sc0['means']), n_lock, rot)
    record(0, P, D, A, x, geo, sc0)
    decode_rec(recs[-1], views[-1])
    for t in range(int(rounds)):
        if stop_now():
            break
        Pt, Dt, At = thetas[-1]
        side = float(R.quad_side(Pt))
        f_extra = int(max(1, round(side / FIT_SIDE)))
        mp_fit = side / max(n_lock, 1) / f_extra
        mp = int(np.clip(np.ceil(mp_fit * 1.5), 4, 24))
        canvas, cmask, core_c = canvas_from_bits(bits, quiet=QUIET, mp=mp)
        try:
            P1, D1, A1 = fit_once(torch, device, u8, Pt, Dt, At, canvas, cmask, core_c, n_lock, geo_div=geo_div)
            thetas.append((P1, D1, A1))
        except Exception:
            thetas.append((Pt.copy(), Dt.copy(), At.copy()))
        P1, D1, A1 = thetas[-1]
        x, geo = _invert(u8, P1, D1, A1, Wn)
        sc = score_invert(x, geo, P1, quadA, n_lock)
        bits = force_function(otsu_bits(sc['means']), n_lock, rot)
        record(t + 1, P1, D1, A1, x, geo, sc)
        decode_rec(recs[-1], views[-1])
    chosen = pick_theta(recs)
    t_sel = int(chosen['t'])
    t_cons, cons_how = pick_consensus_strict(decs, t_sel)
    return dict(e2=[r['e2'] for r in recs], e3=[r['e3'] for r in recs], chosen=t_sel, theta_sel=thetas[t_sel], chosen_cons=int(t_cons), cons_how=cons_how, theta_cons=thetas[t_cons], decs=decs, views=views, recs=recs, n_lock=n_lock, rot=int(rot), geo_div=int(geo_div), stopped_early=bool(len(recs) < int(rounds) + 1), sec=time.time() - t0)
