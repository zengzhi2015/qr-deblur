from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
from . import config as CFG
from .config import FROZEN_HASH, VERSION
from .diagnostics import collect_flags, finder_ladder_scores, quad_degenerate
_STATE = {}

def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)

def _arr(x):
    return np.asarray(x, float).tolist()

def load_runtime():
    if _STATE.get('torch') is None:
        import canon_net
        import t20_net as NET
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(CFG.GPU_MEM_FRACTION)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _STATE.update(torch=torch, device=device, netA=NET.load(torch, CFG.WEIGHT_A, device), netB=canon_net.load(torch, CFG.WEIGHT_B_CANON, device))
    return _STATE

def _load_u8(path_or_gray):
    import real_capture_common as C
    t0 = time.time()
    src = None
    if isinstance(path_or_gray, (str, Path)):
        src = Path(path_or_gray)
        gray = C.load_gray(src)
    else:
        gray = np.asarray(path_or_gray, float)
    u8 = np.clip(C.downscale(gray, CFG.DOWN) * 255.0, 0, 255).astype(np.uint8)
    return (gray, u8, src, time.time() - t0)

def _versions(torch):
    cuda = ''
    if torch.cuda.is_available():
        cuda = torch.version.cuda or ''
    return {'qrpipe': VERSION, 'config_hash': FROZEN_HASH, 'weight_sha256': dict(CFG.WEIGHT_SHA256), 'weight_sha256_match': CFG.WEIGHT_SHA256 == CFG.WEIGHT_SHA256_EXPECTED, 'torch': getattr(torch, '__version__', ''), 'cuda': cuda}

def _rounds_from_af(af):
    decs = {int(d['t']): d for d in af.get('decs') or []}
    out = []
    for rec in af.get('recs') or []:
        t = int(rec['t'])
        d = decs.get(t, {})
        out.append({'P': _arr(rec['P']), 'D': _arr(rec['D']), 'A': _arr(rec['A']), 'e2': float(rec.get('e2', float('nan'))), 'e3': float(rec.get('e3', float('nan'))), 'payload': d.get('payload') or '', 'n_agree': int(d.get('n_agree') or 0), 'which': d.get('which') or '', 'lam_used': float(d.get('lam_used') or CFG.LAM), 'ladder': d.get('ladder') or [], 'sec': None})
    return out

def _save_thetas(save_dir, rounds):
    d = Path(save_dir) / 'canon'
    d.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(rounds):
        p = d / f'theta_round{i}.json'
        p.write_text(json.dumps({'t': i, 'P': rec['P'], 'D': rec['D'], 'A': rec['A']}, ensure_ascii=False, indent=2), encoding='utf-8')
    return d

def _abs_d_mod(P, D, n_mod):
    import t18_rigid as R
    mp = float(R.quad_side(P)) / max(int(n_mod or 25), 1)
    return float(np.linalg.norm(np.asarray(D, float), axis=-1).mean() / max(mp, 1e-09))

def _arm_from_dec(af, t, extra):
    d = next((x for x in af.get('decs') or [] if int(x['t']) == int(t)), None) or {}
    out = dict(payload=d.get('payload', ''), which=d.get('which', ''), chosen=int(t), e2=af.get('e2'))
    out.update(extra)
    if out.get('cons_how') == 'none':
        out['payload'] = ''
    return out

def _run_chain(u8, quadA, netB, torch, device, save_dir):
    import canon_net
    import t18_rigid as R
    from t24_altfit import altfit, decode_round
    from t24_search_e2 import search_e2
    t_net = time.time()
    P, D, A = canon_net.chain_tight(u8, quadA, netB, torch, device)
    t_chain = time.time()
    g_a = decode_round(u8, P, D, A, lam_ladder=CFG.LAM_LADDER)
    a_arm = dict(payload=g_a.get('payload') or '', which=g_a.get('which') or '', n_agree=int(g_a.get('n_agree') or 0), lam_used=g_a.get('lam_used'), ladder=g_a.get('ladder') or [], sec=time.time() - t_chain)
    t_a = time.time()
    if CFG.EARLY_STOP and a_arm['payload'] and (a_arm['n_agree'] >= CFG.CONSENSUS_MIN):
        out = {'net': 'canon', 'quadA': _arr(quadA), 'theta0': {'P': _arr(P), 'D': _arr(D), 'A': _arr(A)}, 'search': None, 'rounds': [], 'decision': {'status': 'decoded', 'payload': a_arm['payload'], 'cons_how': 'a_arm', 'chosen_round': -1, 'n_agree': a_arm['n_agree'], 'lam_used': a_arm['lam_used']}, 'n_lock': 0, 'rot': 0, 'n_lock_ladder_probe': None, 'flags': {'quad_degenerate': bool(quad_degenerate(P))}, 'timing': {'theta0_chain': t_chain - t_net, 'theta0': t_a - t_chain, 'search': 0.0, 'altfit_rounds': [], 'decode': [], 'total': time.time() - t_net}, 'a': {k: a_arm[k] for k in ('payload', 'which', 'n_agree', 'lam_used', 'ladder', 'sec')}, 'e': {}, 'ec': {}}
        if save_dir is not None:
            nd = Path(save_dir) / 'canon'
            nd.mkdir(parents=True, exist_ok=True)
            (nd / 'result.json').write_text(json.dumps(out, ensure_ascii=False, default=_js, indent=2), encoding='utf-8')
        return out
    se = search_e2(u8, quadA, P, D, A)
    t_se = time.time()
    P2, D2, A2 = se['theta2']
    af = altfit(u8, quadA, P2, D2, A2, rounds=CFG.ALTFIT_ROUNDS, torch=torch, device=device, geo_div=CFG.GEO_DIV, lam_ladder=CFG.LAM_LADDER, early_stop=CFG.EARLY_STOP)
    t_af = time.time()
    Pe, De, Ae = af['theta_sel']
    Pc, Dc, Ac = af['theta_cons']
    n_use = af.get('n_lock') or 25
    extra_e = dict(k=float(se['k']), dphi=float(se['dphi']), cons_how='e2', absD_mod=_abs_d_mod(Pe, De, n_use), e2_star=float(se['e2_star']), P=_arr(Pe), D=_arr(De), A=_arr(Ae), geo_div=int(af.get('geo_div') or CFG.GEO_DIV))
    extra_c = dict(k=float(se['k']), dphi=float(se['dphi']), cons_how=af.get('cons_how', 'e2'), absD_mod=_abs_d_mod(Pc, Dc, n_use), e2_star=float(se['e2_star']), P=_arr(Pc), D=_arr(Dc), A=_arr(Ac), geo_div=int(af.get('geo_div') or CFG.GEO_DIV))
    e_arm = _arm_from_dec(af, af['chosen'], extra_e)
    ec_arm = _arm_from_dec(af, af.get('chosen_cons', af['chosen']), extra_c)
    payload = ec_arm.get('payload') or ''
    cons_how = ec_arm.get('cons_how') or af.get('cons_how', 'e2')
    chosen = int(ec_arm.get('chosen', af.get('chosen_cons', af['chosen'])))
    flags = collect_flags(P, se['k'], se['dphi'], cons_how)
    e2_list = [float(x) for x in af.get('e2') or []]
    e2_chosen = e2_list[chosen] if 0 <= chosen < len(e2_list) else float('nan')
    e2_high = bool(np.isfinite(e2_chosen) and e2_chosen > CFG.E2_ALARM)
    flags['e2_high'] = {'triggered': e2_high, 'e2': e2_chosen, 'threshold': CFG.E2_ALARM}
    if payload:
        status = 'decoded'
    else:
        status = 'rejected_geometry_not_converged' if e2_high else 'rejected'
    ladder = None
    views = af.get('views') or []
    if views and views[0].get('x') is not None:
        geo0 = views[0]['geo']
        core = R.canvas_core_view(views[0]['x'], geo0)
        ladder = finder_ladder_scores(core, geo0['S_core'])
    rounds = _rounds_from_af(af)
    out = {'net': 'canon', 'quadA': _arr(quadA), 'theta0': {'P': _arr(P), 'D': _arr(D), 'A': _arr(A)}, 'search': {'k': float(se['k']), 'dphi': float(se['dphi']), 'e2_star': float(se['e2_star']), 'sec': float(se.get('sec') or t_se - t_a)}, 'rounds': rounds, 'decision': {'status': status, 'payload': payload, 'cons_how': cons_how, 'chosen_round': chosen}, 'n_lock': int(af.get('n_lock') or 0), 'rot': int(af.get('rot') or 0), 'n_lock_ladder_probe': ladder, 'flags': flags, 'timing': {'theta0_chain': t_chain - t_net, 'theta0': t_a - t_chain, 'search': float(se.get('sec') or t_se - t_a), 'altfit_rounds': [float(af.get('sec') or t_af - t_se)], 'decode': [], 'total': time.time() - t_net}, 'a': {}, 'e': {}, 'ec': {'payload': payload, 'cons_how': cons_how, 'chosen': chosen, 'e2': [float(x) for x in af.get('e2') or []], 'e3': [float(x) if x is not None else None for x in af.get('e3') or []], 'n_agree': int(ec_arm.get('n_agree') or 0), 'which': ec_arm.get('which') or '', 'k': float(se['k']), 'dphi': float(se['dphi']), 'geo_div': int(af.get('geo_div') or CFG.GEO_DIV), 'P': _arr(af['theta_cons'][0]), 'D': _arr(af['theta_cons'][1]), 'A': _arr(af['theta_cons'][2])}}
    out['a'] = {'payload': a_arm.get('payload') or '', 'which': a_arm.get('which') or '', 'n_agree': int(a_arm.get('n_agree') or 0), 'lam_used': a_arm.get('lam_used'), 'ladder': a_arm.get('ladder') or [], 'sec': a_arm.get('sec')}
    out['stopped_early'] = bool(af.get('stopped_early'))
    out['e'] = {'payload': e_arm.get('payload') or '', 'cons_how': e_arm.get('cons_how'), 'chosen': e_arm.get('chosen'), 'e2': [float(x) for x in e_arm.get('e2') or af.get('e2') or []]}
    if save_dir is not None:
        nd = _save_thetas(save_dir, rounds)
        (nd / 'result.json').write_text(json.dumps(out, ensure_ascii=False, default=_js, indent=2), encoding='utf-8')
    return out

def run_photo(path_or_gray, save_dir=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    gray, u8, src, t_load = _load_u8(path_or_gray)
    key = src.stem[-6:] if src is not None else 'array'
    direct = {'payload': '', 'n_agree': 0, 'sec': 0.0}
    if CFG.DIRECT_GATE:
        import qr_decoders as Q
        t_d = time.time()
        try:
            dd = Q.standard_decode_image(u8.astype(np.float64) / 255.0)
            direct = {'payload': dd.get('payload') or '', 'n_agree': int(dd.get('n_agree') or 0), 'rung': dd.get('rung_name', ''), 'sec': time.time() - t_d}
        except Exception as exc:
            direct = {'payload': '', 'n_agree': 0, 'err': str(exc), 'sec': time.time() - t_d}
        if direct['payload'] and direct['n_agree'] >= CFG.CONSENSUS_MIN:
            out = {'net': 'canon', 'quadA': None, 'theta0': None, 'search': None, 'rounds': [], 'decision': {'status': 'decoded_direct', 'payload': direct['payload'], 'cons_how': 'direct', 'chosen_round': -1, 'n_agree': direct['n_agree']}, 'direct': direct, 'flags': {}, 'timing': {'load': t_load, 'direct': direct['sec'], 'total': t_load + direct['sec']}, 'input': {'file': str(src) if src is not None else None, 'key': key, 'shape_gray': list(gray.shape), 'shape_u8': list(u8.shape), 'down': CFG.DOWN}, 'versions': {'qrpipe': VERSION, 'config_hash': FROZEN_HASH}}
            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                (Path(save_dir) / 'result.json').write_text(json.dumps(out, ensure_ascii=False, default=_js, indent=2), encoding='utf-8')
            return out
    st = load_runtime()
    torch, device = (st['torch'], st['device'])
    from t20_real import letterbox_quad
    t_q = time.time()
    quadA = letterbox_quad(u8, st['netA'], torch, device)
    t_quad = time.time() - t_q
    t0 = time.time()
    out = _run_chain(u8, quadA, st['netB'], torch, device, save_dir)
    common = {'input': {'file': str(src) if src is not None else None, 'key': key, 'shape_gray': list(gray.shape), 'shape_u8': list(u8.shape), 'down': CFG.DOWN}, 'quadA': _arr(quadA), 'direct': direct, 'versions': _versions(torch), 'timing_common': {'load': t_load, 'quad': t_quad, 'total': time.time() - t0 + t_load + t_quad}}
    out.update(common)
    out['timing']['load'] = t_load
    out['timing']['quad'] = t_quad
    out['timing']['total'] = float(out['timing']['total']) + t_load + t_quad
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / 'result.json').write_text(json.dumps(out, ensure_ascii=False, default=_js, indent=2), encoding='utf-8')
    return out
