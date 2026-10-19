from __future__ import annotations
import sys
import numpy as np
from .config import PROBE24
if str(PROBE24) not in sys.path:
    sys.path.insert(0, str(PROBE24))
PAYLOAD = 'RID-0090|CTRL'
LABELS = {'noise': 'noise canvas 300x300', 'badgrid': 'wrong_grid (true finders + random data, 25x25)'}

def _bad_grid(rng):
    m25 = rng.random((25, 25)) < 0.5
    fp = np.ones((7, 7), bool)
    fp[1:6, 1:6] = False
    fp[2:5, 2:5] = True
    for r0, c0 in ((0, 0), (0, 18), (18, 0)):
        m25[r0:r0 + 7, c0:c0 + 7] = fp
    xc = np.pad(np.kron((~m25).astype(np.float32), np.ones((10, 10), np.float32)), 40, constant_values=1.0)
    return (xc, dict(S=330, m=40.0, S_core=250.0))

def run_controls(n_neg=100):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    import qr_decoders as Q
    from t21_fit_diag import make_canvas
    print('## decoder_chain 10.1 control', flush=True)
    canvas, _, _, n = make_canvas(PAYLOAD, 10, 'Q', quiet=4, mp=10)
    pad = 4 + 2
    geo = dict(S=int(canvas.shape[0]), m=float(pad * 10), S_core=float(n * 10))
    rng = np.random.default_rng(0)
    print('| input | consensus | n_agree | tier |', flush=True)
    print('|---|---|---|---|', flush=True)
    for sd in (0.0, 0.05, 0.1):
        xn = canvas + (rng.normal(0, sd, canvas.shape) if sd else 0)
        r = Q.standard_decode(np.clip(xn, 0, 1), geo)
        ok = r['payload'] == PAYLOAD
        tag = 'ok' if ok else 'wrong' if r['payload'] else 'empty'
        print(f"| truth binary canvas + noise std={sd:g} | {tag} | {r['n_agree']} | {r.get('rung_name', '')} |", flush=True)
        if sd == 0.0 and (not ok):
            print('decoder_chain_failed', flush=True)
            return {'pos_ok': False, 'positive': None, 'negative': {}}
    print('positive_control (truth binary): 1.0', flush=True)
    print('| input | count | nonempty_rate | wrong_rate |', flush=True)
    print('|---|---|---|---|', flush=True)
    negs = {}
    for kind in ('noise', 'badgrid'):
        ne = wr = 0
        for _ in range(n_neg):
            if kind == 'noise':
                r = Q.standard_decode(rng.random((300, 300)), dict(S=300, m=20.0, S_core=260.0))
            else:
                r = Q.standard_decode(*_bad_grid(rng))
            s = r['payload']
            ne += bool(s)
            wr += bool(s) and s != PAYLOAD
        print(f'| {LABELS[kind]} | {n_neg} | {ne / n_neg:.3f} | {wr / n_neg:.3f} |', flush=True)
        negs[kind] = {'n': n_neg, 'nonempty': round(ne / n_neg, 3), 'wrong': round(wr / n_neg, 3)}
    return {'pos_ok': True, 'positive': '1.0', 'negative': negs}
