from __future__ import annotations
import numpy as np
from . import config as CFG

def _side_lengths(P):
    P = np.asarray(P, float).reshape(4, 2)
    return np.array([np.linalg.norm(P[(i + 1) % 4] - P[i]) for i in range(4)], float)

def quad_degenerate(P) -> bool:
    P = np.asarray(P, float).reshape(4, 2)
    if not np.isfinite(P).all():
        return True
    sides = _side_lengths(P)
    avg = float(sides.mean())
    if avg <= 1e-09:
        return True
    for i in range(4):
        for j in range(i + 1, 4):
            if float(np.linalg.norm(P[i] - P[j])) < CFG.QUAD_MIN_CORNER_FRAC * avg:
                return True
    r02 = max(sides[0], sides[2]) / max(min(sides[0], sides[2]), 1e-09)
    r13 = max(sides[1], sides[3]) / max(min(sides[1], sides[3]), 1e-09)
    if r02 > CFG.QUAD_OPP_RATIO or r13 > CFG.QUAD_OPP_RATIO:
        return True
    crosses = []
    for i in range(4):
        a = P[(i + 1) % 4] - P[i]
        b = P[(i + 2) % 4] - P[(i + 1) % 4]
        crosses.append(float(a[0] * b[1] - a[1] * b[0]))
    signs = [np.sign(c) for c in crosses if abs(c) > 1e-09]
    if signs and min(signs) < 0 < max(signs):
        return True
    return False

def search_at_edge(k, dphi) -> bool:
    k = float(k)
    dphi = float(dphi)
    if any((abs(k - e) < 1e-09 for e in CFG.SEARCH_K_EDGES)):
        return True
    if abs(abs(dphi) - CFG.SEARCH_DPHI_EDGE) < 1e-09:
        return True
    return False

def no_consensus(cons_how: str) -> bool:
    return str(cons_how or '') in ('e2', 'none')

def collect_flags(P, k, dphi, cons_how) -> dict:
    return {'quad_degenerate': bool(quad_degenerate(P)), 'search_at_edge': bool(search_at_edge(k, dphi)), 'no_consensus': bool(no_consensus(cons_how))}

def finder_ladder_scores(v, S_core):
    import t18_rigid as R
    rows = []
    for n in R.N_CORE_LADDER:
        lv = R.finder_levels(v, S_core, n_core=n)
        rows.append({'n': int(n), 'score': float(lv['score'])})
    rows.sort(key=lambda r: -r['score'])
    gap = None
    if len(rows) >= 2:
        gap = float(rows[0]['score'] - rows[1]['score'])
    return {'best': rows[0] if rows else None, 'second': rows[1] if len(rows) > 1 else None, 'gap': gap, 'all': rows}
