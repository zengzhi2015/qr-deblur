import hashlib
import json
import os
import sys
from pathlib import Path
WORK = Path(__file__).resolve().parents[1]
PROBE24 = WORK / 'scripts' / 'probe24'
PATHNET_FIX = WORK / 'scripts' / 'pathnet_fix'
WEIGHT_A = WORK / 'experiments' / 'probe24' / 't20' / 'model_A2_v2r.pt'
WEIGHT_B_CANON = WORK / 'experiments' / 'pathnet_fix' / 'model_canrollw_long.pt'
os.environ.setdefault('T18_DEVICE', 'cuda')
for _p in (PROBE24, PATHNET_FIX):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
VERSION = '1.2.0'
DOWN = 2
ALTFIT_ROUNDS = 3
GEO_DIV = 1
LAM = 0.001
CONSENSUS_MIN = 2
DIRECT_GATE = True
LAM_LADDER = (0.001, 0.03, 0.01, 0.003)
EARLY_STOP = True
E2_ALARM = 0.15
GPU_MEM_FRACTION = 0.5
QUAD_MIN_CORNER_FRAC = 0.2
QUAD_OPP_RATIO = 3.0
SEARCH_K_EDGES = (0.7, 1.35)
SEARCH_DPHI_EDGE = 10.0
CANON_NET_PY = PATHNET_FIX / 'canon_net.py'

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
WEIGHT_SHA256 = {p.name: _sha256_file(p) if p.exists() else '' for p in (WEIGHT_A, WEIGHT_B_CANON, CANON_NET_PY)}
WEIGHT_SHA256_EXPECTED = {'model_A2_v2r.pt': '2d1e1449352d5d52524e132b8374cfa66e68ee912b11f964ea3a9df763adf9c4', 'model_canrollw_long.pt': '4e3f70f7188e27513f80dc28e333d5e07b5d4bcbb2c7f9770e74aea4a7409d7a', 'canon_net.py': '5c901e51b0b0e5a8abd1514a200613090104a6899fce84f7d7fc01f12d3389e3'}
FROZEN = {'version': VERSION, 'down': DOWN, 'weight_a': WEIGHT_A.name, 'weight_b': WEIGHT_B_CANON.name, 'weight_sha256': WEIGHT_SHA256_EXPECTED, 'altfit_rounds': ALTFIT_ROUNDS, 'geo_div': GEO_DIV, 'lam': LAM, 'consensus_min': CONSENSUS_MIN, 'direct_gate': DIRECT_GATE, 'lam_ladder': list(LAM_LADDER), 'early_stop': EARLY_STOP, 'e2_alarm': E2_ALARM, 'gpu_mem_fraction': GPU_MEM_FRACTION, 'quad_min_corner_frac': QUAD_MIN_CORNER_FRAC, 'quad_opp_ratio': QUAD_OPP_RATIO}
FROZEN_HASH = hashlib.sha256(json.dumps(FROZEN, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
