from __future__ import annotations
import numpy as np
import t18_rigid as R
LAM, ITERS = (0.001, 40)
ITERS_LONG, LONG_PX = (80, 24.0)

def path_of(PDA, W):
    Pm, D, A = PDA
    p9 = R.path_from_PDA(Pm, D, A, 9)
    if not np.isfinite(p9).all() or p9.min() < -W or p9.max() > 2 * W:
        return None
    return R.path_from_PDA(Pm, D, A, R.n_inverse_samples(p9))

def n_iters(path):
    ext = float(np.linalg.norm(np.asarray(path)[-1] - np.asarray(path)[0], axis=-1).max())
    return ITERS_LONG if ext >= LONG_PX else ITERS
