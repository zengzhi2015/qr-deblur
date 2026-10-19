from __future__ import annotations
import numpy as np
import t18_rigid as R
import t20_net as NET

def letterbox_quad(u8, netA, torch, device):
    import cv2
    h, w = u8.shape
    S = max(h, w)
    sq = np.full((S, S), int(np.median(u8)), np.uint8)
    oy, ox = ((S - h) // 2, (S - w) // 2)
    sq[oy:oy + h, ox:ox + w] = u8
    thumb = cv2.resize(sq, (R.NET, R.NET), interpolation=cv2.INTER_AREA)
    mu, _ = NET.predict(netA, torch, thumb[None], device)
    return R.decode_quad(mu[0], float(S)) - np.array([ox, oy], float)
