from __future__ import annotations
import numpy as np
import t18_rigid as R
import t20_net as NET
N_LAB = 52
SD, SA = (40.0, 50.0)
TRIU = np.triu_indices(8)

def build(torch, device):
    nn = torch.nn
    net = NET.build(torch, 26, variant='v3').to(device)
    old_d, old_g, old_s = (net.path_dense[-1], net.path_glob[-1], net.sig[-1])
    net.path_dense[-1] = nn.Conv2d(old_d.in_channels, 44, 1).to(device)
    net.path_glob[-1] = nn.Linear(old_g.in_features, 44).to(device)
    net.sig[-1] = nn.Linear(old_s.in_features, N_LAB).to(device)
    return net

def load(torch, path, device):
    ck = torch.load(path, map_location=device)
    assert ck.get('layout') == 'canon52', path
    net = build(torch, device)
    net.load_state_dict(ck['state'])
    net.eval()
    return net

def canon_angle_np(P):
    s = np.roll(P, -1, 0) - P
    a = np.arctan2(s[:, 1], s[:, 0])
    z = (np.hypot(s[:, 0], s[:, 1]) * np.exp(4j * a)).sum()
    return -np.angle(z) / 4.0

def decode_np(mu):
    mu = np.asarray(mu, float)
    M = np.zeros((8, 8))
    M[TRIU] = mu[8:44]
    M = M + M.T - np.diag(np.diag(M))
    w, V = np.linalg.eigh(M)
    v = max(w[-1], 0.0) * V[:, -1]
    return (mu[:8].reshape(4, 2), (v / SD).reshape(4, 2), (mu[44:52] / SA).reshape(4, 2))

def chain_tight(u8, quad, net, torch, device):
    import cv2
    q = np.asarray(quad, float)
    h, w = u8.shape
    for _ in range(2):
        q = q[R.order_quad(q)]
        ang = canon_angle_np(q)
        c = q.mean(0)
        M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), -np.degrees(ang), 1.0)
        im = cv2.warpAffine(u8, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        qr = q @ M[:, :2].T + M[:, 2]
        x0, y0, side = R.window_for_quad(qr)
        crop = cv2.resize(R.crop_window(im, (x0, y0, side)), (R.NET_B, R.NET_B), interpolation=cv2.INTER_AREA)
        net.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            mu = net(NET.standardise(torch, torch.from_numpy(crop)[None].to(device)))['mu'].float().cpu().numpy()[0]
        Pr, Dr, Ar = decode_np(mu)
        Pr = Pr * side + np.array([x0, y0], float)
        Ri = np.linalg.inv(M[:, :2])
        q, D, A = ((Pr - M[:, 2]) @ Ri.T, Dr * side @ Ri.T, Ar * side @ Ri.T)
    return (q, D, A)
