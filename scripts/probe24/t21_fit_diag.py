from __future__ import annotations
import numpy as np
import real_capture_common as C
import t18_rigid as R
N_SEG = 32
ITERS_PRE, ITERS_GEO = (40, 150)
DWELL_SMOOTH = 2e-05
FIT_SIDE = 420
SIDE_PEN = 0.001

def dlt_h(torch, src, dst):
    n = src.shape[0]
    x, y = (src[..., 0], src[..., 1])
    u, v = (dst[..., 0], dst[..., 1])
    z, o = (torch.zeros_like(x), torch.ones_like(x))
    r1 = torch.stack([x, y, o, z, z, z, -u * x, -u * y], -1)
    r2 = torch.stack([z, z, z, x, y, o, -v * x, -v * y], -1)
    Amat = torch.cat([r1, r2], 1)
    b = torch.cat([u, v], 1)
    h = torch.linalg.solve(Amat, b)
    return torch.cat([h, torch.ones(n, 1, device=h.device, dtype=h.dtype)], 1).reshape(n, 3, 3)

class Model:

    def __init__(self, torch, dev, canvas, canvas_mask, core_canvas, crop_box, photo_crop, Pm, D, A, n_seg=N_SEG):
        self.torch, self.dev = (torch, dev)
        self.canvas = torch.from_numpy(canvas.astype(np.float32)).to(dev)[None, None]
        self.cmask = torch.from_numpy(canvas_mask.astype(np.float32)).to(dev)[None, None]
        self.core = torch.from_numpy(core_canvas.astype(np.float32)).to(dev)
        x0, y0, x1, y1 = crop_box
        self.off = torch.tensor([x0, y0], dtype=torch.float32, device=dev)
        H, W = (y1 - y0, x1 - x0)
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        self.base = torch.stack([xs + x0, ys + y0, torch.ones_like(xs)], -1).reshape(-1, 3)
        self.hw = (H, W)
        self.nx = xs / max(W - 1, 1) * 2 - 1
        self.ny = ys / max(H - 1, 1) * 2 - 1
        self.y = torch.from_numpy(photo_crop.astype(np.float32)).to(dev)
        self.s = torch.linspace(-1, 1, n_seg, device=dev)
        self.P = torch.tensor(np.asarray(Pm, np.float32), device=dev, requires_grad=True)
        self.D = torch.tensor(np.asarray(D, np.float32), device=dev, requires_grad=True)
        self.A = torch.tensor(np.asarray(A, np.float32), device=dev, requires_grad=True)
        self.wl = torch.zeros(n_seg, device=dev, requires_grad=True)
        self.ill = torch.zeros(5, device=dev, requires_grad=True)
        lin = np.power(np.clip(photo_crop, 0, 1), R.GAMMA)
        self.lev = torch.tensor([float(np.percentile(lin, 3)), float(np.percentile(lin, 97))], device=dev, requires_grad=True)
        self.side0 = float(R.quad_side(np.asarray(Pm, float)))

    def weights(self):
        return self.torch.softmax(self.wl, 0)

    def path(self):
        s = self.s[:, None, None]
        return self.P[None] + s * self.D[None] + 0.5 * s * s * self.A[None]

    def render(self, canvas=None, mask_only=False):
        torch = self.torch
        Hs = dlt_h(torch, self.core[None].expand(self.s.shape[0], -1, -1), self.path())
        Hinv = torch.linalg.inv(Hs)
        p = self.base @ Hinv.transpose(1, 2)
        z = torch.where(p[..., 2:3].abs() < 1e-08, torch.full_like(p[..., 2:3], 1e-08), p[..., 2:3])
        xy = p[..., :2] / z
        h, w = self.canvas.shape[-2:]
        g = torch.stack([(xy[..., 0] + 0.5) / w * 2 - 1, (xy[..., 1] + 0.5) / h * 2 - 1], -1).reshape(-1, *self.hw, 2)
        src = self.cmask if mask_only else canvas if canvas is not None else self.canvas
        out = torch.nn.functional.grid_sample(src.expand(g.shape[0], -1, -1, -1), g, mode='bilinear', padding_mode='zeros', align_corners=False)[:, 0]
        if mask_only:
            return out.min(0).values
        wts = self.weights()[:, None, None]
        return (out * wts).sum(0)

    def forward(self):
        torch = self.torch
        code = self.render()
        b, wh = (self.lev[0], self.lev[1])
        lin = b + (wh - b) * code
        ill = 1 + self.ill[0] * self.nx + self.ill[1] * self.ny + self.ill[2] * self.nx ** 2 + self.ill[3] * self.nx * self.ny + self.ill[4] * self.ny ** 2
        lin = torch.clamp(lin * ill, 0, 1)
        return torch.pow(lin + 1e-06, 1.0 / R.GAMMA)

    def _blur(self, img, sigma):
        if sigma <= 0:
            return img
        torch = self.torch
        r = int(np.ceil(3 * sigma))
        xs = torch.arange(-r, r + 1, device=self.dev, dtype=torch.float32)
        k = torch.exp(-0.5 * (xs / sigma) ** 2)
        k = k / k.sum()
        x = img[None, None]
        x = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (r, r, 0, 0), mode='replicate'), k[None, None, None, :])
        x = torch.nn.functional.conv2d(torch.nn.functional.pad(x, (0, 0, r, r), mode='replicate'), k[None, None, :, None])
        return x[0, 0]

    def loss(self, mask, sigma=0.0):
        torch = self.torch
        yh = self.forward()
        r = self._blur(yh, sigma) - self._blur(self.y, sigma)
        hub = torch.nn.functional.huber_loss(r, torch.zeros_like(r), reduction='none', delta=0.05)
        w = self.weights()
        smooth = ((w[2:] - 2 * w[1:-1] + w[:-2]) ** 2).sum() * len(w) ** 2
        side = torch.stack([(self.P[(k + 1) % 4] - self.P[k]).norm() for k in range(4)]).mean()
        side_pen = ((side / self.side0 - 1.0) / 0.25) ** 2 * SIDE_PEN
        return ((hub * mask).sum() / mask.sum().clamp(min=1.0) + DWELL_SMOOTH * smooth + side_pen, r)

    def fit(self, params, iters, lrs, mask, sigma=0.0):
        torch = self.torch
        opt = torch.optim.Adam([{'params': [p], 'lr': lr} for p, lr in zip(params, lrs)])
        for _ in range(iters):
            opt.zero_grad()
            l, _ = self.loss(mask, sigma)
            l.backward()
            opt.step()

def make_canvas(payload, version, ec, quiet, mp):
    m = C._core_matrix(payload, version, ec)
    n = m.shape[0]
    pad = quiet + 2
    S = (n + 2 * pad) * mp
    canvas = np.ones((S, S), np.float32)
    core = np.kron((~m).astype(np.float32), np.ones((mp, mp), np.float32))
    canvas[pad * mp:(pad + n) * mp, pad * mp:(pad + n) * mp] = core
    mask = np.zeros((S, S), np.float32)
    q0, q1 = ((pad - quiet) * mp, (pad + n + quiet) * mp)
    mask[q0:q1, q0:q1] = 1.0
    core_corners = np.array([[pad * mp, pad * mp], [(pad + n) * mp, pad * mp], [(pad + n) * mp, (pad + n) * mp], [pad * mp, (pad + n) * mp]], float) - 0.5
    return (canvas, mask, core_corners, n)

def rot_canvas(canvas, k):
    return np.ascontiguousarray(np.rot90(canvas, k))
