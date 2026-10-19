import numpy as np
LOGSIG_RANGE = 6.0

def build(torch, n_lab, width=(24, 48, 96, 192, 192), blocks=(1, 2, 2, 1, 1), dec=64, variant='v2'):
    if variant == 'v3':
        width, blocks, dec = ((32, 64, 128, 256, 256), (1, 2, 3, 2, 1), 96)
    cin = 3 if variant == 'v3' else 1
    nn = torch.nn
    F = torch.nn.functional

    def cba(a, b, s=1):
        return nn.Sequential(nn.Conv2d(a, b, 3, s, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

    class Res(nn.Module):

        def __init__(self, c):
            super().__init__()
            self.a = cba(c, c)
            self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

        def forward(self, x):
            return F.relu(x + self.b(self.a(x)), inplace=True)

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.with_path = n_lab > 8
            c0, c1, c2, c3, c4 = width
            self.cin = cin
            self.stem = nn.Sequential(cba(cin, 16, 1), cba(16, c0, 2), *[Res(c0) for _ in range(blocks[0])])
            self.s1 = nn.Sequential(cba(c0, c1, 2), *[Res(c1) for _ in range(blocks[1])])
            self.s2 = nn.Sequential(cba(c1, c2, 2), *[Res(c2) for _ in range(blocks[2])])
            self.s3 = nn.Sequential(cba(c2, c3, 2), *[Res(c3) for _ in range(blocks[3])])
            self.s4 = nn.Sequential(cba(c3, c4, 2), *[Res(c4) for _ in range(blocks[4])])
            self.l1, self.l2, self.l3, self.l4 = (nn.Conv2d(c, dec, 1) for c in (c1, c2, c3, c4))
            self.fuse = nn.Sequential(cba(dec, dec), cba(dec, dec))
            self.heat = nn.Sequential(cba(dec, dec), nn.Conv2d(dec, 4, 1))
            self.reg = nn.Sequential(cba(dec, dec), nn.Conv2d(dec, 4 * 2, 1))
            self.sig = nn.Sequential(nn.Linear(c4, 128), nn.ReLU(inplace=True), nn.Linear(128, n_lab))
            if self.with_path:
                self.att = nn.Sequential(cba(dec, dec), nn.Conv2d(dec, 1, 1))
                self.path_dense = nn.Sequential(cba(dec, dec), nn.Conv2d(dec, 18, 1))
                self.path_glob = nn.Sequential(nn.Linear(c4, 128), nn.ReLU(inplace=True), nn.Linear(128, 18))
            with torch.no_grad():
                self.heat[-1].bias.fill_(-2.0)
                self.reg[-1].weight.mul_(0.1)
                self.reg[-1].bias.zero_()
                self.sig[-1].weight.mul_(0.1)
                self.sig[-1].bias.zero_()
                if self.with_path:
                    self.path_dense[-1].weight.mul_(0.1)
                    self.path_dense[-1].bias.zero_()
                    self.path_glob[-1].weight.mul_(0.1)
                    self.path_glob[-1].bias.zero_()

        def forward(self, x):
            if self.cin == 3:
                N_, _, H_, W_ = x.shape
                yy = torch.linspace(-1, 1, H_, device=x.device, dtype=x.dtype).view(1, 1, H_, 1).expand(N_, 1, H_, W_)
                xx = torch.linspace(-1, 1, W_, device=x.device, dtype=x.dtype).view(1, 1, 1, W_).expand(N_, 1, H_, W_)
                x = torch.cat([x, xx, yy], 1).contiguous(memory_format=torch.channels_last)
            f1 = self.s1(self.stem(x))
            f2 = self.s2(f1)
            f3 = self.s3(f2)
            f4 = self.s4(f3)
            p = self.l4(f4)
            p = self.l3(f3) + F.interpolate(p, size=f3.shape[-2:], mode='nearest')
            p = self.l2(f2) + F.interpolate(p, size=f2.shape[-2:], mode='nearest')
            p = self.l1(f1) + F.interpolate(p, size=f1.shape[-2:], mode='nearest')
            p = self.fuse(p)
            heat = self.heat(p)
            reg = self.reg(p)
            N, _, h, w = heat.shape
            prob = torch.softmax(heat.flatten(2).float(), -1)
            gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32), torch.arange(w, device=x.device, dtype=torch.float32), indexing='ij')
            gx = ((gx + 0.5) / w).reshape(1, 1, -1)
            gy = ((gy + 0.5) / h).reshape(1, 1, -1)
            regf = reg.flatten(2).float()
            off = torch.stack([(prob[:, k, None] * regf[:, 2 * k:2 * k + 2]).sum(-1) for k in range(4)], 1)
            px = (prob * gx).sum(-1) + off[..., 0] / w
            py = (prob * gy).sum(-1) + off[..., 1] / h
            Pm = torch.stack([px, py], -1).reshape(N, 8)
            parts = [Pm]
            g = f4.float().mean((-2, -1))
            if self.with_path:
                a = torch.softmax(self.att(p).flatten(2).float(), -1)
                pooled = (a * self.path_dense(p).flatten(2).float()).sum(-1)
                parts.append(pooled + self.path_glob(g))
            mu = torch.cat(parts, 1)
            logsig = LOGSIG_RANGE * torch.tanh(self.sig(g) / LOGSIG_RANGE)
            return dict(mu=mu, logsig=logsig, heat=heat.float())
    return Net()

def standardise(torch, xb_u8):
    x = xb_u8.float() / 255.0
    mu = x.mean((-2, -1), keepdim=True)
    sd = x.std((-2, -1), keepdim=True)
    return ((x - mu) / torch.clamp(sd, min=1e-06))[:, None]

def predict(net, torch, X_u8, device, bs=128):
    net.eval()
    mus, sigs = ([], [])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        for s in range(0, len(X_u8), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X_u8[s:s + bs])).to(device)
            o = net(standardise(torch, xb))
            mus.append(o['mu'].float().cpu().numpy())
            sigs.append(torch.exp(o['logsig']).float().cpu().numpy())
    return (np.concatenate(mus), np.concatenate(sigs))

def load(torch, path, device):
    ck = torch.load(path, map_location=device)
    net = build(torch, int(ck['n_lab']), variant=ck['variant']).to(device)
    net.load_state_dict(ck['state'])
    net.eval()
    return net
