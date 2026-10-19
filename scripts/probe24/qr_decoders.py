import os
import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np
import t18_rigid as R
WORK = Path(__file__).resolve().parents[2]
VENV_WX = WORK / 'venv_wx'
WECHAT_WORKER_PY = Path(__file__).resolve().parent / 'wechat_worker.py'
_worker = None

def _wechat_python() -> Path:
    env = os.environ.get('QRD_WECHAT_PYTHON', '').strip()
    if env:
        return Path(env)
    return VENV_WX / ('Scripts' if os.name == 'nt' else 'bin') / ('python.exe' if os.name == 'nt' else 'python')

def _wechat_worker():
    global _worker
    if _worker is None or _worker.poll() is not None:
        py = _wechat_python()
        if not py.exists():
            raise RuntimeError(f'wechat python missing: {py}')
        if not WECHAT_WORKER_PY.exists():
            raise RuntimeError(f'wechat worker script missing: {WECHAT_WORKER_PY}')
        _worker = subprocess.Popen([str(py), str(WECHAT_WORKER_PY)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1)
    return _worker

def read_wechat(u8: np.ndarray) -> str:
    w = _wechat_worker()
    p = Path(tempfile.gettempdir()) / f'wx_{os.getpid()}.png'
    cv2.imwrite(str(p), u8)
    w.stdin.write(str(p) + '\n')
    w.stdin.flush()
    return w.stdout.readline().rstrip('\n')

def read_zxing(u8: np.ndarray) -> str:
    import zxingcpp
    r = zxingcpp.read_barcodes(np.ascontiguousarray(u8))
    return r[0].text if r else ''

def _u8(c):
    return (np.clip(c, 0, 1) * 255).round().astype(np.uint8)
DECODERS = ('pyzbar', 'zxing', 'wechat')

def _read_pyzbar(u8):
    from qr_deblur_theory.decode import decode_pyzbar
    return decode_pyzbar(u8.astype(np.float64) / 255.0)['decoded_payload']
_READERS = {'pyzbar': _read_pyzbar, 'zxing': read_zxing, 'wechat': read_wechat}

def _consensus(readings):
    if not readings:
        return ('', 0, '')
    by = {}
    for dec, i, name, s in readings:
        e = by.setdefault(s, dict(decs=set(), rung=i, rung_name=name))
        e['decs'].add(dec)
        if i < e['rung']:
            e['rung'], e['rung_name'] = (i, name)
    s, e = max(by.items(), key=lambda kv: (len(kv[1]['decs']), -kv[1]['rung']))
    return (s, len(e['decs']), e['rung_name'])

def standard_decode_candidates(cands, decoders=DECODERS):
    readings = []
    per = {}
    for dec in decoders:
        reader = _READERS[dec]
        for i, (name, c) in enumerate(cands):
            s = reader(_u8(c))
            if s:
                readings.append((dec, i, name, s))
                per.setdefault(dec, (s, i, name))
                break
    payload, n_agree, rung_name = _consensus(readings)
    out = dict(payload=payload, n_agree=n_agree, rung_name=rung_name, readings=readings)
    for dec in decoders:
        s, i, name = per.get(dec, ('', -1, ''))
        out[dec] = s
        out[dec + '_rung'] = i
        out[dec + '_rung_name'] = name
    return out

def standard_decode(x, geo):
    info = R.decode_canvas_info_levels(x, geo)
    out = standard_decode_candidates(R.ladder_candidates(info['vn'], geo))
    out.update(u8=_u8(info['vn']), black=info['black'], white=info['white'], n_core_est=info['n_core'], finder_score=info['finder_score'])
    return out

def standard_decode_image(img01):
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu
    v = np.clip(np.asarray(img01, np.float64), 0, 1)
    cands = [('raw', v)]
    for s in (1.0, 2.0, 3.0):
        cands.append((f'gauss sigma={s:.2f}', np.clip(gaussian_filter(v, s), 0, 1)))
    g = cands[2][1]
    if g.max() - g.min() > 1e-06:
        cands.append(('otsu', (g > threshold_otsu(g)).astype(np.float64)))
    return standard_decode_candidates(cands)
