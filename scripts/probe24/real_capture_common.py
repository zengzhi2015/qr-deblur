from pathlib import Path
import numpy as np

def load_gray(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ('RGBA', 'LA'):
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im.convert('RGBA')).convert('RGB')
    else:
        im = im.convert('RGB')
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

def downscale(g: np.ndarray, f: int) -> np.ndarray:
    if f <= 1:
        return g
    h, w = g.shape
    h2, w2 = (h - h % f, w - w % f)
    return g[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean((1, 3))

def _core_matrix(payload: str, version: int, ec: str) -> np.ndarray:
    import qrcode
    ecc = {'L': qrcode.constants.ERROR_CORRECT_L, 'M': qrcode.constants.ERROR_CORRECT_M, 'Q': qrcode.constants.ERROR_CORRECT_Q, 'H': qrcode.constants.ERROR_CORRECT_H}[ec]
    q = qrcode.QRCode(version=version, error_correction=ecc, box_size=1, border=0)
    q.add_data(payload)
    q.make(fit=False)
    return np.array(q.get_matrix(), dtype=bool)
