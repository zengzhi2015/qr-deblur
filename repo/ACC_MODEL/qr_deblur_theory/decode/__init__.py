from __future__ import annotations
import numpy as np

def to_u8_rgb(gray: np.ndarray) -> np.ndarray:
    g = np.clip(gray, 0, 1)
    u8 = (g * 255.0).round().astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)

def decode_pyzbar(image_float: np.ndarray) -> dict:
    from pyzbar import pyzbar
    rgb = to_u8_rgb(image_float)
    objs = pyzbar.decode(rgb)
    if not objs:
        return {'decoded_payload': '', 'decoded_nonempty': False, 'decoder_pass': False, 'error_code': 'empty'}
    payload = objs[0].data.decode('utf-8', 'replace')
    return {'decoded_payload': payload, 'decoded_nonempty': True, 'decoder_pass': True, 'error_code': ''}
