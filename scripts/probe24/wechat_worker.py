import sys
import cv2
det = cv2.wechat_qrcode_WeChatQRCode()
for line in sys.stdin:
    p = line.strip()
    if not p:
        continue
    try:
        g = cv2.imread(p, 0)
        res, _ = det.detectAndDecode(g)
        s = next((t for t in res if t), '')
    except Exception:
        s = ''
    sys.stdout.write(s.replace('\n', ' ') + '\n')
    sys.stdout.flush()
