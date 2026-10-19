# Third-party notices

The code in this repository is under the MIT License (see `LICENSE`).
The pipeline imports the runtime dependencies listed below. Each installed distribution ships its
own license text at the path given.

## Main environment

| Package | Version tested | License | License file in installed package |
|---|---|---|---|
| pyzbar | 0.1.9 | MIT | `pyzbar-0.1.9.dist-info/LICENSE.txt` |
| zxing-cpp | 3.1.1 | Apache-2.0 | `zxing_cpp-3.1.1.dist-info/licenses/LICENSE` |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | `opencv_python_headless-5.0.0.93.dist-info/LICENSE.txt` (+ `LICENSE-3RD-PARTY.txt`) |
| numpy | 2.5.2 | BSD | `numpy-2.5.2.dist-info/licenses/` |
| torch | 2.11.0+cu128 | BSD-3-Clause | `torch-2.11.0+cu128.dist-info/LICENSE` |
| pillow | 12.3.0 | HPND-derived (see package) | `pillow-12.3.0.dist-info/licenses/` |
| qrcode | 8.2 | BSD | `qrcode-8.2.dist-info/LICENSE` |
| scipy | 1.18.1 | BSD | installed package metadata |
| scikit-image | 0.26.0 | BSD | installed package metadata |

## WeChat decoder environment

| Package | Version tested | License | License file in installed package |
|---|---|---|---|
| opencv-contrib-python-headless | 5.0.0.93 | Apache-2.0 | `opencv_contrib_python_headless-5.0.0.93.dist-info/LICENSE.txt` (+ `LICENSE-3RD-PARTY.txt`) |
| numpy | 2.5.3 | BSD | `licenses/` in the package |

The WeChat decoder is called as `cv2.wechat_qrcode_WeChatQRCode()` without separate model files.

## Network weights

The two weight files (`experiments/probe24/t20/model_A2_v2r.pt` and
`experiments/pathnet_fix/model_canrollw_long.pt`) were trained by the authors on synthetic images
produced by the authors' own generator. No third-party pretrained checkpoint was loaded.
The weights are distributed separately from the download link in `README.md` under the same MIT License as the code.

## Database

The photographs are not part of this repository. They are distributed separately, together with
their own license file, from the download location given in `README.md`.
