# qr-deblur: Decoding Motion-Blurred QR Codes

Official inference implementation for **Decoding Motion-Blurred QR Codes: Motion-Parameter Error Tolerance and Model-Based Recovery** (version `1.2.0`).

- **Code repository:** [https://github.com/zengzhi2015/qr-deblur](https://github.com/zengzhi2015/qr-deblur) (MIT License)
- **Network weights (separate download):** [Google Drive (`QR_DEB_TRL_WEIGHTS`)](https://drive.google.com/drive/folders/16fElOj7ZKO8GkIW4ZtmDDfpUHBI10mL3?usp=sharing) (MIT License)
- **Photograph & sensor benchmark (separate download):** [Google Drive (`QR_DEB_TRL_DATASET`)](https://drive.google.com/drive/folders/1tKBJlFb34EyO09QVZq4hT7_R2pXC0vUD) (CC BY 4.0)

The two network checkpoints and the 955-photograph handheld benchmark are distributed via the Google Drive folders above.

---

## Pipeline Overview

For each motion-blurred photograph, `qrpipe` runs five stages in sequence:

```text
Input Photograph
  └─> [0] Direct Gate       Try pyzbar / ZXing-C++ / WeChat on the input image
  └─> [1] Corner Locator    Predict the four QR corners (model_A2_v2r.pt)
  └─> [2] Path Network      Predict an initial 52-D quadratic corner trajectory (model_canrollw_long.pt)
  └─> [3] Tolerance Search  Grid search over blur length & direction scored by fixed QR function modules
  └─> [4] Alternating Fit   Three rounds of joint binary-module estimation and trajectory refitting
  └─> [5] Inversion & Read  Regularised linear inversion + 2-of-3 decoder agreement (pyzbar, ZXing-C++, WeChat)
```

A decoded string is accepted only when at least **two of the three** independent decoders return the identical payload.

---

## Quick Start

| Step | Command / Action |
| :--- | :--- |
| **1. Clone** | `git clone https://github.com/zengzhi2015/qr-deblur.git && cd qr-deblur` |
| **2. Environments** | Create `venv` (PyTorch + main deps) and `venv_wx` (WeChat OpenCV contrib) — see [§2](#2-install-the-environments) |
| **3. Weights** | Download [`QR_DEB_TRL_WEIGHTS`](https://drive.google.com/drive/folders/16fElOj7ZKO8GkIW4ZtmDDfpUHBI10mL3?usp=sharing) and copy `experiments/` into the repo root — see [§3](#3-download-and-verify-the-weights) |
| **4. Self-test** | `python -m qrpipe --controls` |
| **5. Run one photo** | `python -m qrpipe path/to/photo.jpg --out out/one` |
| **6. Run benchmark** | `python -m qrpipe --database path/to/QR_DEB_TRL_DATASET --out out/eval` |

---

## 1. Clone the Repository

```bash
git clone https://github.com/zengzhi2015/qr-deblur.git
cd qr-deblur
```

Run all commands below from this repository root (the directory containing `qrpipe/`).

---

## 2. Install the Environments

Tested with **Python 3.13** on Windows with an NVIDIA GPU and CUDA (`torch 2.11.0+cu128`). The pipeline uses CUDA by default.

### 2.1 Main environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> **Linux note:** `pyzbar` requires the system shared library `libzbar0` (e.g. `sudo apt install libzbar0`). On Windows the DLL is bundled with the wheel.

### 2.2 WeChat decoder environment

The WeChat QR decoder requires `opencv-contrib-python-headless`, which conflicts with `opencv-python-headless` if installed in the same virtual environment. It is therefore isolated in a second virtual environment and invoked via a persistent worker subprocess:

```bash
python -m venv venv_wx
# Windows:
venv_wx\Scripts\pip install -r requirements_wechat.txt
# Linux / macOS:
# venv_wx/bin/pip install -r requirements_wechat.txt
```

If this environment is created at `./venv_wx` inside the repository root, `qrpipe` locates it automatically on both Windows and Linux/macOS. If you place it elsewhere, set `QRD_WECHAT_PYTHON` to its Python executable:

```powershell
# PowerShell
$env:QRD_WECHAT_PYTHON = "D:\envs\wx\Scripts\python.exe"
```

```bash
# Linux / macOS
export QRD_WECHAT_PYTHON=/path/to/wx/bin/python
```

---

## 3. Download and Verify the Weights

Network checkpoints are **not** stored in this Git repository. Download them from Google Drive:

- **Weights folder:** [https://drive.google.com/drive/folders/16fElOj7ZKO8GkIW4ZtmDDfpUHBI10mL3?usp=sharing](https://drive.google.com/drive/folders/16fElOj7ZKO8GkIW4ZtmDDfpUHBI10mL3?usp=sharing)

Copy the `experiments/` directory from the download into the root of this repository so that the two checkpoints used by the paper sit at:

| Path in repository | Role | Size | Expected SHA-256 |
| :--- | :--- | :--- | :--- |
| `experiments/probe24/t20/model_A2_v2r.pt` | Corner locator network | 10,141,885 B | `2d1e1449352d5d52524e132b8374cfa66e68ee912b11f964ea3a9df763adf9c4` |
| `experiments/pathnet_fix/model_canrollw_long.pt` | 52-D canonical-roll path network | 25,016,575 B | `4e3f70f7188e27513f80dc28e333d5e07b5d4bcbb2c7f9770e74aea4a7409d7a` |

Verify the checksums locally:

```powershell
# PowerShell
Get-FileHash -Algorithm SHA256 experiments\probe24\t20\model_A2_v2r.pt, experiments\pathnet_fix\model_canrollw_long.pt
```

```bash
# Linux / macOS
sha256sum experiments/probe24/t20/model_A2_v2r.pt experiments/pathnet_fix/model_canrollw_long.pt
```

Every output `result.json` also records `versions.weight_sha256_match`, which is `true` when both checkpoints match the hashes above.

---

## 4. Check the Installation

Run the built-in synthetic decoder controls (no weights or dataset required):

```bash
python -m qrpipe --controls
```

- **Positive control:** clean synthetic QR codes (`positive=1.0` expected; exit code `0`).
- **Negative controls:** pure noise images and phase-scrambled QR grids (`nonempty` and `wrong` $\le 0.01$).
- Note: `pyzbar` may print harmless C-level assertion warnings on pure-noise images (e.g. `WARNING: ... pdf417.c ... Assertion ... failed`).

---

## 5. Download the Photograph Benchmark

The handheld motion-blurred QR benchmark (`QR_DEB_TRL_DATASET`, CC BY 4.0) is hosted on Google Drive:

- **Dataset folder:** [https://drive.google.com/drive/folders/1tKBJlFb34EyO09QVZq4hT7_R2pXC0vUD](https://drive.google.com/drive/folders/1tKBJlFb34EyO09QVZq4hT7_R2pXC0vUD)

Download and unzip the folder anywhere on disk (it does not need to be inside this repository). Keep its directory layout intact:

```text
QR_DEB_TRL_DATASET/
├── capture_metadata.csv
├── LICENSE
├── README.md
├── photos/
│   ├── session_01/ ... session_08/
│   │   ├── static/
│   │   ├── translation/
│   │   ├── zoom/
│   │   ├── rotation/
│   │   └── free/
│   │       ├── s01_translation_003_reference.jpg    # still capture of the same card
│   │       └── s01_translation_003_observation.jpg  # handheld motion-blurred capture
└── sensors/
```

The dataset contains 1,195 photograph pairs in total. The default evaluation benchmark consists of the **955 motion-blurred observation photographs** (`translation`: 241, `zoom`: 239, `rotation`: 239, `free`: 236) where `role == "observation"` and `included_in_eval == "true"` in `capture_metadata.csv`.

---

## 6. Run Inference

### Single photograph

```bash
python -m qrpipe "D:\data\QR_DEB_TRL_DATASET\photos\session_01\zoom\s01_zoom_001_observation.jpg" --out out/one
```

Omitting `--out` prints the JSON summary directly to standard output.

### Multiple photographs (batch)

```bash
python -m qrpipe photo_a.jpg photo_b.jpg photo_c.jpg --out out/batch
```

### Full 955-photograph benchmark (or filtered subsets)

```bash
# Full 955-photograph evaluation benchmark
python -m qrpipe --database "D:\data\QR_DEB_TRL_DATASET" --out out/eval

# Filter by session, motion category, or count (--session and --category are repeatable)
python -m qrpipe --database "D:\data\QR_DEB_TRL_DATASET" --out out/test \
    --session session_01 --category translation --limit 10
```

- `--category` accepts `translation`, `zoom`, `rotation`, or `free`.
- **Automatic resume:** any photograph that already has `<out>/<photo_stem>/result.json` is skipped on re-run, so interrupted evaluations resume cleanly with the exact same command.

---

## 7. Output Format

For each processed photograph, `qrpipe` writes `--out/<photo_stem>/result.json`. Its `decision` block contains:

| Field | Description |
| :--- | :--- |
| `status` | `decoded_direct` (read directly from the input photograph by $\ge 2$ decoders), `decoded` (read after trajectory estimation and deblurring), `rejected` (no payload reached 2-of-3 agreement), or `rejected_geometry_not_converged` (rejected with known-module error above `E2_ALARM`). |
| `payload` | Accepted payload string, or `""` when rejected. |
| `cons_how` | Acceptance route (`direct`, `a_arm`, `consensus`, `max_agree`, `none`, or the regularisation ladder step). |
| `chosen_round` | Alternating-fit round index (`0`..`3`) that produced the accepted image, or `-1` for direct readout. |

When `--database` is used, `qrpipe` additionally writes `--out/database_summary.csv` with columns:

```text
pair_id,status,decoded_payload,ground_truth_payload,payload_correct
```

and prints total photographs, decoded count, and exact ground-truth match count.

---

## Repository Structure

```text
.
├── qrpipe/                   # CLI entry point, frozen config (v1.2.0), pipeline, batch & database runners
├── scripts/
│   ├── pathnet_fix/          # 52-D canonical-roll path network (canon_net.py)
│   └── probe24/              # Corner locator, rigid-plane geometry, search, alternating fit, 3-decoder wrapper
├── repo/ACC_MODEL/           # QR code decoding helpers
├── experiments/              # Place downloaded model_A2_v2r.pt and model_canrollw_long.pt here (git-ignored)
├── requirements.txt          # Main environment dependencies
├── requirements_wechat.txt   # Isolated WeChat decoder environment dependencies
├── LICENSE                   # MIT License (code and weights)
└── THIRD_PARTY_NOTICES.md    # Runtime dependency licenses and notices
```

---

## Citation

If you use this code, the network weights, or the handheld QR motion benchmark, please cite our paper, dataset, and code release:

```bibtex
@misc{zeng2026code,
  author       = {Zeng, Zhi and Su, Qiaochen and Xiong, Si and Wang, Yaoqi and Cai, Yufei and Liu, Heng and Hu, Yi},
  title        = {qr-deblur: Inference Code for Decoding Motion-Blurred {QR} Codes},
  year         = {2026},
  howpublished = {Software (MIT License)},
  url          = {https://github.com/zengzhi2015/qr-deblur}
}

@misc{zeng2026data,
  author       = {Zeng, Zhi and Su, Qiaochen and Xiong, Si and Wang, Yaoqi and Cai, Yufei and Liu, Heng and Hu, Yi},
  title        = {Handheld {QR} Motion Photographs},
  year         = {2026},
  howpublished = {Dataset (CC BY 4.0)},
  url          = {https://drive.google.com/drive/folders/1tKBJlFb34EyO09QVZq4hT7_R2pXC0vUD}
}
```

---

## License

- **Code & network weights:** MIT License (`LICENSE`).
- **Photograph & sensor benchmark:** CC BY 4.0 (distributed with `QR_DEB_TRL_DATASET`).
- **Third-party runtime libraries:** see `THIRD_PARTY_NOTICES.md`.
