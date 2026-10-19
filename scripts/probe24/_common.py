import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ACC = ROOT / 'repo' / 'ACC_MODEL'

def init_env():
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[v] = '1'
    s = str(ACC)
    if s not in sys.path:
        sys.path.insert(0, s)
