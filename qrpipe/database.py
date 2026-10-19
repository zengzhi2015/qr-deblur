from __future__ import annotations
import csv
import json
from pathlib import Path

CATEGORIES = ('translation', 'zoom', 'rotation', 'free')

def _rows(db_root):
    csv_path = Path(db_root) / 'capture_metadata.csv'
    if not csv_path.is_file():
        raise SystemExit(f'capture_metadata.csv not found in {db_root}')
    with csv_path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))

def select_photos(db_root, categories=None, sessions=None, limit=None):
    """Motion photographs of the default evaluation set, as (pair_id, path, ground-truth payload)."""
    root = Path(db_root)
    out = []
    for r in _rows(root):
        if r['role'] != 'observation' or r['included_in_eval'].strip().lower() != 'true':
            continue
        if categories and r['category'] not in categories:
            continue
        if sessions and r['session'] not in sessions:
            continue
        out.append((r['pair_id'], root / r['relative_path'], r['payload']))
    if limit:
        out = out[:limit]
    return out

def write_summary(selected, out_dir):
    """Write database_summary.csv and print the counts."""
    out_dir = Path(out_dir)
    lines = ['pair_id,status,decoded_payload,ground_truth_payload,payload_correct']
    n_dec = n_ok = 0
    for pair_id, path, truth in selected:
        rec = json.loads((out_dir / path.stem / 'result.json').read_text(encoding='utf-8'))
        d = rec['decision']
        decoded = d['status'].startswith('decoded')
        ok = decoded and d['payload'] == truth
        n_dec += decoded
        n_ok += ok
        lines.append(','.join([pair_id, d['status'], '"' + str(d['payload']).replace('"', '""') + '"',
                               '"' + truth.replace('"', '""') + '"', str(bool(ok)).lower()]))
    (out_dir / 'database_summary.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    n = len(selected)
    print(f'database: photos={n} decoded={n_dec} payload_correct={n_ok} '
          f'(summary: {out_dir / "database_summary.csv"})', flush=True)
    return n, n_dec, n_ok
