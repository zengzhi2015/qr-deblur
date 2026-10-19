from __future__ import annotations
import json
from pathlib import Path
from .pipeline import run_photo, _js

def run_photos(photos, out_dir):
    """Run photos in order. A photo whose result.json already exists under out_dir is not run again."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, photo in enumerate(photos):
        photo = Path(photo)
        key = photo.stem
        done = out_dir / key / 'result.json'
        rec = None
        if done.is_file():
            try:
                rec = json.loads(done.read_text(encoding='utf-8'))
                print(f'batch {i + 1}/{len(photos)} {key} (already done)', flush=True)
            except ValueError:
                rec = None
        if rec is None:
            rec = run_photo(photo, save_dir=out_dir / key)
            print(f'batch {i + 1}/{len(photos)} {key}', flush=True)
        rows.append({'key': key, 'file': str(photo), 'result': rec})
    (out_dir / 'batch_rows.json').write_text(json.dumps(rows, ensure_ascii=False, default=_js, indent=2), encoding='utf-8')
    return rows
