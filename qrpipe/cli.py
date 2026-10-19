from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .config import VERSION, FROZEN_HASH
from .database import CATEGORIES

def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    p = argparse.ArgumentParser(description=f'QR deblur pipeline {VERSION}: corners -> path net -> length/direction search -> alternating fit -> dual inversion -> three decoders.')
    p.add_argument('photo', nargs='*', type=Path, help='Photo path(s); may be combined with --controls.')
    p.add_argument('--out', type=Path, default=None, help='Output directory (result.json and per-round paths).')
    p.add_argument('--controls', action='store_true', help='Run the decoder controls before processing the photo(s).')
    p.add_argument('--batch', action='store_true', help='Treat all photo arguments as a batch (writes under --out).')
    p.add_argument('--database', type=Path, default=None, help='Root folder of the downloaded photo database: run its default evaluation set (955 motion photographs) under --out. Finished photos are skipped on re-run.')
    p.add_argument('--category', action='append', choices=CATEGORIES, help='With --database: only this motion type (repeatable).')
    p.add_argument('--session', action='append', help='With --database: only this session, e.g. session_01 (repeatable).')
    p.add_argument('--limit', type=int, default=None, help='With --database: only the first N selected photographs.')
    args = p.parse_args(argv)
    print(f'qrpipe {VERSION} config_hash {FROZEN_HASH[:12]}', flush=True)
    if args.controls:
        from .controls import run_controls
        ctrl = run_controls()
        print(f"controls: positive={ctrl.get('positive')} negative={ctrl.get('negative')}", flush=True)
        if not args.photo:
            return 0 if ctrl.get('pos_ok') else 2
    if args.database is not None:
        if args.out is None:
            p.error('--out is required with --database.')
        from .database import select_photos, write_summary
        from .batch import run_photos
        selected = select_photos(args.database, args.category, args.session, args.limit)
        if not selected:
            p.error('No photographs selected.')
        run_photos([path for _, path, _ in selected], out_dir=args.out)
        write_summary(selected, args.out)
        return 0
    if not args.photo:
        p.error('Photo path required, --database, or --controls only.')
    if args.batch or len(args.photo) > 1:
        from .batch import run_photos
        if args.out is None:
            p.error('--out is required for batch.')
        run_photos(args.photo, out_dir=args.out)
        return 0
    from .pipeline import run_photo, _js
    rec = run_photo(args.photo[0], save_dir=args.out)
    d = rec['decision']
    print(f"{d['status']}={d['payload']!r}={d['cons_how']}round={d['chosen_round']}", flush=True)
    if args.out is None:
        print(json.dumps(rec, ensure_ascii=False, default=_js)[:2000], flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
