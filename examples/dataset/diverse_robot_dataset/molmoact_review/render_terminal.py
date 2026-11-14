"""Render twelve terminal samples per camera of MolmoAct v3 candidates, including the last source frame.

    uv run python lerobot/examples/dataset/diverse_robot_dataset/molmoact_review/render_terminal.py --component household --episodes 7333 3355 [--seconds 2] [--end-s 9.5]

Sheets go to outputs/_annotation/diverse_v3_review/<component>/terminal_claude/ (the review sheets and the build tree are never touched).
"""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lerobot/examples/dataset/diverse_robot_dataset'))
from prepare_molmoact import CAMERAS, Paths, contact_sheet, episode_rows, read_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--component', required=True, choices=('household', 'tabletop'))
parser.add_argument('--episodes', nargs='+', type=int, required=True)
parser.add_argument('--seconds', type=float, default=2.0)
parser.add_argument('--end-s', type=float, help='Optional end time for an event detail window')
args = parser.parse_args()
paths = Paths(f'molmoact_{args.component}', ROOT / 'outputs/diverse_robot_dataset_v3_build')
output = ROOT / 'outputs/_annotation/diverse_v3_review' / args.component / 'terminal_claude'
info = read_json(paths.metadata_root / 'meta/info.json')
rows = {int(r['episode_index']): r for r in episode_rows(paths.metadata_root)}
for index in args.episodes:
    row = rows[index]
    last = (int(row['length']) - 1) / float(info['fps'])
    end = last if args.end_s is None else min(last, args.end_s)
    offsets = np.linspace(max(0., end - args.seconds), end, 12).tolist()
    suffix = 'terminal' if args.end_s is None else f'detail_{end:g}'
    for camera in CAMERAS:
        label = camera.rsplit('.', 1)[-1]
        video = paths.source_root / info['video_path'].format(
            video_key=camera, chunk_index=int(row[f'videos/{camera}/chunk_index']),
            file_index=int(row[f'videos/{camera}/file_index']))
        print(contact_sheet(video, float(row[f'videos/{camera}/from_timestamp']), offsets,
                            output / f'episode_{index:06d}.{label}.{suffix}.jpg',
                            header=f'ep {index} {label} {suffix} through {end:.3f}s; last source time {last:.3f}s'), flush=True)
