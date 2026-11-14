"""Render one camera of a MolmoAct v3 candidate densely over [start, end] at a stride, 400 px tiles, 12 per page.

    uv run python lerobot/examples/dataset/diverse_robot_dataset/molmoact_review/render_dense.py --component household --episode 7333 --camera wrist --start 8 --end 14 --stride 0.33

Sheets go to outputs/_annotation/diverse_v3_review/<component>/dense_claude/ (the review sheets and the build tree are never touched).
"""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'lerobot/examples/dataset/diverse_robot_dataset'))
from prepare_molmoact import Paths, contact_sheet, episode_rows, read_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--component', required=True, choices=('household', 'tabletop'))
parser.add_argument('--episode', type=int, required=True)
parser.add_argument('--camera', required=True, choices=('primary', 'secondary', 'wrist'))
parser.add_argument('--start', type=float, required=True)
parser.add_argument('--end', type=float, required=True)
parser.add_argument('--stride', type=float, default=0.5)
parser.add_argument('--columns', type=int, default=4)
parser.add_argument('--tile-width', type=int, default=400)
parser.add_argument('--per-page', type=int, default=12)
args = parser.parse_args()
paths = Paths(f'molmoact_{args.component}', ROOT / 'outputs/diverse_robot_dataset_v3_build')
output = ROOT / 'outputs/_annotation/diverse_v3_review' / args.component / 'dense_claude'
info = read_json(paths.metadata_root / 'meta/info.json')
row = next(r for r in episode_rows(paths.metadata_root) if int(r['episode_index']) == args.episode)
camera = f'observation.images.{args.camera}'
video = paths.source_root / info['video_path'].format(
    video_key=camera, chunk_index=int(row[f'videos/{camera}/chunk_index']), file_index=int(row[f'videos/{camera}/file_index']))
shard_start = float(row[f'videos/{camera}/from_timestamp'])
duration = float(row[f'videos/{camera}/to_timestamp']) - shard_start
end = min(args.end, duration - 0.05)
count = int(math.floor((end - args.start) / args.stride + 1e-9)) + 1
offsets = [round(args.start + i * args.stride, 3) for i in range(count)]
for page in range(0, len(offsets), args.per_page):
    chunk = offsets[page : page + args.per_page]
    name = f'episode_{args.episode:06d}.dense.{args.camera}.{chunk[0]:05.2f}-{chunk[-1]:05.2f}s.jpg'
    header = f'ep {args.episode}  {args.camera}  dense {chunk[0]:.2f}-{chunk[-1]:.2f}s stride {args.stride:g}s of {duration:.1f}s'
    print(contact_sheet(video, shard_start, chunk, output / name, header=header, tile_width=args.tile_width, columns=args.columns), flush=True)
