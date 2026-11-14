#!/usr/bin/env bash
# One sim layout from one real episode's frame 0: scene (free bodies) -> cached frames + robot-free background -> material
# textures -> photo scene -> real-vs-sim check. usage: scripts/build_layout.sh EP OUT_DIR [PIXELS_JSON] [FRAMES_DIR] (from lerobot/scripts/sim_builders/real2sim)
set -euo pipefail
EP=$1; OUT=$2; PIX=${3:-work/layouts/pixels_r1016_ep${EP}_f0.json}; FRAMES=${4:-$OUT/frames}
ROOT=/home/user/Documents/Research/RL/LeRobot/outputs/rebot_2026-10-16-annotated-v1; REF=work/scene_nail; M=$(pwd)
export MUJOCO_GL=egl
mkdir -p "$OUT/photoreal"; cp -n $REF/wrist_camera.json "$OUT/" 2>/dev/null || true
[ -d "$FRAMES" ] || uv run python scripts/scene_frames.py $ROOT $EP "$FRAMES" --n 8
uv run python scripts/scene_from_episode.py $ROOT $EP 0 $REF/camera_pass2.json "$PIX" "$OUT"
[ -f "$OUT/frames_every.npz" ] || uv run python scripts/scene_masks.py $ROOT $EP "$OUT"
uv run python scripts/scene_background.py "$OUT/frames_every.npz" "$OUT/scene_nail.xml" "$OUT/photoreal/background_top_v3.png"
uv run python scripts/object_materials.py "$OUT/scene_nail.xml" "$OUT/photoreal/materials" --top_frame "$FRAMES/top_00000.jpg"
[ "$OUT" = "$REF" ] || cp $REF/rebot_b601_dm_scene_nail_photo.xml "$OUT/rebot_b601_dm_scene_nail_photo.xml"
uv run python scripts/photo_scene.py "$OUT/scene_nail.xml" "$OUT/photoreal/background_top_v3.png" "$OUT/scene_nail_photo.xml" --detail $REF/photoreal/detail_2220.npz --object_tex "$OUT/photoreal/materials" --gain 0.758 0.603 0.452 --robot_xml "$M/$OUT/rebot_b601_dm_scene_nail_photo.xml"
echo "layout ep$EP built in $OUT"
