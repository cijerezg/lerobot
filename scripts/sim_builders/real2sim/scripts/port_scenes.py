"""Copy the four nail layouts + clipper case assets from lerobot/scripts/sim_builders/real2sim into the
rebot_b601_sim package with paths relative to the scene file (assetdir = the scenes/ dir)."""
import re
import shutil
from pathlib import Path

WS = Path("/home/user/Documents/Research/RL/LeRobot")
MIG = WS / "lerobot/scripts/sim_builders/real2sim"
PKG = WS / "lerobot/src/lerobot/robots/rebot_b601_sim"
SCENES = PKG / "scenes"
LAYOUTS = {"nail_ep5": MIG / "work/scene_nail", "nail_ep1": MIG / "work/layouts/ep1", "nail_ep2": MIG / "work/layouts/ep2", "nail_ep3": MIG / "work/layouts/ep3"}
FOLLOWER_ASSETS = str(WS / "lerobot/src/lerobot/robots/rebot_b601_follower/mjcf/assets")

SCENES.mkdir(exist_ok=True)
# shared: clipper case assets, table texture, robot include
shutil.copytree(MIG / "assets/clipper_case", SCENES / "assets/clipper_case", dirs_exist_ok=True)
shutil.copy2(LAYOUTS["nail_ep5"] / "table_top.png", SCENES / "table_top.png")
robot = (LAYOUTS["nail_ep5"] / "rebot_b601_dm_scene_nail_photo.xml").read_text()
robot = robot.replace('<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<mujoco model="rebot_b601_dm">\n  \n',
                      '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<mujoco model="rebot_b601_dm">\n  <!-- rebot_b601_follower/mjcf/rebot_b601_dm.xml with the photo-fitted materials (lerobot/scripts/sim_builders/real2sim robot_materials.py)\n       and the calibrated wrist camera; mesh paths relative to the scene\'s assetdir (scenes/). -->\n')
n = 0
for stl in re.findall(r'file="([A-Za-z0-9_]+\.(?:STL|stl))"', robot):
    robot = robot.replace(f'file="{stl}"', f'file="../../rebot_b601_follower/mjcf/assets/{stl}"'); n += 1
assert not re.search(r'file="[A-Za-z0-9_]+\.(?:STL|stl)"', robot)
(SCENES / "rebot_b601_dm_photo.xml").write_text(robot)

for name, src in LAYOUTS.items():
    dst = SCENES / name
    (dst / "materials").mkdir(parents=True, exist_ok=True)
    for f in (src / "photoreal/materials").glob("*_photo.*"):
        shutil.copy2(f, dst / "materials" / f.name)
    shutil.copy2(src / "photoreal/materials/materials.json", dst / "materials/materials.json")
    shutil.copy2(src / "photoreal/background_top_v3.png", dst / "background_top.png")
    shutil.copy2(src / "objects.json", dst / "objects.json")
    xml = (src / "scene_nail_photo.xml").read_text()
    xml = xml.replace(f'{WS}/outputs/', 'outputs/')
    xml = xml.replace(f'meshdir="{FOLLOWER_ASSETS}"', 'assetdir=".."')
    xml = xml.replace(f'<include file="{src}/rebot_b601_dm_scene_nail_photo.xml"/>', '<include file="../rebot_b601_dm_photo.xml"/>')
    xml = xml.replace(f'file="{src}/photoreal/materials/', f'file="{name}/materials/')
    xml = xml.replace(f'file="{src}/table_top.png"', 'file="table_top.png"')
    xml = xml.replace(f'file="{MIG}/assets/clipper_case/', 'file="assets/clipper_case/')
    assert str(WS) not in xml, [l for l in xml.splitlines() if str(WS) in l][:3]
    (dst / "scene.xml").write_text(xml)
    print(name, sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6, "MB")
print("shared", sum(f.stat().st_size for f in (SCENES / "assets").rglob("*") if f.is_file()) / 1e6 + (SCENES / "table_top.png").stat().st_size / 1e6, "MB")
