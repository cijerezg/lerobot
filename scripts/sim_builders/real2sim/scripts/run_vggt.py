"""frames dir -> vggt.npz (poses, intrinsics, depth, conf, images at 518 res). usage: run_vggt.py frames_dir out.npz"""
import sys, glob, numpy as np, torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
frames = sorted(glob.glob(sys.argv[1] + "/*.jpg")); out = sys.argv[2]
dev = "cuda"; dt = torch.bfloat16
model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
images = load_and_preprocess_images(frames).to(dev)
print("images", tuple(images.shape))
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dt):
    agg, ps_idx = model.aggregator(images[None])
    pose_enc = model.camera_head(agg)[-1]
    extr, intr = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    depth, depth_conf = model.depth_head(agg, images[None], ps_idx)
extr = extr[0].float().cpu().numpy(); intr = intr[0].float().cpu().numpy()
depth = depth[0].float().cpu().numpy(); depth_conf = depth_conf[0].float().cpu().numpy()
pts = unproject_depth_map_to_point_map(depth, extr, intr)
np.savez_compressed(out, frames=np.array(frames), extrinsics=extr, intrinsics=intr, depth=depth, depth_conf=depth_conf,
                    world_points=pts.astype(np.float32), images=(images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8))
print("depth", depth.shape, "conf median %.2f" % np.median(depth_conf), "saved", out)
