"""Inputs of the tip-offset adapter for every depth frame of a root: the base model's pooled image tokens of the wrist
and top view (frozen vision tower, the policy's own resize), and the 3D point of each of the 27 x 27 wrist patches
from the depth frame, in the tool frame. Run build.py first. See README.md."""
import argparse, glob, json, os
import av, cv2, numpy as np, pandas as pd, torch
import torchvision.transforms.functional as TF
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from lerobot.annotation.corrections.arm import Arm, frame

BASE = "outputs/models/MolmoAct2"
FX, FY, CX, CY = 394.9832, 394.9832, 322.5604, 238.6966  # D405 at 640 x 480
GRID = 27  # the tower's patch grid on the resized 378 x 378 image
BATCH = 64


def base_tensors(prefix):
    """The base checkpoint's tensors whose name starts with prefix, prefix removed."""
    index = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
    out = {}
    for shard in sorted({f for k, f in index.items() if k.startswith(prefix)}):
        with safe_open(f"{BASE}/{shard}", "pt") as f:
            out.update({k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)})
    return out


def subtask_embeddings(texts):
    """{text: the base model's input embeddings of its tokens, [tokens, width]}."""
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    table = base_tensors("model.transformer.wte.")["embedding"]
    return {t: table[tokenizer(t, add_special_tokens=False).input_ids].float() for t in texts}


class Tower:
    """The base vision tower up to its 2 x 2 pooling: uint8 images [B, H, W, 3] -> [B, 196, 1152]."""

    def __init__(self):
        config = AutoConfig.from_pretrained(BASE, trust_remote_code=True)
        cls = get_class_from_dynamic_module("modeling_molmoact2.MolmoAct2VisionBackbone", BASE)
        self.net = cls(config.vit_config, config.adapter_config)
        missing, unexpected = self.net.load_state_dict(base_tensors("model.vision_backbone."), strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        self.net = self.net.cuda().eval()
        self.size = list(config.vit_config.image_default_input_size)
        self.patch = config.vit_config.image_patch_size
        idx = torch.full((GRID + 1, GRID + 1), -1)
        idx[:GRID, :GRID] = torch.arange(GRID * GRID).view(GRID, GRID)  # padded at the bottom and right, like arange_for_pooling
        self.idx = idx.view(14, 2, 14, 2).permute(0, 2, 1, 3).reshape(196, 4).cuda()

    @torch.no_grad()
    def __call__(self, images):
        net, p = self.net, self.patch
        x = TF.resize(torch.from_numpy(images).cuda().permute(0, 3, 1, 2), self.size, TF.InterpolationMode.BILINEAR, antialias=False)
        b, c, h, w = x.shape
        x = x.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p * p * c)
        x = (x.float() / 255 * 2 - 1).to(net.dtype)
        features = net.encode_image(x[:, None])[:, 0]  # [B, 729, 2304]
        valid = self.idx >= 0
        to_pool = features[:, self.idx.clip(0)] * valid[None, :, :, None]  # [B, 196, 4, 2304]
        to_pool = to_pool.reshape(b * 196, 4, -1)
        query = to_pool.sum(1, keepdim=True) / valid.sum(1).repeat(b)[:, None, None]
        assert net.adapter_config.pooling_attention_mask
        pooled = net.image_pooling_2d(query, to_pool, attn_mask=valid.repeat(b, 1)[:, None, None, :])
        return pooled.reshape(b, 196, -1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    args = ap.parse_args()
    root = args.root.rstrip("/")
    name = os.path.basename(root)
    out = f"outputs/cache/tip_offset/{name}"
    os.makedirs(out, exist_ok=True)

    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(root + "/data/*/*.parquet"))]).sort_values("index").reset_index(drop=True)
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(root + "/meta/episodes/*/*.parquet"))]).set_index("episode_index")
    targets = pd.read_parquet(f"lerobot/annotation_data/tip_offset/{name}/targets.parquet")
    has_depth = np.array([os.path.exists(f"{root}/depth/wrist.depth/episode-{e:06d}/frame-{i:06d}.png") for e, i in zip(df.episode_index, df.frame_index)])
    rows = np.flatnonzero(has_depth)  # depth is recorded every 3rd frame
    n = len(rows)
    np.save(out + "/rows.npy", rows)
    wrist_out = np.lib.format.open_memmap(out + "/wrist.npy", "w+", np.float16, (n, 196, 1152))
    top_out = np.lib.format.open_memmap(out + "/top.npy", "w+", np.float16, (n, 196, 1152))
    points_out = np.zeros((n, GRID, GRID, 4), np.float32)  # x, y, z in the tool frame (m), share of valid depth pixels
    surface_out = np.full((n, 3), np.nan, np.float32)  # the depth point at the target pixel, tool frame (m)

    def video(key, e):
        """Frames of one episode's video, in order (rgb)."""
        r = eps.loc[e]
        c = av.open(f"{root}/videos/{key}/chunk-{r[f'videos/{key}/chunk_index']:03d}/file-{r[f'videos/{key}/file_index']:03d}.mp4")
        st = c.streams.video[0]
        t0 = r[f"videos/{key}/from_timestamp"]
        c.seek(int(max(t0 - 1, 0) / st.time_base), stream=st)
        for fr in c.decode(st):
            if float(fr.pts * st.time_base) >= t0 - 1e-3:
                yield fr.to_ndarray(format="rgb24")

    tower, arm = Tower(), Arm()
    vv, uu = np.mgrid[0:480, 0:640]
    rays = torch.tensor(np.stack([(uu - CX) / FX, (vv - CY) / FY, np.ones_like(uu, dtype=float)], -1), dtype=torch.float32).cuda()
    cell = torch.tensor((vv * GRID // 480) * GRID + uu * GRID // 640).cuda().reshape(-1)  # the wrist patch of each depth pixel
    row_of = {r: k for k, r in enumerate(rows)}
    batch = []

    def flush():
        if batch:
            ks = [k for k, _, _ in batch]
            wrist_out[ks] = tower(np.stack([w for _, w, _ in batch])).float().cpu().numpy()
            top_out[ks] = tower(np.stack([t for _, _, t in batch])).float().cpu().numpy()
            batch.clear()

    for e in sorted(df.episode_index.unique()):
        g = df[df.episode_index == e]
        wrist, top = video("observation.images.wrist", e), video("observation.images.top", e)
        for i, q in zip(g.index, np.stack(g["observation.state"])):
            rgb, top_rgb = next(wrist), next(top)
            if i not in row_of:
                continue
            k = row_of[i]
            tip, a, j = arm.gripper.fk(q)
            cam_p, cam_r = arm.gripper.wrist(q)
            to_tool = frame(a, j).T  # tool = to_tool @ (cam_r @ camera + cam_p - tip)
            depth = torch.from_numpy(cv2.imread(f"{root}/depth/wrist.depth/episode-{e:06d}/frame-{g.frame_index[i]:06d}.png", cv2.IMREAD_UNCHANGED).astype(np.float32)).cuda() * 1e-4
            ok = (depth > 0).reshape(-1)
            p = (rays * depth[..., None]).reshape(-1, 3) @ torch.tensor((to_tool @ cam_r).T, dtype=torch.float32).cuda() + torch.tensor(to_tool @ (cam_p - tip), dtype=torch.float32).cuda()
            count = torch.bincount(cell[ok], minlength=GRID * GRID).float()
            mean = torch.stack([torch.bincount(cell[ok], p[ok, c], GRID * GRID) for c in range(3)], 1) / count.clip(1)[:, None]
            points_out[k] = torch.cat([mean, (count / torch.bincount(cell, minlength=GRID * GRID))[:, None]], 1).view(GRID, GRID, 4).cpu().numpy()
            t = targets.iloc[i]
            if t.in_view:  # the depth point at the target pixel: valid pixels within 2 pixels of it
                u, v = int(t.u), int(t.v)
                window = torch.zeros(480, 640, dtype=torch.bool, device="cuda")
                window[max(v - 2, 0) : v + 3, max(u - 2, 0) : u + 3] = True
                window = window.reshape(-1) & ok
                if window.any():
                    surface_out[k] = p[window].mean(0).cpu().numpy()
            batch.append((k, rgb, top_rgb))
            if len(batch) == BATCH:
                flush()
        flush()
        print(f"episode {e}: {len(g)} frames", flush=True)
    wrist_out.flush(); top_out.flush()
    np.save(out + "/points.npy", points_out)
    np.save(out + "/surface.npy", surface_out)
    print("wrote", out, n, "depth frames")
