"""Train the tip-offset adapter on the cached inputs (features.py) of one or more roots, in episode groups: each
group is predicted by a model that did not train on it. Writes the held-out predictions and the numbers. See README.md."""
import argparse, json, math, os
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from features import subtask_embeddings
from model import GRID, Adapter

ap = argparse.ArgumentParser()
ap.add_argument("roots", nargs="+")
ap.add_argument("--out", default="outputs/analysis/tip_offset/adapter_v1")
ap.add_argument("--groups", type=int, default=5)
ap.add_argument("--only", type=int, default=None, help="train this group only (a quick look)")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--weight_decay", type=float, default=0.05)
ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--sigma", type=float, default=0.75, help="width of the label around the target pixel, in wrist patches")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(0)

# data, all on the GPU
frames, wrist, top, points, surface = [], [], [], [], []
for k, root in enumerate(args.roots):
    name = os.path.basename(root.rstrip("/"))
    cache = f"outputs/cache/tip_offset/{name}"
    t = pd.read_parquet(f"lerobot/annotation_data/tip_offset/{name}/targets.parquet").iloc[np.load(cache + "/rows.npy")]
    frames.append(t.assign(root=name, episode=k * 1000 + t.episode_index))
    wrist.append(torch.from_numpy(np.load(cache + "/wrist.npy")).cuda())
    top.append(torch.from_numpy(np.load(cache + "/top.npy")).cuda())
    points.append(torch.from_numpy(np.load(cache + "/points.npy")).cuda())
    surface.append(np.load(cache + "/surface.npy"))
frames = pd.concat(frames).reset_index(drop=True)
wrist, top, points, surface = torch.cat(wrist), torch.cat(top), torch.cat(points), np.concatenate(surface)
n = len(frames)
has_target = torch.tensor(((frames.kind != "none") & frames.in_view).values).cuda()

# subtask text: the base model's token embeddings, padded to the longest
texts = sorted(frames.subtask.unique())
emb = subtask_embeddings(texts)
length = max(len(e) for e in emb.values())
text = torch.zeros(len(texts), length, emb[texts[0]].shape[1])
text_mask = torch.zeros(len(texts), length, dtype=torch.bool)
for k, s in enumerate(texts):
    text[k, : len(emb[s])], text_mask[k, : len(emb[s])] = emb[s], True
text, text_mask = text.cuda(), text_mask.cuda()
text_id = torch.tensor(frames.subtask.map(texts.index).values).cuda()

# label: a bump around the target pixel over the 27 x 27 wrist patches, or everything on the last slot ("no target")
gu, gv = torch.tensor(frames.u.values * GRID / 640).cuda(), torch.tensor(frames.v.values * GRID / 480).cuda()
centre = torch.arange(GRID).cuda() + 0.5
bump = torch.exp(-((centre[None, :, None] - gv[:, None, None]) ** 2 + (centre[None, None, :] - gu[:, None, None]) ** 2) / (2 * args.sigma**2)).reshape(n, -1)
label = torch.zeros(n, GRID * GRID + 1).cuda()
label[has_target, :-1] = (bump / bump.sum(1, keepdim=True))[has_target].float()
label[~has_target, -1] = 1

# episode groups: the episodes in a fixed random order, dealt out in turn
episodes = np.random.default_rng(0).permutation(frames.episode.unique())
group = frames.episode.map({e: k % args.groups for k, e in enumerate(episodes)}).values


def batch_inputs(i, scale):
    depth = points[i].clone()
    depth[..., :3] = depth[..., :3] / scale * (depth[..., 3:] > 0)
    return wrist[i].float(), top[i].float(), depth, text[text_id[i]], text_mask[text_id[i]]


def vector(weights, i):
    """The weighted mean of the patches' depth points [B, 3]; patches without a valid depth pixel are left out."""
    w = weights * (points[i][..., 3].reshape(len(i), -1) > 0)
    return ((w / w.sum(1, keepdim=True).clip(1e-9))[..., None] * points[i][..., :3].reshape(len(i), -1, 3)).sum(1)


pred = pd.DataFrame(index=frames.index, columns=["group", "loss", "p_none", "peak", "adapter_x", "adapter_y", "adapter_z", "peak_x", "peak_y", "peak_z", "label_x", "label_y", "label_z"], dtype=float)
log = {}
for g in range(args.groups) if args.only is None else [args.only]:
    train, held = torch.tensor(np.flatnonzero(group != g)).cuda(), torch.tensor(np.flatnonzero(group == g)).cuda()
    valid = points[train][..., 3] > 0
    scale = points[train][..., :3][valid].norm(dim=-1).median()  # the depth points enter in units of their median distance
    model = Adapter(text.shape[-1], length, dropout=args.dropout).cuda()
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.weight_decay), dict(params=rest, weight_decay=0.0)], lr=args.lr, betas=(0.9, 0.95))
    steps = args.epochs * (len(train) // args.batch)
    warmup = steps // 20
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(s / steps, 1)))))
    print(f"group {g}: {len(train)} train frames, {len(held)} held out, {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, {steps} steps", flush=True)

    def evaluate(index):
        model.eval()
        rows = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for i in index.split(256):
                scores = model(*batch_inputs(i, scale)).float()
                p = scores.softmax(-1)
                loss = -(label[i] * scores.log_softmax(-1)).sum(-1)
                peak = p[:, :-1].argmax(-1)
                rows.append(torch.cat([loss[:, None], p[:, -1:], peak[:, None], vector(p[:, :-1], i), vector(F.one_hot(peak, GRID * GRID).float(), i), vector(label[i, :-1], i)], 1))
        model.train()
        return torch.cat(rows).cpu().numpy()

    for epoch in range(args.epochs):
        total = 0.0
        for i in train[torch.randperm(len(train), device="cuda")].split(args.batch)[:-1]:
            with torch.autocast("cuda", torch.bfloat16):
                scores = model(*batch_inputs(i, scale))
            loss = -(label[i] * scores.float().log_softmax(-1)).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        held_loss = evaluate(held)[:, 0].mean()
        log.setdefault(g, []).append((total / (len(train) // args.batch), float(held_loss)))
        print(f"  epoch {epoch + 1}: train loss {log[g][-1][0]:.3f}, held-out loss {held_loss:.3f}", flush=True)
    pred.loc[held.cpu().numpy(), pred.columns[1:]] = evaluate(held)
    pred.loc[held.cpu().numpy(), "group"] = g
    torch.save(dict(model=model.state_dict(), scale=float(scale), texts=texts, args=vars(args), held_out=sorted(int(e) for e in episodes[np.arange(len(episodes)) % args.groups == g])), f"{args.out}/group_{g}.pt")
    del model, opt

# the numbers, on held-out frames only
done = pred.group.notna().values
frames = pd.concat([frames, pred], axis=1)
frames["surface_x"], frames["surface_y"], frames["surface_z"] = surface.T
frames.to_parquet(args.out + "/held_out.parquet")
f = frames[done]
truth = ((f.kind != "none") & f.in_view).values
said = (f.p_none < 0.5).values
both = f[truth & said]
tr, tc = np.floor(both.v * GRID / 480), np.floor(both.u * GRID / 640)
off = np.maximum(np.abs(both.peak // GRID - tr), np.abs(both.peak % GRID - tc))
xyz = lambda a: both[[a + c for c in "xyz"]].values.astype(float)
q = lambda a, b: " / ".join(f"{v:.1f}" for v in np.nanpercentile(100 * np.linalg.norm(a - b, axis=1), [50, 90]))
line = lambda name, a: f"  {name:28s}  {q(a, xyz('surface_')):>20s}    {q(a, xyz('')):>20s}"
out = [
    f"held-out frames {len(f)}: {truth.sum()} with a target in the wrist view, {(~truth).sum()} without",
    f"target frames the adapter calls target: {(truth & said).sum() / truth.sum():.3f}",
    f"other frames the adapter calls no target: {(~truth & ~said).sum() / (~truth).sum():.3f}",
    f"peak on the target's patch: {(off == 0).mean():.3f}, within one patch: {(off <= 1).mean():.3f}, within two: {(off <= 2).mean():.3f}",
    "vector error cm, p50 / p90      to the depth point at the target pixel    to the tip at the event (FK)",
    line("adapter, all patches", xyz("adapter_")),
    line("adapter, peak patch only", xyz("peak_")),
    line("the label's own weights", xyz("label_")),
    line("no adapter (zero vector)", 0 * xyz("")),
    "last epoch, train loss / held-out loss per group: " + ", ".join(f"{v[-1][0]:.2f} / {v[-1][1]:.2f}" for v in log.values()),
]
print("\n".join(out))
json.dump(dict(args=vars(args), log=log, summary=out), open(args.out + "/numbers.json", "w"), indent=1)
print("wrote", args.out)
