#!/usr/bin/env python

import argparse
import logging
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_action_stats(root_dir: str, chunk_size: int = 50, encoding: str = "delta"):
    root = Path(root_dir)
    meta_dir = root / "meta"
    data_dir = root / "data"
    
    if not meta_dir.exists() or not data_dir.exists():
        logger.error(f"Invalid dataset structure in {root_dir}")
        return
        
    # 1. Load basic info
    with open(meta_dir / "info.json") as f:
        info = json.load(f)
    
    # 2. Load episode metadata (to know where episodes start/end)
    # We need this to ensure we don't compute displacements across episode boundaries
    episode_meta_files = sorted(list((meta_dir / "episodes").rglob("*.parquet")))
    if not episode_meta_files:
        logger.error("No episode metadata found.")
        return
        
    episodes_df = pd.concat([pd.read_parquet(f) for f in episode_meta_files])
    logger.info(f"Loaded metadata for {len(episodes_df)} episodes")

    # 3. Load actual data (Actions and States only)
    # This is the FAST part - only reading what we need
    data_files = sorted(list(data_dir.rglob("*.parquet")))
    logger.info(f"Loading {len(data_files)} data files...")
    
    # We read only the columns we need
    columns = [ACTION, OBS_STATE]
    data_df = pd.concat([pq.read_table(f, columns=columns).to_pandas() for f in data_files])
    
    # Convert to torch tensors for stats
    # Note: Action and state might be arrays in the parquet, pandas will hold them as object arrays of numpy
    actions = torch.from_numpy(np.stack(data_df[ACTION].values))
    states = torch.from_numpy(np.stack(data_df[OBS_STATE].values))
    
    logger.info(f"Loaded {len(actions)} frames. Computing stats for {encoding} encoding...")
    
    all_chunks = []
    
    # Iterate through episodes to avoid cross-episode chunks
    for _, ep in tqdm(episodes_df.iterrows(), total=len(episodes_df)):
        start_idx = int(ep["dataset_from_index"])
        end_idx = int(ep["dataset_to_index"])
        
        ep_actions = actions[start_idx:end_idx]
        ep_states = states[start_idx:end_idx]
        
        # For each frame in the episode, compute its encoded chunk relative to its start
        # A chunk starts at index 'i' and goes to 'i + chunk_size'
        num_valid_starts = len(ep_actions) - chunk_size + 1
        if num_valid_starts <= 0:
            continue
            
        for i in range(num_valid_starts):
            anchor = ep_states[i]
            targets = ep_actions[i : i + chunk_size]
            
            if encoding == "anchor":
                chunk_data = targets - anchor[None, :]
            elif encoding == "delta":
                d_0 = targets[0] - anchor
                if targets.shape[0] > 1:
                    d_rest = torch.diff(targets, dim=0)
                    chunk_data = torch.cat([d_0.unsqueeze(0), d_rest], dim=0)
                else:
                    chunk_data = d_0.unsqueeze(0)
            else: # absolute
                chunk_data = targets
                
            all_chunks.append(chunk_data)

    if not all_chunks:
        logger.error(f"No valid {encoding} samples found.")
        return

    # [N, T, D]
    all_chunks = torch.stack(all_chunks)
    
    stats = {}
    stats["min"] = all_chunks.min(dim=0).values
    stats["max"] = all_chunks.max(dim=0).values
    stats["mean"] = all_chunks.mean(dim=0)
    stats["std"] = all_chunks.std(dim=0)
    
    # Quantiles (p0.01, p99.99) - wider than p1/p99 to preserve rare but
    # important movements (e.g. gripper open/close on mostly-static joints)
    q01 = []
    q99 = []
    for t in range(chunk_size):
        data_t = all_chunks[:, t, :]
        q01.append(torch.quantile(data_t, 0.0001, dim=0))
        q99.append(torch.quantile(data_t, 0.9999, dim=0))
        
    stats["q01"] = torch.stack(q01)
    stats["q99"] = torch.stack(q99)
    
    return stats

def plot_stats(stats: dict, output_path: Path, repo_id: str, encoding: str):
    """Plot the min/max range of all 6 joints in a 3x2 grid."""
    t = np.arange(len(stats["min"]))
    num_joints = stats["min"].shape[1]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    axes = axes.flatten()
    
    for i in range(num_joints):
        ax = axes[i]
        ax.fill_between(t, 
                        stats["min"][:, i].numpy(), 
                        stats["max"][:, i].numpy(), 
                        alpha=0.2, color="gray", label="Min/Max Range")
        
        ax.fill_between(t, 
                        stats["q01"][:, i].numpy(), 
                        stats["q99"][:, i].numpy(), 
                        alpha=0.4, color="blue", label="0.01%-99.99% Range")
        
        ax.plot(t, stats["mean"][:, i].numpy(), label="Mean", color="black", linestyle="--")
        
        ax.set_title(f"Joint {i} ({encoding})")
        ax.set_xlabel("Step (T)")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    # Hide unused subplots if any (though we expect 6)
    for i in range(num_joints, len(axes)):
        axes[i].axis("off")
        
    fig.suptitle(f"{encoding.capitalize()} Action Progression - {repo_id}", fontsize=16)
    
    save_path = output_path / f"stats_plot_{encoding}_{repo_id.replace('/', '_')}.png"
    plt.savefig(save_path)
    logger.info(f"Plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to local dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/stats", help="Where to save stats")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action horizon")
    parser.add_argument("--encoding", type=str, required=True, choices=["absolute", "anchor", "delta"], help="Action encoding method")
    args = parser.parse_args()

    stats = compute_action_stats(args.root, args.chunk_size, args.encoding)
    if stats:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use basename of root for filename
        repo_id = Path(args.root).name
        save_file = output_path / f"action_stats_{args.encoding}_{repo_id.replace('/', '_')}.pt"
        torch.save(stats, save_file)
        logger.info(f"Stats saved to {save_file}")
        
        # Plot for all joints (3x2 grid)
        plot_stats(stats, output_path, repo_id, args.encoding)
        
        for k, v in stats.items():
            logger.info(
                f"Stat '{k}': shape {v.shape} | "
                f"range [{v.min():.3f}, {v.max():.3f}] | "
                f"mean {v.mean():.3f}"
            )

if __name__ == "__main__":
    main()
