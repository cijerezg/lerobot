import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def visualize_chunks(plots_dir: str):
    """
    Recreates matplotlib plots from saved numpy arrays of action chunks and side camera frames.
    
    Args:
        plots_dir (str): Path to the base outputs/plots directory containing 'trajectories' and 'images' subdirs.
    """
    plots_path = Path(plots_dir)
    traj_dir = plots_path / "trajectories"
    img_dir = plots_path / "images"
    out_dir = plots_path / "recreated_figures"
    
    if not traj_dir.exists():
        print(f"Error: Trajectories directory not found at {traj_dir}")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all chunk files
    chunk_files = sorted(list(traj_dir.glob("chunk_*.npy")))
    if not chunk_files:
        print(f"No chunk_*.npy files found in {traj_dir}")
        return
        
    print(f"Found {len(chunk_files)} action chunks. Recreating plots in {out_dir}...")
    
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
    
    for chunk_path in tqdm(chunk_files):
        try:
            # Extract the index number from the filename (e.g., chunk_0042.npy -> 0042)
            idx_str = chunk_path.stem.split('_')[1]
            
            # Load actions
            acts_np = np.load(chunk_path)
            
            # Create figure
            # We recreate the 1x3 subplot structure. If top camera is missing, we leave it blank.
            fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [1, 1, 2]})
            
            # --- Panel 0: Top Camera (Placeholder since we didn't save it) ---
            axes[0].text(0.5, 0.5, "Top Camera\n(Not Saved)", horizontalalignment='center', verticalalignment='center')
            axes[0].axis('off')
            
            # --- Panel 1: Side Camera ---
            img_path = img_dir / f"side_{idx_str}.npy"
            if img_path.exists():
                img_side = np.load(img_path)
                axes[1].imshow(img_side)
                axes[1].set_title("Side Camera")
            else:
                axes[1].text(0.5, 0.5, "Side Camera\n(Missing)", horizontalalignment='center', verticalalignment='center')
            axes[1].axis('off')
            
            # --- Panel 2: Actions ---
            for i in range(min(6, acts_np.shape[1])):
                axes[2].plot(acts_np[:, i], label=joint_names[i])
            axes[2].set_title(f"Pi05 Generated Action Chunk (Inference pass {idx_str})")
            axes[2].set_xlabel("Step Index")
            axes[2].set_ylabel("Joint Position")
            axes[2].legend()
            axes[2].grid(True)
            
            plt.tight_layout()
            
            # Save figure
            out_path = out_dir / f"recreated_plot_{idx_str}.png"
            plt.savefig(out_path)
            plt.close(fig)
            
        except Exception as e:
            print(f"Error processing {chunk_path.name}: {e}")
            plt.close('all')

    print(f"Finished! Plots saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recreate action chunk plots from saved numpy arrays.")
    parser.add_argument(
        "--plots-dir", 
        type=str, 
        required=True, 
        help="Path to the outputs/plots directory (should contain 'trajectories' and 'images' folders)"
    )
    
    args = parser.parse_args()
    visualize_chunks(args.plots_dir)
