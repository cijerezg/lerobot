import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def visualize_episode(dataset_path, output_path, episode_idx=0, fps=30):
    dataset_path = Path(dataset_path)
    print(f"Loading dataset from {dataset_path}...")
    
    # Use LeRobotDataset to handle loading correctly
    # We use a dummy repo_id for local loading as per the pattern in subtask_annotate.py
    try:
        dataset = LeRobotDataset(repo_id="local/dataset", root=dataset_path, download_videos=False)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if episode_idx >= len(dataset.meta.episodes):
        print(f"Episode {episode_idx} not found (total {dataset.meta.total_episodes})")
        return

    print(f"Loaded dataset with {dataset.meta.total_episodes} episodes.")
    
    # Get episode range
    ep_meta = dataset.meta.episodes[episode_idx]
    start_idx = ep_meta["dataset_from_index"]
    end_idx = ep_meta["dataset_to_index"]
    length = end_idx - start_idx
    print(f"Episode {episode_idx}: Start={start_idx}, End={end_idx}, Length={length}")

    # Check for subtask_index feature
    if "subtask_index" not in dataset.features:
        print("Dataset does not contain 'subtask_index' feature!")
        return

    # Load subtask names manually since LeRobotDataset doesn't load custom metadata yet
    subtasks_path = dataset.root / "meta/subtasks.parquet"
    if not subtasks_path.exists():
        print(f"Dataset metadata does not contain subtasks at {subtasks_path}!")
        return
        
    subtasks_df = pd.read_parquet(subtasks_path)
    # The parquet has the skill name as the index (named 'subtask') and 'subtask_index' as a column.
    # We want to look up by 'subtask_index' to get the 'subtask' name.
    # Check if 'subtask_index' is a column or index
    if "subtask_index" in subtasks_df.columns:
        subtasks_df = subtasks_df.reset_index().set_index("subtask_index")
    
    print(f"Loaded {len(subtasks_df)} subtasks.")

    # Get video key - try to find one
    video_keys = [k for k in dataset.meta.video_keys]
    if not video_keys:
        print("No video keys found in dataset!")
        return
    
    # Prefer 'observation.images.side' or 'observation.images.base' if available
    video_key = next((k for k in video_keys if "side" in k), video_keys[0])
    if "base" in video_keys:
         video_key = next((k for k in video_keys if "base" in k), video_key)
         
    print(f"Using video key: {video_key}")

    # Get video path from metadata
    video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, video_key)
    if not video_path.exists():
        print(f"Video file not found: {video_path}")
        return
    
    # Get timestamps for video sync
    # LeRobotDataset handles frame access, but for visualization we might want to read the video directly 
    # to ensure we see exactly what's in the file, OR we can trust the dataset loader.
    # Using the dataset loader is safer for alignment, but might be slower if we iterate one by one.
    # However, let's use the dataset loader to be 100% sure about alignment with subtask_index.
    
    # Setup output writer
    # We need to get the first frame to know dimensions
    first_frame = dataset[start_idx][video_key]
    # Convert from torch (C, H, W) to numpy (H, W, C)
    if hasattr(first_frame, "permute"):
        first_frame = first_frame.permute(1, 2, 0).numpy()
    
    # Scale to 0-255 if needed (LeRobot usually returns float 0-1 or uint8)
    if first_frame.dtype == np.float32 or first_frame.dtype == np.float64:
        first_frame = (first_frame * 255).astype(np.uint8)
        
    height, width, _ = first_frame.shape
    print(f"Video dimensions: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Iterate through the episode
    print("Processing frames...")
    for i in range(length):
        dataset_idx = start_idx + i
        item = dataset[dataset_idx]
        
        # Get frame
        frame = item[video_key]
        if hasattr(frame, "permute"):
            frame = frame.permute(1, 2, 0).numpy()
        
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).astype(np.uint8)
            
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get subtask
        subtask_idx = item["subtask_index"].item()
        # Handle unannotated frames (-1)
        if subtask_idx == -1:
            subtask_name = "Unannotated"
        else:
            # Use iloc to get the row by position (since subtask_index is 0, 1, 2...)
            # And access the 'subtask' column for the name
            subtask_name = subtasks_df.iloc[subtask_idx]["subtask"]
        
        # Draw text
        text = f"Skill: {subtask_name}"
        cv2.putText(frame_bgr, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(frame_bgr)
        
        if i % 50 == 0:
            print(f"Processed {i}/{length} frames. Current skill: {subtask_name}")

    out.release()
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize annotated LeRobot dataset")
    parser.add_argument("--dataset", type=str, default="outputs/annotated_dataset-v1", help="Path to the dataset")
    parser.add_argument("--output", type=str, default="outputs/visualization_ep0-v1.mp4", help="Path to output video")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to visualize")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_episode(
        dataset_path=args.dataset,
        output_path=args.output,
        episode_idx=args.episode
    )

