import sys
from pathlib import Path

# Add src to python path to ensure we can import lerobot modules
sys.path.append("lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import convert_image_to_video_dataset

# Paths
DATA_DIR = Path("outputs/train/2026-03-25/20-39-50_default/online_buffer")
OUTPUT_DIR = Path("outputs/train/2026-03-25/20-39-50_default/online_buffer_video")

def main():
    if not DATA_DIR.exists():
        print(f"Error: Dataset directory {DATA_DIR} does not exist.")
        return

    print(f"Loading dataset from {DATA_DIR}...")
    try:
        dataset = LeRobotDataset(root=DATA_DIR, repo_id="cijerezg/dummy_dataset")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    print(f"Converting dataset to video format at {OUTPUT_DIR}...")
    # Use h264 for better compatibility with standard tools
    try:
        convert_image_to_video_dataset(
            dataset=dataset,
            output_dir=OUTPUT_DIR,
            vcodec="h264", 
            repo_id="online_buffer_video",
            # Force overwrite if needed/possible, though the tool might raise if exists
        )
        print("Conversion complete!")
        print(f"New dataset is available at: {OUTPUT_DIR}")
        print("You can now run subtask_annotate.py with --dataset-dir", OUTPUT_DIR)
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
