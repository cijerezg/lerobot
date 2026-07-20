import argparse
import sys
from pathlib import Path

# Add src to python path to ensure we can import lerobot modules
sys.path.append("lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import convert_image_to_video_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert an online buffer dataset to video format.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/train/2026-04-16/20-09-41_default/dataset"),
        help="Path to the input dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/train/2026-04-16/20-09-41_default/dataset_video"),
        help="Path to write the converted video dataset.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Dataset directory {args.data_dir} does not exist.")
        return

    print(f"Loading dataset from {args.data_dir}...")
    try:
        dataset = LeRobotDataset(root=args.data_dir, repo_id="cijerezg/dummy_dataset")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Converting dataset to video format at {args.output_dir}...")
    try:
        convert_image_to_video_dataset(
            dataset=dataset,
            output_dir=args.output_dir,
            vcodec="h264",
            repo_id="online_buffer_video",
        )
        print("Conversion complete!")
        print(f"New dataset is available at: {args.output_dir}")
        print("You can now run subtask_annotate.py with --data-dir", args.output_dir)
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
