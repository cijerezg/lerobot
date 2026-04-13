#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Automatic Skill Annotation for LeRobot Datasets.

This script performs automatic subtask/skill labeling for ANY LeRobot dataset using
Vision-Language Models (VLMs). It segments each robot demonstration into short atomic
skills (1-3 seconds each) and creates a new dataset with subtask annotations.

Supported VLMs:
- Gemma 4 (default): "google/gemma-4-31B-it"
- Qwen2-VL/Qwen3-VL

Usage:
python examples/dataset/annotate.py \
    --repo-id your-username/your-dataset \
    --video-key observation.images.base \
    --model google/gemma-4-31B-it \
    --output-dir /path/to/output \
    --push-to-hub
"""

import argparse
import json
import re
import subprocess
import tempfile
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from rich.console import Console

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Skill Annotation Data Structures

class Skill:
    """Represents a single atomic skill/subtask in a demonstration."""

    def __init__(self, name: str, start: float, end: float):
        self.name = name
        self.start = start
        self.end = end

    def to_dict(self) -> dict:
        return {"name": self.name, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict) -> "Skill":
        return cls(name=data["name"], start=data["start"], end=data["end"])

    def __repr__(self) -> str:
        return f"Skill(name='{self.name}', start={self.start:.2f}, end={self.end:.2f})"


class EpisodeSkills:
    """Container for all skills in an episode."""

    def __init__(self, episode_index: int, description: str, skills: list[Skill]):
        self.episode_index = episode_index
        self.description = description
        self.skills = skills

    def to_dict(self) -> dict:
        return {
            "episode_index": self.episode_index,
            "description": self.description,
            "skills": [s.to_dict() for s in self.skills],
        }


# VLM Interface

class BaseVLM(ABC):
    @abstractmethod
    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        pass

    @abstractmethod
    def segment_skills(
        self, video_path: Path, episode_duration: float, coarse_goal: str | None = None
    ) -> list[Skill]:
        pass

    @abstractmethod
    def segment_skills_batch(
        self, video_paths: list[Path], episode_durations: list[float], coarse_goal: str | None = None
    ) -> list[list[Skill]]:
        pass


def create_skill_segmentation_prompt(coarse_goal: str | None = None) -> str:
    goal_context = f'The overall goal of this demonstration is: "{coarse_goal}"\n\n' if coarse_goal else ""

    return textwrap.dedent(f"""\
        # Role
        You are an advanced Robotics Vision System specializing in precise temporal action segmentation for robot manipulation demonstrations.

        # Task
        {goal_context}Analyze the provided robot demonstration video and segment it into a continuous sequence of short, atomic manipulation skills based strictly on visual evidence.

        # Strict Boundary Requirements
        1. **Continuous Timeline**: There must be NO temporal gaps. The `end` timestamp of Skill N must be the exact `start` timestamp of Skill N+1.
        2. **Full Coverage**: The sequence MUST start at exactly 0.0 and end exactly at the total duration of the video.
        3. **Inter-frame Gaps**: If an action completes in the unseen gap between two frames, assume it took the maximum possible time. Set its end timestamp to immediately before the next frame begins.
        4. **Timestamps**: Use standard floats for all seconds.

        # Output Format
        Output ONLY a valid JSON object. Do not include markdown code blocks or explanations. Use the exact schema below:

        {{
          "skills": [
            {{"name": "<action_string_1>", "start": 0.0, "end": <float_t1>}},
            {{"name": "<action_string_2>", "start": <float_t1>, "end": <float_t2>}}
          ]
        }}
    """)


# Gemma 4 Implementation

class Gemma4VL(BaseVLM):
    """Gemma 4 model for skill segmentation."""

    def __init__(self, model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.console = Console()
        self.device = device
        self.model_name = model_name

        self.console.print(f"[cyan]Loading Gemma 4 VLM: {model_name}...[/cyan]")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device
        )

        self.console.print(f"[green]✓ Model loaded successfully on {device}[/green]")

    def _sample_video_frames(self, video_path: Path, target_fps: float = 1.0) -> list[Image.Image]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0 or np.isnan(original_fps):
            original_fps = 30.0

        frame_interval = max(1, int(round(original_fps / target_fps)))
        frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                
            frame_idx += 1
            
        cap.release()
        return frames

    def segment_skills(
        self, video_path: Path, episode_duration: float, coarse_goal: str | None = None
    ) -> list[Skill]:
        prompt = create_skill_segmentation_prompt(coarse_goal)
        duration_str = f"{int(episode_duration // 60):02d}:{int(episode_duration % 60):02d}"

        frames = self._sample_video_frames(video_path, target_fps=4.0)

        messages = [
            {"role": "user", "content": [
                {"type": "video"},
                {"type": "text", "text": prompt + f"\n\nVideo duration: {duration_str} (~{episode_duration:.1f}s). Segment into atomic skills."}
            ]}
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=text,
            videos=frames,
            num_frames=len(frames),
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1500, 
                do_sample=False, 
                temperature=0.0
            )

        response = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0].strip()

        return self._parse_skills_response(response)

    def segment_skills_batch(
        self, video_paths: list[Path], episode_durations: list[float], coarse_goal: str | None = None
    ) -> list[list[Skill]]:
        prompt = create_skill_segmentation_prompt(coarse_goal)
        
        all_messages = []
        all_videos = []
        
        for video_path, duration in zip(video_paths, episode_durations):
            duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
            frames = self._sample_video_frames(video_path, target_fps=4.0)
            all_videos.append(frames)
            
            messages = [
                {"role": "user", "content": [
                    {"type": "video"},
                    {"type": "text", "text": prompt + f"\n\nVideo duration: {duration_str} (~{duration:.1f}s). Segment into atomic skills."}
                ]}
            ]
            all_messages.append(messages)

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in all_messages
        ]

        min_frames = min(len(f) for f in all_videos)
        inputs = self.processor(
            text=texts,
            videos=all_videos,
            num_frames=min_frames,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=1500, 
                do_sample=False, 
                temperature=0.0
            )

        responses = self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

        all_skills = []
        for idx, response in enumerate(responses):
            try:
                skills = self._parse_skills_response(response.strip())
                all_skills.append(skills)
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to parse response for video {idx}: {e}[/yellow]")
                all_skills.append([])
        
        return all_skills

    def _parse_skills_response(self, response: str) -> list[Skill]:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        try:
            data = json.loads(response)
            skills_data = data.get("skills", data)
            if isinstance(skills_data, list):
                return [Skill.from_dict(s) for s in skills_data]
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                skills_data = data.get("skills", [])
                return [Skill.from_dict(s) for s in skills_data]

        raise ValueError(f"Could not parse skills from response: {response[:200]}...")


# VLM Registry

VLM_REGISTRY: dict[str, type[BaseVLM]] = {
    "google/gemma-4-27B-it": Gemma4VL,
    "google/gemma-4-31B-it": Gemma4VL,
    "google/gemma-4-4B-it": Gemma4VL,
}

def get_vlm(model_name: str, device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> BaseVLM:
    if model_name in VLM_REGISTRY:
        return VLM_REGISTRY[model_name](model_name, device, torch_dtype)

    model_lower = model_name.lower()
    if "gemma-4" in model_lower:
        return Gemma4VL(model_name, device, torch_dtype)

    raise ValueError(f"Unknown model: {model_name}.")


# Video Extraction Utilities

class VideoExtractor:
    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def extract_episode_video(
        self, video_path: Path, start_timestamp: float, end_timestamp: float, target_fps: int = 1,
    ) -> Path:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.close()

        duration = end_timestamp - start_timestamp
        cmd = [
            "ffmpeg", "-i", str(video_path), "-ss", str(start_timestamp), "-t", str(duration),
            "-r", str(target_fps), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-an", "-y", str(tmp_path),
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e}") from e

        return tmp_path


# Skill Annotation Pipeline

class SkillAnnotator:
    def __init__(self, vlm: BaseVLM, video_extractor: VideoExtractor | None = None, console: Console | None = None, batch_size: int = 8):
        self.vlm = vlm
        self.console = console or Console()
        self.video_extractor = video_extractor or VideoExtractor(self.console)
        self.batch_size = batch_size

    def annotate_dataset(self, dataset: LeRobotDataset, video_key: str, episodes: list[int] | None = None, skip_existing: bool = False) -> dict[int, EpisodeSkills]:
        episode_indices = episodes or list(range(dataset.meta.total_episodes))
        annotations: dict[int, EpisodeSkills] = {}
        coarse_goal = self._get_coarse_goal(dataset)

        for batch_start in range(0, len(episode_indices), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(episode_indices))
            batch_episodes = episode_indices[batch_start:batch_end]
            
            try:
                batch_annotations = self._annotate_episodes_batch(dataset, batch_episodes, video_key, coarse_goal)
                for ep_idx in batch_episodes:
                    if ep_idx in batch_annotations and batch_annotations[ep_idx]:
                        annotations[ep_idx] = EpisodeSkills(
                            episode_index=ep_idx, description=coarse_goal, skills=batch_annotations[ep_idx]
                        )
                        self.console.print(f"[green]✓ Episode {ep_idx}: {len(batch_annotations[ep_idx])} skills identified[/green]")
            except Exception as e:
                self.console.print(f"[red]✗ Batch failed: {e}[/red]")

        return annotations

    def _get_coarse_goal(self, dataset: LeRobotDataset) -> str:
        if dataset.meta.tasks is not None and len(dataset.meta.tasks) > 0:
            first_task = dataset.meta.tasks.index[0]
            if first_task:
                return str(first_task)
        return "Perform the demonstrated manipulation task."

    def _annotate_episodes_batch(self, dataset: LeRobotDataset, episode_indices: list[int], video_key: str, coarse_goal: str) -> dict[int, list[Skill]]:
        extracted_paths = []
        durations = []
        valid_episode_indices = []
        
        for ep_idx in episode_indices:
            try:
                video_path = dataset.root / dataset.meta.get_video_file_path(ep_idx, video_key)
                ep = dataset.meta.episodes[ep_idx]
                start_ts = float(ep[f"videos/{video_key}/from_timestamp"])
                end_ts = float(ep[f"videos/{video_key}/to_timestamp"])
                duration = end_ts - start_ts
                
                extracted_path = self.video_extractor.extract_episode_video(video_path, start_ts, end_ts, target_fps=1)
                extracted_paths.append(extracted_path)
                durations.append(duration)
                valid_episode_indices.append(ep_idx)
            except Exception:
                continue
        
        if not extracted_paths: return {}
        
        try:
            all_skills = self.vlm.segment_skills_batch(extracted_paths, durations, coarse_goal)
            return {ep_idx: skills for ep_idx, skills in zip(valid_episode_indices, all_skills)}
        finally:
            for path in extracted_paths:
                if path.exists(): path.unlink()


# Metadata Writer
def get_skill_for_timestamp(skills: list[Skill], timestamp: float) -> Skill | None:
    for skill in skills:
        if skill.start <= timestamp < skill.end:
            return skill
        if timestamp >= skill.end and skill == skills[-1]:
            return skill
    return skills[-1] if skills else None


def save_skill_annotations(dataset: LeRobotDataset, annotations: dict[int, EpisodeSkills], output_dir: Path | None = None, repo_id: str | None = None) -> LeRobotDataset:
    console = Console()
    if not annotations:
        console.print("[yellow]No annotations to save[/yellow]")
        return dataset

    # Step 1: Build subtasks DataFrame and skill→index mapping
    all_skill_names = set(skill.name for ep in annotations.values() for skill in ep.skills)
    console.print(f"[cyan]Found {len(all_skill_names)} unique subtasks[/cyan]")
    subtasks_df = pd.DataFrame(
        [{"subtask": name, "subtask_index": i} for i, name in enumerate(sorted(all_skill_names))]
    ).set_index("subtask")
    skill_to_idx = {name: int(subtasks_df.loc[name, "subtask_index"]) for name in all_skill_names}

    # Step 2: Build per-frame subtask_index array
    console.print(f"[cyan]Creating subtask_index array for {len(dataset)} frames...[/cyan]")
    subtask_indices = np.full(len(dataset), -1, dtype=np.int64)
    for ep_idx, ep_skills in annotations.items():
        ep = dataset.meta.episodes[ep_idx]
        for frame_idx in range(ep["dataset_from_index"], ep["dataset_to_index"]):
            timestamp = float(dataset.hf_dataset[int(frame_idx)]["timestamp"])
            skill = get_skill_for_timestamp(ep_skills.skills, timestamp)
            if skill and skill.name in skill_to_idx:
                subtask_indices[frame_idx] = skill_to_idx[skill.name]

    # Step 3: Save subtasks.parquet to the original dataset root
    subtasks_path = dataset.root / "meta" / "subtasks.parquet"
    subtasks_path.parent.mkdir(parents=True, exist_ok=True)
    subtasks_df.to_parquet(subtasks_path, engine="pyarrow", compression="snappy")
    console.print(f"[green]✓ Saved subtasks to {subtasks_path}[/green]")

    # Step 4: Save raw skill annotations as skills.json for reference
    skills_path = dataset.root / "meta" / "skills.json"
    skills_data = {
        "coarse_description": annotations[next(iter(annotations))].description,
        "skill_to_subtask_index": skill_to_idx,
        "episodes": {str(ep_idx): ann.to_dict() for ep_idx, ann in annotations.items()},
    }
    with open(skills_path, "w") as f:
        json.dump(skills_data, f, indent=2)
    console.print(f"[green]✓ Saved skill annotations to {skills_path}[/green]")

    # Step 5: Add subtask_index feature and write new dataset
    output_dir = Path(output_dir) if output_dir else dataset.root.parent / f"{dataset.root.name}_with_subtasks"
    repo_id = repo_id or f"{dataset.repo_id}_with_subtasks"
    console.print("[cyan]Adding subtask_index feature to dataset...[/cyan]")
    new_dataset = add_features(
        dataset=dataset,
        features={"subtask_index": (subtask_indices, {"dtype": "int64", "shape": (1,), "names": None})},
        output_dir=output_dir,
        repo_id=repo_id,
    )

    # Step 6: Copy subtasks.parquet and skills.json into the new output directory
    shutil.copy(subtasks_path, output_dir / "meta" / "subtasks.parquet")
    shutil.copy(skills_path, output_dir / "meta" / "skills.json")

    console.print(f"[bold green]✓ Successfully added subtask_index feature![/bold green]")
    console.print(f"  New dataset saved to: {new_dataset.root}")
    console.print(f"  Total subtasks: {len(subtasks_df)}")
    return new_dataset


def main():
    parser = argparse.ArgumentParser(description="Automatic skill annotation for LeRobot datasets using VLMs")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=str)
    data_group.add_argument("--repo-id", type=str)
    
    parser.add_argument("--video-key", type=str, required=True)
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()
    console = Console()

    dataset = LeRobotDataset(root=args.data_dir, repo_id="local/dataset" if args.data_dir else args.repo_id, download_videos=not bool(args.data_dir))
    
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    vlm = get_vlm(args.model, args.device, dtype_map[args.dtype])
    
    annotator = SkillAnnotator(vlm=vlm, console=console, batch_size=args.batch_size)
    annotations = annotator.annotate_dataset(dataset=dataset, video_key=args.video_key)
    
    save_skill_annotations(dataset, annotations, args.output_dir)

if __name__ == "__main__":
    main()