import logging
import json
import gradio as gr
from pathlib import Path
import traceback
import cv2

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05_full.annotate.subtask_annotate import (
    VideoExtractor,
    Skill,
    EpisodeSkills,
    load_skill_annotations,
    create_subtasks_dataframe,
    create_subtask_index_array,
    save_subtasks
)
from lerobot.datasets.dataset_tools import add_features
from rich.console import Console

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset_metadata(data_dir, repo_id, video_key):
    """Loads dataset and existing annotations."""
    try:
        if data_dir:
            dataset = LeRobotDataset(repo_id="local/dataset", root=data_dir, download_videos=False)
        elif repo_id:
            dataset = LeRobotDataset(repo_id=repo_id, download_videos=True)
        else:
            return None, "Error: Must provide either Data Directory or Repo ID", gr.update(maximum=0), {}, ""

        if video_key not in dataset.meta.video_keys:
            available = ", ".join(dataset.meta.video_keys)
            return None, f"Error: Video key '{video_key}' not found. Available: {available}", gr.update(maximum=dataset.meta.total_episodes - 1), {}, ""

        # Load existing annotations
        existing_ann_data = load_skill_annotations(dataset.root)
        annotations = {}
        coarse_desc = "Perform the demonstrated manipulation task."
        
        if existing_ann_data and "episodes" in existing_ann_data:
            coarse_desc = existing_ann_data.get("coarse_description", coarse_desc)
            for ep_idx_str, ep_data in existing_ann_data["episodes"].items():
                skills = [Skill.from_dict(s) for s in ep_data.get("skills", [])]
                annotations[int(ep_idx_str)] = EpisodeSkills(
                    episode_index=int(ep_idx_str),
                    description=ep_data.get("description", coarse_desc),
                    skills=skills
                )
                
        msg = f"Successfully loaded dataset with {dataset.meta.total_episodes} episodes. Found {len(annotations)} existing episode annotations."
        return dataset, msg, gr.update(maximum=dataset.meta.total_episodes - 1, value=0), annotations, coarse_desc
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(traceback.format_exc())
        return None, f"Error loading dataset: {e}", gr.update(maximum=0), {}, ""



def load_episode_video(dataset, episode_idx, video_key, annotations, coarse_goal):
    """Extracts video for the given episode and prepares the text area."""
    if dataset is None:
        return None, "Please load a dataset first.", ""
    
    try:
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, video_key)
        if not video_path.exists():
            return None, f"Video not found: {video_path}", ""
        
        ep = dataset.meta.episodes[episode_idx]
        start_ts = float(ep[f"videos/{video_key}/from_timestamp"])
        end_ts = float(ep[f"videos/{video_key}/to_timestamp"])
        duration = end_ts - start_ts

        extractor = VideoExtractor()
        extracted_path = extractor.extract_episode_video(
            video_path, start_ts, end_ts, target_fps=30
        )
        
        # Format existing annotations or provide a template
        annotation_text = ""
        if episode_idx in annotations:
            ep_skills = annotations[episode_idx].skills
            for skill in ep_skills:
                annotation_text += f"{skill.start:.2f}, {skill.end:.2f}, {skill.name}\n"
        else:
            annotation_text = f"0.00, {duration:.2f}, New Skill"
            
        status = f"Loaded episode {episode_idx} (Duration: {duration:.2f}s)"
        return str(extracted_path), status, annotation_text
    except Exception as e:
        logger.error(f"Failed to load video: {e}")
        logger.error(traceback.format_exc())
        return None, f"Error: {e}", ""

def parse_annotations_text(text):
    """Parses the multiline CSV text into Skill objects."""
    skills = []
    for line in text.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            try:
                start = float(parts[0])
                end = float(parts[1])
                name = ",".join(parts[2:]) # In case skill name has commas
                skills.append(Skill(name=name, start=start, end=end))
            except ValueError:
                raise ValueError(f"Invalid number format in line: '{line}'")
        else:
            raise ValueError(f"Invalid format in line: '{line}'. Expected 'start, end, name'")
    return skills

def save_episode_annotations(dataset, episode_idx, annotation_text, coarse_goal, annotations):
    """Saves the given text area as EpisodeSkills and updates the annotations dictionary."""
    if dataset is None:
        return annotations, "Error: No dataset loaded."
    
    try:
        skills = parse_annotations_text(annotation_text)
        if not skills:
            return annotations, "Error: No skills parsed."
            
        ep_skills = EpisodeSkills(
            episode_index=episode_idx,
            description=coarse_goal,
            skills=skills
        )
        
        # Create a new dict to trigger Gradio state update
        new_annotations = annotations.copy()
        new_annotations[episode_idx] = ep_skills
        
        # Save working progress to skills.json by performing a mini compile just for the json,
        # but to be safe and match `subtask_annotate.py`, we wait until "Compile Dataset"
        # However, to prevent data loss, let's write to a backup raw json or just keep in state.
        # Actually, let's let the user know it's saved in session state.
        return new_annotations, f"Saved annotations for episode {episode_idx} in session state. Remember to click 'Compile Dataset'."
    except Exception as e:
        return annotations, f"Error parsing annotations: {e}"

def compile_dataset(dataset, annotations, output_dir):
    """Compiles the full dataset protecting the original dataset completely."""
    if dataset is None:
        return "Error: No dataset loaded."
    if not annotations:
        return "Error: No annotations to save."
    if not output_dir or output_dir.strip() == "":
        return "Error: An Output Directory is strongly recommended (or required) so the original dataset isn't overwritten."
        
    try:
        out_path = Path(output_dir)
        # Prevent overwriting original dataset
        if str(out_path.resolve()) == str(dataset.root.resolve()):
            return "Error: Output directory cannot be the same as the original dataset. Please provide a new path."
            
        console = Console()
        console.print("[cyan]Creating subtasks DataFrame...[/cyan]")
        subtasks_df, skill_to_subtask_idx = create_subtasks_dataframe(annotations)
        
        console.print("[cyan]Creating subtask_index array...[/cyan]")
        subtask_indices = create_subtask_index_array(dataset, annotations, skill_to_subtask_idx)
        
        # Add feature using dataset_tools (this generates the new dataset)
        feature_info = {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
        repo_id = f"{dataset.repo_id}_with_subtasks"
        new_dataset = add_features(
            dataset=dataset,
            features={
                "subtask_index": (subtask_indices, feature_info),
            },
            output_dir=out_path,
            repo_id=repo_id,
        )
        
        # After creating the new dataset, write our explicit metadata to ITS meta folder
        output_meta_dir = out_path / "meta"
        output_meta_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save subtasks.parquet directly to the new dataset
        save_subtasks(subtasks_df, out_path, console)
        
        # 2. Save the compiled skills.json to the new dataset
        skills_path = output_meta_dir / "skills.json"
        
        existing_skills_data = load_skill_annotations(dataset.root)
        new_episodes = {str(ep_idx): ann.to_dict() for ep_idx, ann in annotations.items()}
        
        if existing_skills_data:
            merged_episodes = existing_skills_data.get("episodes", {}).copy()
            merged_episodes.update(new_episodes)
            merged_skill_to_subtask = existing_skills_data.get("skill_to_subtask_index", {}).copy()
            merged_skill_to_subtask.update(skill_to_subtask_idx)
            coarse_desc = existing_skills_data.get("coarse_description", annotations[next(iter(annotations))].description)
            skills_data = {
                "coarse_description": coarse_desc,
                "skill_to_subtask_index": merged_skill_to_subtask,
                "episodes": merged_episodes,
            }
        else:
            skills_data = {
                "coarse_description": annotations[next(iter(annotations))].description,
                "skill_to_subtask_index": skill_to_subtask_idx,
                "episodes": new_episodes,
            }
            
        with open(skills_path, "w") as f:
            json.dump(skills_data, f, indent=2)
            
        console.print(f"[bold green]✓ Successfully compiled dataset to {out_path}[/bold green]")
        return f"Success! Dataset compiled with subtasks at: {new_dataset.root}"
    except Exception as e:
        logger.error(f"Failed to compile dataset: {e}")
        logger.error(traceback.format_exc())
        return f"Error compiling dataset: {e}"

def build_app():
    with gr.Blocks(title="LeRobot Manual Skill Annotator") as app:
        gr.Markdown("# 🤖 LeRobot Manual Skill Annotator")
        gr.Markdown("A simple UI to manually annotate atomic skills (Pick, Place, etc.) for your episodes. This mimics the VLM output in `subtask_annotate.py`.")
        
        dataset_state = gr.State(None)
        annotations_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Load Dataset")
                data_dir_input = gr.Textbox(label="Local Data Directory (e.g., outputs/annotated_dataset-v1)", placeholder="/path/to/dataset")
                repo_id_input = gr.Textbox(label="Or HF Repo ID", placeholder="user/dataset")
                video_key_input = gr.Textbox(label="Video Key", value="observation.images.base")
                load_btn = gr.Button("Load Dataset", variant="primary")
                dataset_status = gr.Textbox(label="Dataset Status", interactive=False)
                
                gr.Markdown("### 2. Output Settings")
                output_dir_input = gr.Textbox(label="Output Directory (Optional)", placeholder="/path/to/save_new_dataset")
                compile_btn = gr.Button("Compile Dataset (Saves Everything)", variant="primary")
                compile_status = gr.Textbox(label="Compile Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 3. Annotate Episodes")
                with gr.Row():
                    episode_slider = gr.Slider(minimum=0, maximum=0, step=1, label="Episode Index", interactive=True)
                    load_ep_btn = gr.Button("Load Episode Video")
                
                ep_status = gr.Textbox(label="Episode Status", interactive=False)
                video_player = gr.Video(label="Episode Video Tape", interactive=False)
                gr.Markdown("*Tip: Use the three-dots menu (⋮) on the video player to increase playback speed (e.g., to 1.5x or 2x).*")
                
                coarse_goal_input = gr.Textbox(label="Coarse Goal", value="Perform the demonstrated manipulation task.")
                
                gr.Markdown("#### Annotations (Format: `start_time, end_time, skill_name`)")
                gr.Markdown("Example:\n`0.00, 2.50, Move to block`\n`2.50, 4.00, Grasp block`")
                annotation_input = gr.Textbox(label="Edit Skills", lines=10)
                
                save_ep_btn = gr.Button("Save Episode Annotations", variant="secondary")
                save_status = gr.Textbox(label="Save Status", interactive=False)
                
        # Callbacks
        load_btn.click(
            load_dataset_metadata,
            inputs=[data_dir_input, repo_id_input, video_key_input],
            outputs=[dataset_state, dataset_status, episode_slider, annotations_state, coarse_goal_input]
        )
        
        load_ep_btn.click(
            load_episode_video,
            inputs=[dataset_state, episode_slider, video_key_input, annotations_state, coarse_goal_input],
            outputs=[video_player, ep_status, annotation_input]
        )
        
        save_ep_btn.click(
            save_episode_annotations,
            inputs=[dataset_state, episode_slider, annotation_input, coarse_goal_input, annotations_state],
            outputs=[annotations_state, save_status]
        )
        
        compile_btn.click(
            compile_dataset,
            inputs=[dataset_state, annotations_state, output_dir_input],
            outputs=[compile_status]
        )

    return app

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
