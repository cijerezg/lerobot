"""Checkpoint folders for the object-choice sim runs: internal 1200 with the released vision encoder / language model put back.

    uv run python lerobot/adhocs/object_choice/merge_checkpoints.py

Writes outputs/analysis/object_choice_2026-11-17/checkpoints/<cell>/pretrained_model (refuses to overwrite):
  base_vision      vision backbone = base; language model, action expert and the rest = internal 1200
  base_vision_lm   vision backbone + language model (transformer and lm_head) = base; action expert and the rest = internal 1200
The all-internal cell is the internal checkpoint itself. A folder holds internal's config / processor files and one model.safetensors,
so run_sim_eval.sh takes it as any checkpoint. Base tensors are cast to the dtype internal stores under the same name.
"""
import glob
import os
import shutil

from safetensors.torch import load_file, save_file

INTERNAL = "outputs/runs/rebot-internal-only-20261021/checkpoints/001200/pretrained_model"
BASE = "outputs/models/MolmoAct2"
OUT = "outputs/analysis/object_choice_2026-11-17/checkpoints"
VISION = ("model.model.vision_backbone.",)
LANGUAGE = ("model.model.transformer.", "model.lm_head.")
CELLS = {"base_vision": VISION, "base_vision_lm": VISION + LANGUAGE}


def main() -> None:
    internal = load_file(os.path.join(INTERNAL, "model.safetensors"))
    # the released files: several shards, names without the policy wrapper's leading "model."
    base = {"model." + name: value for file in sorted(glob.glob(os.path.join(BASE, "model-*.safetensors"))) for name, value in load_file(file).items()
            if ("model." + name).startswith(VISION + LANGUAGE)}
    for cell, prefixes in CELLS.items():
        folder = os.path.join(OUT, cell, "pretrained_model")
        os.makedirs(folder)  # an existing folder stops the script
        swapped = [name for name in internal if name.startswith(prefixes)]
        assert set(swapped) == {name for name in base if name.startswith(prefixes)}, cell
        state = dict(internal)
        for name in swapped:
            assert base[name].shape == internal[name].shape, name
            state[name] = base[name].to(internal[name].dtype)
        save_file(state, os.path.join(folder, "model.safetensors"), metadata={"format": "pt"})
        for file in os.listdir(INTERNAL):
            if file != "model.safetensors":
                shutil.copyfile(os.path.join(INTERNAL, file), os.path.join(folder, file))
        print(f"{cell}: {len(swapped)} of {len(state)} tensors from base ({sum(state[n].numel() for n in swapped) / 1e9:.2f} B parameters) -> {folder}")


if __name__ == "__main__":
    main()
