"""Memory prompt clauses in the MolmoAct2 prompt seam (_build_robot_text)."""

import pytest

pytest.importorskip("transformers", reason="molmoact2 processor module imports policy deps")

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.molmoact2.configuration_molmoact2 import (  # noqa: E402
    infer_molmoact2_max_sequence_length,
)
from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    MolmoAct2MaskedNormalizerProcessorStep,
    _build_robot_text,
)
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402


def build(**overrides):
    kwargs = {
        "task": "fold the towel",
        "discrete_state_string": "",
        "num_images": 0,
    }
    kwargs.update(overrides)
    return _build_robot_text(**kwargs)


def test_legacy_prompt_is_byte_identical_when_memory_off():
    """All memory params None → the exact base prompt (no optional clause renders)."""
    prompt = build()
    assert prompt == (
        "<|im_start|>user\n"
        "The task is to fold the towel. "
        "Given these, what action should the robot take to complete the task?<|im_end|>\n"
        "<|im_start|>assistant\n<action_output>"
    ), f"unexpected legacy prompt: {prompt}"


def test_current_subtask_clause():
    prompt = build(current_subtask="grab the far side")
    assert "The current step is grab the far side." in prompt


def test_metadata_clause_partial_rendering():
    prompt = build(metadata={"quality": 5, "mistake": False, "speed": "fast"})
    assert "The quality is 5 of 5." in prompt
    assert "The robot made no mistakes." in prompt
    assert "The speed is fast." in prompt

    prompt = build(metadata={"mistake": True})
    assert "The robot made a mistake." in prompt
    assert "quality" not in prompt

    prompt = build(metadata={})
    assert "quality" not in prompt and "mistake" not in prompt and "speed" not in prompt


def test_history_clause_renders_placeholders():
    from lerobot.policies.molmoact2.processor_molmoact2 import STATE_HISTORY_TOKEN

    prompt = build(num_history_states=3)
    assert (
        f"The recent states of the robot, oldest to newest, are {STATE_HISTORY_TOKEN * 3}." in prompt
    )
    # Image history never renders in the prompt (MEM video encoder path).
    assert "earlier frames" not in prompt


def test_extract_history_image_stack_from_complementary():
    from types import SimpleNamespace

    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep

    # Stub the HF image processor: one crop per frame (crop_mode "resize"), so the
    # reshape/order/times logic is tested without loading the checkpoint processor.
    def fake_image_processor(images):
        return {"pixel_values": np.arange(len(images), dtype=np.float32)[:, None, None].repeat(
            4, axis=1
        ).repeat(5, axis=2)}

    stub = SimpleNamespace(
        processor=SimpleNamespace(image_processor=fake_image_processor),
        history_stride_seconds=1.0,
    )
    extract = MolmoAct2PackInputsProcessorStep._extract_history_image_stack
    frames = torch.randint(0, 255, (2, 3, 3, 8, 8), dtype=torch.uint8)  # (B, T_h, C, H, W)
    complementary = {
        "history.observation.images.top": frames,
        "history.observation.images.top_is_pad": torch.zeros(2, 3, dtype=torch.bool),
        "history.observation.images.wrist": frames,
        "history.observation.images.wrist_is_pad": torch.zeros(2, 3, dtype=torch.bool),
    }
    image_keys = ["observation.images.top", "observation.images.wrist"]

    out, times = extract(stub, complementary, image_keys, 2)
    assert out.shape == (2, 2, 3, 4, 5)  # (B, cams, T_h, patches, patch_dim)
    assert out.dtype == torch.bfloat16
    # Flat order is (b, cam, t): sample 1 / cam wrist / slot 0 is flat frame 9.
    assert out[1, 1, 0, 0, 0].item() == 9.0
    # Times are seconds-before-now, oldest → newest.
    assert times.tolist() == [3.0, 2.0, 1.0]

    assert extract(stub, {}, image_keys, 2) is None
    # A camera subset is a config error, not a silent partial stack.
    with pytest.raises(ValueError, match="exactly the prompt cameras"):
        extract(
            stub,
            {"history.observation.images.top": frames},
            image_keys,
            2,
        )


def test_history_clause_off_when_zero():
    assert "recent states" not in build(num_history_states=0)


def test_history_state_normalized_same_as_current_state():
    """history.{OBS_STATE} rides in COMPLEMENTARY_DATA (batch_to_transition routes any
    "history.*" key there, not OBSERVATION), so the base NormalizerProcessorStep would
    skip it entirely and it would reach the prompt as raw joint values. The MolmoAct2
    normalizer must apply the exact same OBS_STATE stats to it."""
    features = {OBS_STATE: PolicyFeature(FeatureType.STATE, (2,))}
    norm_map = {FeatureType.STATE: NormalizationMode.MIN_MAX}
    stats = {OBS_STATE: {"min": torch.tensor([0.0, -1.0]), "max": torch.tensor([1.0, 1.0])}}
    normalizer = MolmoAct2MaskedNormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)

    current_state = torch.tensor([[0.5, 0.0]])
    history_state = torch.tensor([[[0.5, 0.0], [1.0, 1.0]]])  # (B=1, T_h=2, D=2)
    transition = create_transition(
        observation={OBS_STATE: current_state},
        complementary_data={"history.observation.state": history_state},
    )

    normalized = normalizer(transition)
    normalized_current = normalized[TransitionKey.OBSERVATION][OBS_STATE]
    normalized_history = normalized[TransitionKey.COMPLEMENTARY_DATA]["history.observation.state"]

    # Same stats, same math: the history's first (oldest) frame equals the current state's result.
    assert torch.allclose(normalized_history[0, 0], normalized_current[0], atol=1e-6)
    assert torch.allclose(normalized_history[0, 1], torch.tensor([1.0, 1.0]), atol=1e-6)


def test_max_sequence_length_budgets_history_clause():
    """The inferred sequence cap must grow with the history clause, or enabling
    memory.history_keys trips the pack step's hard length check."""
    kwargs = dict(
        num_images=2, state_dim=7, action_dim=7, action_horizon=50, include_discrete_action=True
    )
    base = infer_molmoact2_max_sequence_length(**kwargs)
    with_history = infer_molmoact2_max_sequence_length(**kwargs, history_num_samples=64)
    # 64 placeholder positions + 16 lead-in = 80 raw tokens, which survives the
    # round-up-to-64 regardless of the base's slack. (Realistic T_h is 5; the
    # budget is linear so the large value only makes the assertion robust.)
    assert with_history > base
    assert infer_molmoact2_max_sequence_length(**kwargs, history_num_samples=0) == base


def test_clause_order_task_then_memory_then_state():
    prompt = build(current_subtask="step two", discrete_state_string="<state_start><state_0><state_end>")
    task_pos = prompt.index("The task is to")
    current_pos = prompt.index("The current step is")
    state_pos = prompt.index("The current state of the robot is")
    assert task_pos < current_pos < state_pos


# ── Phase 3: subtask generation prompt + vocab snap ──────────────────────────

from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    _build_subtask_generation_text,
    build_generation_answer,
    parse_generation_answer,
    snap_to_subtask_vocab,
)

VOCAB = ["reach the cup", "grasp the cup", "lift the cup"]


def test_generation_prompt_asks_for_next_step():
    prompt = _build_subtask_generation_text(
        task="put the cup on the shelf",
        discrete_state_string="",
        num_images=0,
    )
    assert "what step should the robot perform next?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    # The action-output token never appears — this prompt decodes text, not actions.
    assert "<action" not in prompt


def test_snap_to_subtask_vocab():
    assert snap_to_subtask_vocab("grasp the cup", VOCAB) == 1
    assert snap_to_subtask_vocab("  Grasp the Cup. ", VOCAB) == 1  # case/punct-insensitive
    assert snap_to_subtask_vocab("grasp cup", VOCAB) == 1          # fuzzy
    assert snap_to_subtask_vocab("do a backflip", VOCAB) == -1     # no match


# ── MEM summary memory: generation-prompt clause + answer format ─────────────


def gen_prompt(**overrides):
    kwargs = {
        "task": "fold",
        "discrete_state_string": "",
        "num_images": 0,
    }
    kwargs.update(overrides)
    return _build_subtask_generation_text(**kwargs)


def test_generation_prompt_memory_clause():
    assert " Memory: I picked up the truck." in gen_prompt(summary="I picked up the truck.")
    assert " Memory: none yet." in gen_prompt(summary="")  # empty memory, clause on
    assert "Memory:" not in gen_prompt()  # feature off / dropout


def test_generation_answer_roundtrip():
    answer = build_generation_answer("grasp the cup", "I picked up the truck.")
    assert answer == "Memory: I picked up the truck. Subtask: grasp the cup"
    assert parse_generation_answer(answer) == ("grasp the cup", "I picked up the truck.")

    assert parse_generation_answer(build_generation_answer("grasp the cup", "")) == ("grasp the cup", "")

    # Subtask-only training/decodes carry no memory span.
    assert build_generation_answer("grasp the cup", None) == "grasp the cup"
    assert parse_generation_answer("grasp the cup") == ("grasp the cup", None)
