"""Memory prompt clauses in the MolmoAct2 prompt seam (_build_robot_text)."""

import pytest

pytest.importorskip("transformers", reason="molmoact2 processor module imports policy deps")

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.policies.molmoact2.anchor_encoding import ANCHOR_KEY  # noqa: E402
from lerobot.policies.molmoact2.configuration_molmoact2 import (  # noqa: E402
    infer_molmoact2_max_sequence_length,
)
from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    MolmoAct2MaskedNormalizerProcessorStep,
    MolmoAct2UnifiedLayoutProcessorStep,
    _build_robot_text,
)
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.action_metrics import ACTION_HOLD_KEY  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402


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


def test_unified_layout_pads_state_history_and_action_without_reordering():
    step = MolmoAct2UnifiedLayoutProcessorStep(
state_dim=8, action_dim=8)
    state = torch.arange(7, dtype=torch.float32)[None]
    action = torch.arange(6, dtype=torch.float32)[None, None]
    transition = create_transition(
        observation={OBS_STATE: state},
        action=action,
        complementary_data={"history.observation.state": state[:, None].clone()},
    )

    packed = step(transition)
    packed_state = packed[TransitionKey.OBSERVATION][OBS_STATE]
    packed_action = packed[TransitionKey.ACTION]
    complementary = packed[TransitionKey.COMPLEMENTARY_DATA]

    assert packed_state.tolist() == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]]
    assert packed_action.tolist() == [[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0]]]
    assert complementary["history.observation.state"].tolist() == [
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]]
    ]
    assert complementary["state_dim_is_pad"].tolist() == [[False] * 7 + [True]]
    assert complementary["action_dim_is_pad"].tolist() == [[False] * 6 + [True, True]]


def test_anchor_encoding_subtracts_state_from_action_in_matching_slots():
    """Every packed source is joint-space, so the anchor is just the state prefix.

    The old ee guard lived here; it is gone because FMB, its only trigger, now reads
    measured joints instead of Cartesian deltas.
    """
    from lerobot.policies.molmoact2.anchor_encoding import AnchorEncodeStep

    state = torch.arange(7, dtype=torch.float32)[None]
    action = torch.arange(7, dtype=torch.float32)[None, None].repeat(1, 2, 1) + 3.0
    transition = create_transition(observation={OBS_STATE: state}, action=action)

    encoded = AnchorEncodeStep("anchor")(transition)[TransitionKey.ACTION]
    assert encoded.tolist() == [[[3.0] * 7, [3.0] * 7]]


def test_metadata_clause_partial_rendering():
    prompt = build(metadata={"quality": 5, "mistake": False, "speed": 4})
    assert "The quality is 5 of 5." in prompt
    assert "The robot made no mistakes." in prompt
    assert "The speed is 4 of 5." in prompt

    prompt = build(metadata={"mistake": True})
    assert "The robot made a mistake." in prompt
    assert "quality" not in prompt

    prompt = build(metadata={})
    assert "quality" not in prompt and "mistake" not in prompt and "speed" not in prompt
    assert "precision" not in prompt and "contact" not in prompt


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_precision_clause_renders_each_level(level):
    prompt = build(metadata={"precision": level})
    assert f" The precision is {level} of 5. Given these," in prompt


@pytest.mark.parametrize(
    ("code", "phrase"),
    [
        (0, "a top pinch"),
        (2, "a rim pinch"),
        (11, "an insertion"),
        (13, "a tool drag"),
        (14, "not applicable"),
    ],
)
def test_contact_clause_renders_vocab_phrase(code, phrase):
    prompt = build(metadata={"contact": code})
    assert f" The contact is {phrase}. Given these," in prompt


def test_contact_clause_rejects_unknown_code():
    with pytest.raises(ValueError, match="not in the vocabulary"):
        build(metadata={"contact": 15})


def test_metadata_clause_order_speed_precision_contact():
    prompt = build(metadata={"quality": 4, "mistake": False, "speed": 3, "precision": 4, "contact": 2})
    assert (
        " The quality is 4 of 5. The robot made no mistakes. The speed is 3 of 5."
        " The precision is 4 of 5. The contact is a rim pinch. Given these,"
    ) in prompt


def test_metadata_prompt_byte_identical_without_precision_and_contact():
    """No precision / contact key (every root before the annotation pass) renders
    exactly the speed-era prompt, clause for clause."""
    prompt = build(metadata={"quality": 5, "mistake": False, "speed": 5})
    assert prompt == (
        "<|im_start|>user\n"
        "The task is to fold the towel. The quality is 5 of 5. The robot made no mistakes. "
        "The speed is 5 of 5. "
        "Given these, what action should the robot take to complete the task?<|im_end|>\n"
        "<|im_start|>assistant\n<action_output>"
    )


def _extract(complementary, batch_size):
    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep

    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    return step._extract_metadata(complementary, batch_size)


def test_extract_precision_and_contact_from_columns():
    out = _extract(
        {
            "metadata_quality": torch.tensor([5.0, 3.0, 4.0]),
            "metadata_mistake": torch.tensor([0.0, 1.0, 0.0]),
            "metadata_speed": torch.tensor([2.0, 5.0, 4.0]),
            "metadata_precision": torch.tensor([4.0, -1.0, 1.0], dtype=torch.bfloat16),
            "metadata_contact": torch.tensor([14.0, 2.0, -1.0], dtype=torch.bfloat16),
        },
        batch_size=3,
    )
    assert out[0] == {"quality": 5, "mistake": False, "speed": 2, "precision": 4, "contact": 14}
    # -1 is the unlabelled sentinel: that key alone is left out.
    assert out[1] == {"quality": 3, "mistake": True, "speed": 5, "contact": 2}
    assert out[2] == {"quality": 4, "mistake": False, "speed": 4, "precision": 1}


def test_prompt_byte_identical_when_precision_and_contact_columns_absent_or_unlabelled():
    base = {
        "metadata_quality": torch.tensor([5.0]),
        "metadata_mistake": torch.tensor([0.0]),
        "metadata_speed": torch.tensor([3.0]),
    }
    unlabelled = base | {
        "metadata_precision": torch.full((1,), -1.0, dtype=torch.bfloat16),
        "metadata_contact": torch.full((1,), -1.0, dtype=torch.bfloat16),
    }
    absent_metadata = _extract(base, 1)[0]
    assert absent_metadata == {"quality": 5, "mistake": False, "speed": 3}
    assert _extract(unlabelled, 1)[0] == absent_metadata
    assert build(metadata=_extract(unlabelled, 1)[0]) == build(metadata=absent_metadata)


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
        history_times_seconds=None,
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

    # Configured instants win over the uniform ladder: the -6/-4/-2 window is not
    # three frames at a 6 s stride, and only these stamps tell the ViT which is which.
    stub.history_times_seconds = [-6.0, -4.0, -2.0]
    stub.history_stride_seconds = 2.0
    _, times = extract(stub, complementary, image_keys, 2)
    assert times.tolist() == [6.0, 4.0, 2.0]

    # A stamp list that does not match the gathered window is a config drift, not a
    # window to silently re-stamp.
    stub.history_times_seconds = [-6.0, -4.0]
    with pytest.raises(ValueError, match="drifted apart"):
        extract(stub, complementary, image_keys, 2)
    stub.history_times_seconds = None
    stub.history_stride_seconds = 1.0

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


# ── History slots (docs/notes/memory_history_slot.md §3) ───────────────────────


SLOT_CLAUSE = (
    "Images 3 to 4 are the same cameras as images 1 to 2, 2 seconds earlier, "
    "with the frames from 4 and 6 seconds earlier folded in."
)


def test_history_image_clause_names_the_slots():
    prompt = build(num_images=4, num_history_images=2, num_history_states=3)
    assert prompt.startswith(
        "Image 1<|image|>Image 2<|image|>Image 3<|image|>Image 4<|image|><|im_start|>user\n"
    )
    assert f" {SLOT_CLAUSE} " in prompt
    # Past states first, then the slot clause, then the question.
    assert prompt.index("The recent states") < prompt.index(SLOT_CLAUSE) < prompt.index("Given these")
    # The ages word the clause: a 1 s / 3 s / 5 s ladder names them, newest first.
    custom = build(num_images=4, num_history_images=2, history_ages_seconds=[5.0, 3.0, 1.0])
    assert ", 1 seconds earlier, with the frames from 3 and 5 seconds earlier folded in." in custom


def test_history_image_clause_absent_without_slots():
    """No slots (history off, inplace mode, no history) = the legacy prompt, byte for byte."""
    assert build(num_images=2, num_history_images=0) == build(num_images=2)
    assert "folded in" not in build(num_images=2, num_history_states=3)


def test_generation_prompt_carries_the_slot_clause_too():
    from lerobot.policies.molmoact2.processor_molmoact2 import _build_subtask_generation_text

    kwargs = {"task": "fold the towel", "discrete_state_string": ""}
    with_slots = _build_subtask_generation_text(**kwargs, num_images=4, num_history_images=2)
    assert f" {SLOT_CLAUSE} " in with_slots
    assert "<|image|>" * 1 in with_slots and with_slots.count("<|image|>") == 4
    assert "folded in" not in _build_subtask_generation_text(**kwargs, num_images=2)


def test_max_sequence_length_budgets_the_slots_through_num_images():
    """The slots are ordinary prompt images: C extra images = C x 196 more tokens, and
    the clause text rides in the history clause budget."""
    kwargs = {"state_dim": 7, "action_dim": 7, "action_horizon": 30, "include_discrete_action": True}
    two = infer_molmoact2_max_sequence_length(num_images=2, history_num_samples=3, **kwargs)
    four = infer_molmoact2_max_sequence_length(num_images=4, history_num_samples=3, **kwargs)
    # Raw difference is 2 x 196; the cap rounds to a multiple of 64.
    assert four - two >= 2 * 196 - 64


def test_state_age_stamp_added_to_the_projected_past_states():
    """Each past-state token = W s + b + e_d(t_k), e_d the ViT's sinusoid in the text
    width, ages from history_image_times oldest → newest (memory_history_slot.md §5)."""
    from types import SimpleNamespace

    from torch import nn

    from lerobot.policies.molmoact2.modeling_molmoact2 import (
        MolmoAct2Policy,
        _sinusoidal_seconds_embedding,
    )

    torch.manual_seed(0)
    backbone = SimpleNamespace(vision_backbone=SimpleNamespace(_lerobot_history=None))
    stub = SimpleNamespace(
        _backbone=lambda: backbone,
        model=nn.Linear(1, 1),
        state_projector=nn.Linear(7, 6),
    )
    states = torch.randn(2, 3, 7)
    times = torch.tensor([6.0, 4.0, 2.0])
    batch = {
        "state_values": states,
        "state_values_mask": torch.ones(2, 3, dtype=torch.bool),
        "state_token_id": torch.tensor(9),
        "history_image_times": times,
    }
    MolmoAct2Policy._stash_history_inputs(stub, batch)
    embeds, mask, token_id = backbone._lerobot_state_values
    expected = stub.state_projector(states) + _sinusoidal_seconds_embedding(times, 6)
    torch.testing.assert_close(embeds, expected)
    assert token_id == 9 and mask.all()
    # Two slots with different ages get different stamps on the same state.
    same = torch.zeros(1, 3, 7)
    batch["state_values"] = same
    MolmoAct2Policy._stash_history_inputs(stub, batch)
    embeds = backbone._lerobot_state_values[0][0]
    assert not torch.allclose(embeds[0], embeds[2])


# ── The packed prompt with the real processor (skipped without the base checkpoint) ──

from pathlib import Path  # noqa: E402

BASE_PATH = Path(__file__).resolve().parents[3] / "outputs/MolmoAct2"
CAM_KEYS = ["observation.images.external_0", "observation.images.external_1", "observation.images.wrist_0"]


def _pack_step(**overrides):
    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep

    if not (BASE_PATH / "processor_config.json").is_file():
        pytest.skip("MolmoAct2 base checkpoint not present")
    kwargs = {
        "base_path": str(BASE_PATH),
        "action_mode": "continuous",
        "image_keys": list(CAM_KEYS),
        "chunk_size": 30,
        "max_action_dim": 8,
        "history_times_seconds": [-6.0, -4.0, -2.0],
    }
    kwargs.update(overrides)
    return MolmoAct2PackInputsProcessorStep(**kwargs)


def _history_transition(batch_size, present, history=True):
    torch.manual_seed(0)
    observation = {OBS_STATE: torch.zeros(batch_size, 8)}
    complementary = {}
    if history:
        complementary[f"history.{OBS_STATE}"] = torch.zeros(batch_size, 3, 8)
    for index, key in enumerate(CAM_KEYS):
        observation[key] = torch.randint(0, 255, (batch_size, 3, 64, 64), dtype=torch.uint8)
        if history:
            complementary[f"history.{key}"] = torch.randint(
                0, 255, (batch_size, 3, 3, 64, 64), dtype=torch.uint8
            )
        complementary[f"camera_is_present.{key}"] = torch.tensor([row[index] for row in present])
    return create_transition(
        observation=observation, action=torch.zeros(batch_size, 30, 8), complementary_data=complementary
    )


def test_packed_prompt_with_history_has_slots_placeholders_and_masked_absent_spans():
    from lerobot.policies.depth_pointmap.modeling_stream import wrist_cam_token_indices

    step = _pack_step()
    out = step(_history_transition(2, [[True, True, True], [True, False, True]]))
    comp = out[TransitionKey.COMPLEMENTARY_DATA]
    ids, mask = comp["input_ids"], comp["attention_mask"]
    text = step.processor.tokenizer.decode(ids[1][mask[1].bool()])
    assert (
        "Images 4 to 6 are the same cameras as images 1 to 3, 2 seconds earlier, "
        "with the frames from 4 and 6 seconds earlier folded in." in text
    )
    assert text.count("<extra_0>") == 3 and comp["state_values_mask"].all()
    patches = (ids == step._image_patch_id).sum(dim=1)
    assert patches.tolist() == [6 * 196, 6 * 196]
    # The absent camera (external_1, row 1) attends neither its current nor its history slot.
    for cam_index in (1, 4):
        sel = wrist_cam_token_indices(ids, image_patch_id=step._image_patch_id, num_images=6, cam_index=cam_index)
        assert mask[1, sel[1]].sum() == 0
        assert mask[0, sel[0]].sum() == 196
    assert comp["history_images"].shape[:3] == (2, 3, 3)
    assert comp["history_images_mask"].tolist() == [[True, True, True], [True, False, True]]
    # The slot's pixels are the gathered window's newest (-2 s) frame, not the current
    # one: with random windows the slot crops differ from the current crops; with the
    # newest frame set to the current frame they are the same crops.
    crops = comp["pixel_values"].view(2, 6, *comp["pixel_values"].shape[1:])
    assert not torch.allclose(crops[:, 3:], crops[:, :3])
    same = _history_transition(2, [[True, True, True], [True, True, True]])
    for key in CAM_KEYS:
        same[TransitionKey.COMPLEMENTARY_DATA][f"history.{key}"][:, -1] = same[TransitionKey.OBSERVATION][key]
    crops = step(same)[TransitionKey.COMPLEMENTARY_DATA]["pixel_values"]
    crops = crops.view(2, 6, *crops.shape[1:])
    torch.testing.assert_close(crops[:, 3:], crops[:, :3])


def test_packed_prompt_without_history_is_the_legacy_layout():
    step = _pack_step()
    out = step(_history_transition(1, [[True, True, True]], history=False))
    comp = out[TransitionKey.COMPLEMENTARY_DATA]
    text = step.processor.tokenizer.decode(comp["input_ids"][0][comp["attention_mask"][0].bool()])
    assert "folded in" not in text and "<extra_0>" not in text
    assert (comp["input_ids"] == step._image_patch_id).sum() == 3 * 196
    assert "history_images" not in comp


def test_packed_prompt_inplace_mode_adds_no_slots():
    step = _pack_step(history_image_mode="inplace")
    out = step(_history_transition(1, [[True, True, True]]))
    comp = out[TransitionKey.COMPLEMENTARY_DATA]
    text = step.processor.tokenizer.decode(comp["input_ids"][0][comp["attention_mask"][0].bool()])
    assert "folded in" not in text and text.count("<extra_0>") == 3
    assert (comp["input_ids"] == step._image_patch_id).sum() == 3 * 196
    assert comp["history_images"].shape[:3] == (1, 3, 3)


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


@pytest.mark.parametrize("anchor_encoded", [False, True])
def test_hold_trajectory_uses_the_action_normalizer(anchor_encoded):
    features = {
        OBS_STATE: PolicyFeature(FeatureType.STATE, (2,)),
        ACTION: PolicyFeature(FeatureType.ACTION, (2,)),
    }
    norm_map = {
        FeatureType.STATE: NormalizationMode.MIN_MAX,
        FeatureType.ACTION: NormalizationMode.MIN_MAX,
    }
    stats = {
        OBS_STATE: {"min": torch.tensor([0.0, 0.0]), "max": torch.tensor([10.0, 20.0])},
        ACTION: {
            "min": torch.tensor([-1.0, -2.0]) if anchor_encoded else torch.tensor([0.0, 0.0]),
            "max": torch.tensor([3.0, 6.0]) if anchor_encoded else torch.tensor([10.0, 20.0]),
        },
    }
    normalizer = MolmoAct2MaskedNormalizerProcessorStep(
        features=features, norm_map=norm_map, stats=stats
    )
    state = torch.tensor([[5.0, 10.0]])
    action = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    complementary = {ANCHOR_KEY: state} if anchor_encoded else {}

    normalized = normalizer(
        create_transition(
            observation={OBS_STATE: state},
            action=action,
            complementary_data=complementary,
        )
    )
    hold = normalized[TransitionKey.COMPLEMENTARY_DATA][ACTION_HOLD_KEY]

    if anchor_encoded:
        expected_raw_hold = torch.zeros_like(action)
    else:
        expected_raw_hold = state.unsqueeze(1).expand_as(action)
    expected = normalizer._apply_transform(
        expected_raw_hold,
        ACTION,
        features[ACTION].type,
        inverse=False,
    )
    torch.testing.assert_close(hold, expected)


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


def test_generation_prompt_carries_no_memory_clause():
    """The summary-memory seam was removed 2026-08-04: the generation prompt
    conditions on task + state only, and the answer is the bare subtask."""
    prompt = _build_subtask_generation_text(
        task="fold",
        discrete_state_string="joint tokens here",
        num_images=0,
    )
    assert "Memory:" not in prompt


# ── Continuous state history: projected past states, discrete current state ────


def test_emit_state_values_ships_history_rows_only():
    from types import SimpleNamespace

    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2PackInputsProcessorStep

    emit = MolmoAct2PackInputsProcessorStep._emit_state_values
    history = 10.0 + torch.arange(2 * 3 * 7, dtype=torch.float32).reshape(2, 3, 7)
    history_on = torch.tensor([True, False])  # sample 1 lost its history clause

    stub = SimpleNamespace(_state_history_id=42)
    complementary: dict = {}
    emit(stub, complementary, history, history_on)
    values, mask = complementary["state_values"], complementary["state_values_mask"]
    assert values.shape == (2, 3, 7) and mask.shape == (2, 3)
    torch.testing.assert_close(values, history)
    assert mask.tolist() == [[True, True, True], [False, False, False]]
    assert int(complementary["state_token_id"]) == 42

    # No history window (the legacy prompt, the generation prompt): nothing ships,
    # so the model's scatter stays off and the current state is digit tokens only.
    complementary = {}
    emit(stub, complementary, None, history_on)
    assert "state_values" not in complementary
    assert "state_token_id" not in complementary


def test_state_scatter_adds_masked_rows_onto_placeholders():
    from types import SimpleNamespace

    from torch import nn

    from lerobot.policies.molmoact2.modeling_molmoact2 import _patch_leaf_safe_input_embedding_update

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = SimpleNamespace(wte=nn.Embedding(16, 4), emb_drop=nn.Identity())
            self.vision_backbone = None
            self.config = SimpleNamespace(image_patch_id=7)

        def build_input_embeddings(self, input_ids, images=None, token_pooling=None):
            raise AssertionError("must be replaced")

    backbone = _Backbone()
    with torch.no_grad():
        backbone.transformer.wte.weight.fill_(1.0)
    _patch_leaf_safe_input_embedding_update(backbone)
    # Two samples, three past states each; sample 1's history clause was dropped.
    embeds = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[True, True, True], [False, False, False]])
    backbone._lerobot_state_values = (embeds, mask, 9)
    input_ids = torch.tensor([[5, 9, 9, 9], [5, 5, 5, 5]])

    out, _ = backbone.build_input_embeddings(input_ids)
    torch.testing.assert_close(out[0, 1:], 1.0 + embeds[0])
    torch.testing.assert_close(out[1], torch.ones(4, 4))
    torch.testing.assert_close(out[:, 0], torch.ones(2, 4))
    assert backbone._lerobot_state_values is None  # consume-once

    # Placeholder count must equal the shipped rows per sample.
    backbone._lerobot_state_values = (embeds, mask, 9)
    with pytest.raises(RuntimeError, match="do not match"):
        backbone.build_input_embeddings(torch.tensor([[5, 9, 9, 5], [5, 5, 5, 5]]))
