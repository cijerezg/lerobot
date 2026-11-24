"""Hidden states of every backbone layer for one prompt, and the raw cosines the chat page's Layers tab draws.

A run is one forward of one checkpoint. ``capture`` hooks every block of the language model (and of the action
expert, when the forward reaches it) and keeps each block's output at every position, [layers, positions, width].
Cosines are raw: no mean is subtracted.
"""
from __future__ import annotations

import torch

GROUP_OF_CLAUSE = {"control_mode_first": "control_mode", "task_first": "task", "depth_words": "text"}


@torch.no_grad()
def capture(adapter, forward) -> dict:
    """``forward()`` runs the model once and returns (input ids [S], attention mask [S] or None). Returns the
    ids, the mask, ``encoder`` [L, S, D] and ``expert`` [L, T, D] or None (first call of each block: for the
    action expert that is the first flow step)."""
    backbone = adapter.policy._backbone()
    encoder, expert = {}, {}

    def keep(store, index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store.setdefault(index, hidden.detach().reshape(-1, *hidden.shape[-2:])[0].cpu())

        return hook

    handles = [block.register_forward_hook(keep(encoder, i)) for i, block in enumerate(backbone.transformer.blocks)]
    handles += [block.register_forward_hook(keep(expert, i)) for i, block in enumerate(backbone._require_action_expert().blocks)]
    try:
        ids, mask = forward()
    finally:
        for handle in handles:
            handle.remove()
    return {
        "ids": ids.cpu(),
        "mask": None if mask is None else mask.cpu().bool(),
        "encoder": torch.stack([encoder[i] for i in sorted(encoder)]),
        "expert": torch.stack([expert[i] for i in sorted(expert)]) if expert else None,
    }


def find(row: list[int], pattern: list[int], start: int = 0) -> int | None:
    for at in range(start, len(row) - len(pattern) + 1):
        if row[at : at + len(pattern)] == pattern:
            return at
    return None


def label_tokens(adapter, tokenizer, run: dict) -> tuple[list[str], list[str]]:
    """Per position: the token as text, and its group. Groups: masked (not attended), image (patches), the
    clause of the action prompt it sits in (task, subtask, state, ...), special, text."""
    ids = run["ids"].tolist()
    positions = run["encoder"].shape[1]
    if len(ids) != positions:  # the blocks saw another sequence than the prompt: positions stay unnamed
        return [f"position {i}" for i in range(positions)], ["text"] * positions
    tokens = [tokenizer.decode([i]) for i in ids]
    patch = adapter._image_patch_token_id()
    special = set(tokenizer.all_special_ids)
    starts = {}
    for name, marker in adapter._PROMPT_CLAUSES:
        at = find(ids, tokenizer.encode(marker, add_special_tokens=False))
        if at is not None:
            name = GROUP_OF_CLAUSE.get(name, "metadata" if name.startswith("metadata") else name)
            starts[name] = min(at, starts.get(name, at))
    ordered = sorted(starts.items(), key=lambda item: item[1])
    clause = {}
    if "question" in starts:  # an action prompt: every clause runs to the next one, the question to <|im_end|>
        end = find(ids, tokenizer.encode("<|im_end|>", add_special_tokens=False), starts["question"]) or len(ids)
        for (name, start), stop in zip(ordered, [at for _, at in ordered[1:]] + [end]):
            clause.update(dict.fromkeys(range(start, stop), name))
    groups = []
    for at, token in enumerate(ids):
        if run["mask"] is not None and not run["mask"][at]:
            groups.append("masked")
        elif token == patch:
            groups.append("image")
        elif at in clause:
            groups.append(clause[at])
        else:
            groups.append("special" if token in special else "text")
    return tokens, groups


@torch.no_grad()
def within(hidden: torch.Tensor, groups: list[str], device) -> dict:
    """Per layer, for all attended positions and for each group: mean cosine between two different positions,
    and for all attended positions the mean norm and the participation ratio of the spectrum (how many
    directions the positions spread over; 1 = all on one line)."""
    members = {"all": [i for i, g in enumerate(groups) if g != "masked"]}
    for group in dict.fromkeys(groups):
        if group != "masked":
            members[group] = [i for i, g in enumerate(groups) if g == group]
    pair = {name: [] for name, index in members.items() if len(index) > 1}
    norm, spread = [], []
    for layer in hidden:
        x = layer.to(device).float()[members["all"]]
        unit = torch.nn.functional.normalize(x, dim=-1)
        gram = x @ x.T
        norm.append(float(x.norm(dim=-1).mean()))
        spread.append(float(gram.diagonal().sum() ** 2 / gram.square().sum()))
        full = torch.nn.functional.normalize(layer.to(device).float(), dim=-1)
        for name in pair:
            rows = full[members[name]]
            pair[name].append(float((rows.sum(dim=0).square().sum() - len(rows)) / (len(rows) * (len(rows) - 1))))
    return {"pair": pair, "norm": norm, "spread": spread}


def paired(a: list[str], b: list[str]) -> tuple[list[int], list[int]]:
    """Positions of two runs matched by kind: the k-th image patch with the k-th image patch, the k-th state
    token with the k-th state token, ... in the order of the first run. Masked positions and whatever one run
    has more of are left out."""
    of_b = {}
    for at, group in enumerate(b):
        of_b.setdefault(group, []).append(at)
    seen, first, second = {}, [], []
    for at, group in enumerate(a):
        k = seen.get(group, 0)
        seen[group] = k + 1
        if group != "masked" and k < len(of_b.get(group, [])):
            first.append(at)
            second.append(of_b[group][k])
    return first, second


@torch.no_grad()
def between(a: torch.Tensor, b: torch.Tensor, device, first: list[int], second: list[int]) -> torch.Tensor:
    """Cosine between matched positions of two runs, [layers, pairs]."""
    return torch.stack([
        torch.nn.functional.cosine_similarity(x[first].to(device).float(), y[second].to(device).float(), dim=-1).cpu()
        for x, y in zip(a, b)
    ])
