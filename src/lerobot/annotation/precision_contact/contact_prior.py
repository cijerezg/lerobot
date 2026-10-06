"""Contact-strategy prior by text rules (contact_strategy_rubric.md, "Text rules"): classify(subtask) -> element.
A first guess only: on our own data the frame read overrides it (annotation_principles.md section 3).
"""

import re

CLOTH = r"\b(sock|shirt|t-shirt|cloth|towel|rag|tissue|napkin|wrapper|shorts|sheet|bag|packet|rope|paper towel|tea bag|cable|plastic sheet|sweater|cloths)\b|\bpaper$"
RIM = r"\b(cup|mug|bowl|tape roll|lid|tin|jar|plate|glass|ring|disk)\b"
HANDLE = r"\b(handle|knob|watering can|drawer|fridge|door|portafilter|faucet|pitcher|jug|kettle|pot|pan|tap|lever|toaster)\b"
SIDE = r"\b(bottle|can|cans|spray|motor|book|sanitizer|cylinder|striker)\b"
PLACEHOLDER = r"\b(object|item|ingredient|piece|pieces|thing)\b"
DROP_TARGET = r"\b(basket|bin|box|trash|trash can|cup|pan|bowl|container|drawer|bucket|jar|pot|zone)\b"
INSERT_TARGET = r"\b(slot|shredder|vase|socket|board|rack|group head|peg|stack|fixture|channel|hole)\b"


def classify(sub):
    """Return (element, image_needed, rule). Elements: see vocab.md."""
    s = sub.lower().strip()
    v = s.split()[0]
    if s.startswith("turn on") or s.startswith("turn off"): v = "turn"
    obj = re.sub(r"^\w+( on| off)? (the |a |an |one |next |final )?", "", s)
    if v in ("move", "return"): return "na", False, "carry / free motion"
    if v == "grasp":
        if re.search(PLACEHOLDER, obj): return "top-pinch", True, "placeholder noun: image sets the mode"
        if re.search(CLOTH, obj): return "cloth-pinch", False, "deformable / sheet noun"
        if re.search(HANDLE, obj): return "handle-grasp", False, "handle noun"
        if re.search(RIM, obj): return "rim-pinch", True, "open container / ring noun; image: body wrap or handle instead?"
        if re.search(SIDE, obj): return "side-pinch", True, "tall body noun; image: top-of-cap instead?"
        return "top-pinch", False, "default rigid object"
    if v == "release":
        if re.search(INSERT_TARGET, s): return "insert", True, "fitted target; image: hung / slotted vs set on top"
        if re.search(r"\bon the\b|\bbeside\b|\bat the side\b|\bonto\b", s): return "set-down", False, "open surface target"
        if re.search(DROP_TARGET, s): return "drop", True, "container target; image: lowered to rest = set-down"
        return "drop", True, "unrecognised target"
    if v in ("push",):
        if re.search(r"handle", s): return "handle-grasp", True, "push via a handle; image: pads closed on it?"
        return "push", False, "non-prehensile push"
    if v in ("close",): return "push", False, "close = push the panel"
    if v in ("knock", "hit"): return "strike", False, "impulsive contact"
    if v == "press":
        if re.search(r"\binto\b", s): return "insert", False, "press X into Y = seat a held object"
        return "press", False, "control face"
    if v == "turn":
        if re.search(r"knob|tap|faucet", s): return "handle-grasp", True, "knob: pinch and twist"
        return "press", False, "switch / lamp"
    if v == "rotate":
        if re.search(PLACEHOLDER, obj): return "na", False, "in-hand reorientation of a held object (FMB)"
        if re.search(HANDLE, obj): return "handle-grasp", False, "knob / faucet"
        if re.search(RIM, obj): return "rim-pinch", True, "container turned on the table; image"
        return "side-pinch", True, "body turned on the table; image"
    if v in ("pour", "water", "tilt", "dump"): return "tilt-pour", False, "held container tilted"
    if v in ("wipe", "scrub", "stir", "brush", "sweep"): return "tool-drag", False, "held tool on a surface"
    if v in ("insert", "plug"): return "insert", False, "held object into a fit"
    if v == "stack":
        if re.search(r"peg|base", s): return "insert", False, "ring over a peg"
        return "set-down", False, "stack on top"
    if v in ("fold", "unfold", "straighten", "spread", "flatten"): return "cloth-pinch", True, "cloth verb; image: open-gripper drag = push"
    if v == "pull":
        if re.search(HANDLE, s) or re.search(RIM, s): return "handle-grasp", False, "pull on a handle / lid tab"
        return "cloth-pinch", False, "pull a sheet / rope"
    if v == "open":
        if re.search(r"tongues|tongs", s): return "na", False, "held tool actuated in hand"
        return "handle-grasp", True, "hook or pinch the handle / edge; image"
    if v == "lift":
        if re.search(r"box lid", s): return "pry", False, "closed gripper wedged under the flap"
        if re.search(PLACEHOLDER, obj) or re.search(r"tongues", s): return "na", False, "carried"
        if re.search(CLOTH, obj): return "cloth-pinch", False, "lift the towel = pinch"
        return "top-pinch", True, "lift a lid: pinch or pry; image"
    if v == "hold": return "handle-grasp", False, "sustained pinch"
    if v == "place":
        if re.search(r"fixture|peg|board", s): return "insert", False, "fixture fit"
        return "set-down", False, "place on a surface"
    if v == "shake": return "side-pinch", False, "wrap the hand"
    if v == "plug": return "insert", False, ""
    return "UNMAPPED", True, f"verb {v}"
