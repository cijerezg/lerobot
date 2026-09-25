"""Contact-strategy vocabulary: the per-subtask ``contact`` metadata channel.

Definitions of record: ``pi07_wiki/contact_strategy_rubric.md``. ``code`` is the integer
stored per frame (buffer column ``metadata_contact``); ``phrase`` fills the prompt clause
``The contact is <phrase>.``. ``-1`` means no label covers the frame: the clause is omitted.
"""

from __future__ import annotations

from typing import NamedTuple

CONTACT_VOCAB_VERSION = "contact_vocab_v1_2026-09-24"
UNLABELLED = -1


class ContactElement(NamedTuple):
    slug: str
    code: int
    phrase: str
    definition: str


CONTACT_VOCAB: tuple[ContactElement, ...] = (
    ContactElement(
        "top-pinch", 0, "a top pinch", "pads close from above on a rigid object's sides or top, approach < 45"
    ),
    ContactElement(
        "side-pinch",
        1,
        "a side pinch",
        "pads close on the body from the side, approach >= 45 (wrap on a cylinder, face pinch on a book)",
    ),
    ContactElement("rim-pinch", 2, "a rim pinch", "one pad inside a mouth or ring, one outside its wall, from above"),
    ContactElement(
        "handle-grasp",
        3,
        "a handle grasp",
        "pads close on a handle, bar, knob, tab or lever attached to a larger body",
    ),
    ContactElement(
        "cloth-pinch",
        4,
        "a cloth pinch",
        "pads close on a fold, edge, corner or bunch of a deformable or thin sheet",
    ),
    ContactElement(
        "push", 5, "a push", "sustained lateral contact; the object slides on the surface or swings on a hinge"
    ),
    ContactElement(
        "press", 6, "a press", "contact along the normal on a control face; the object does not translate"
    ),
    ContactElement(
        "strike", 7, "a strike", "impulsive contact by the gripper or a held tool; the gripper keeps moving"
    ),
    ContactElement("pry", 8, "a pry", "closed gripper wedged under an edge and lifted"),
    ContactElement(
        "set-down", 9, "a set-down", "held object lowered until it rests on a surface, then released"
    ),
    ContactElement("drop", 10, "a drop", "pads open while the object is unsupported over the target"),
    ContactElement(
        "insert",
        11,
        "an insertion",
        "held object guided into a fitted opening or over a peg; the fit does the last centimetres",
    ),
    ContactElement("tilt-pour", 12, "a pour", "held container rotated so its contents leave it"),
    ContactElement("tool-drag", 13, "a tool drag", "held object pressed on a surface and moved across it"),
    ContactElement("na", 14, "not applicable", "no contact made, broken or worked"),
)

NA_CODE = 14
_BY_CODE = {e.code: e for e in CONTACT_VOCAB}
_BY_SLUG = {e.slug: e for e in CONTACT_VOCAB}


def phrase_for(code: int) -> str:
    """Clause phrase for a stored code; raises on -1 or anything outside 0-14."""
    try:
        return _BY_CODE[int(code)].phrase
    except KeyError:
        raise ValueError(f"contact code {code!r} is not in the vocabulary (0-14)") from None


def code_for(slug: str) -> int:
    """Stored code for an element slug (``na`` -> 14)."""
    try:
        return _BY_SLUG[slug].code
    except KeyError:
        raise ValueError(f"contact slug {slug!r} is not in the vocabulary") from None
