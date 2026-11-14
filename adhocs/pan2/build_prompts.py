"""Write prompts_robot.json for run_pan2.py: two image-to-video records, 10 s each.

Captions follow the PAN 2 rewriter policy (rewriter/system_prompt.md on the Spark); the negative
is the repo's fixed template for 10 s, taken from rewriter/examples.json.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

STYLE = "Realistic fixed overhead camera footage of a tabletop robot manipulation task, one uninterrupted continuous shot at real-time speed."
LIGHTING = "Even, cool indoor daylight from above and from a bright window strip along the bottom edge; soft short shadows under the arm and objects, matte wood grain, no strong highlights."
VIEW = "Static top-down view from a camera fixed above the table, looking straight down with slight wide-angle perspective. The whole light wood tabletop fills the frame; the robot base sits at the top edge."
ROBOT = {
    "subject_id": "<|Subject1|>",
    "display_name": "Robot arm",
    "subject_attribute": "Slim single robot arm with lime-green link covers printed with black lettering, black servo joints and silver brackets, fixed on a black square base plate at the top edge of the table; a black two-finger parallel gripper with lime-green finger pads at its tip and a thin black cable running back to the base.",
}
QUALITY = {
    "video_quality": "Sharp webcam-like footage, rigid robot links and objects keeping their shape, stable table and background, slight motion blur only on the moving arm.",
    "text_logos_overlays": None,
    "artifacts": None,
}
STATIC = ["static"]

lid = {
    "video_style": STYLE,
    "lighting": LIGHTING,
    "perspective_view": VIEW,
    "subjects_lifecycle": [
        ROBOT,
        {
            "subject_id": "<|Subject2|>",
            "display_name": "Headphone case lid",
            "subject_attribute": "Rigid black teardrop-shaped lid shell of a zippered headphone case, dark felt lined, joined by a fabric hinge to the matching base shell that holds black over-ear headphones.",
        },
    ],
    "init_frame_visual_description": "A light wood-laminate table seen from directly above. In the center a black headphone case lies fully open like a clamshell: its empty felt-lined lid shell on the left, its base shell with folded black over-ear headphones on the right, the two joined along a vertical hinge. The robot arm reaches down from its black base plate at the top center; its gripper is at the upper left rim of the lid shell, one finger inside the shell and one outside the wall. Left of the case sit a roll of gray duct tape, a white paper cup with green print near the top left corner and a small white pill bottle with a yellow label near the bottom left. A dark wicker basket holding a few items stands at the right edge, a clear organizer box with colored parts and a few small colored plugs lie at the top right. Dark wood flooring surrounds the table; a white surface and a black handle cross the bottom edge.",
    "events": [
        {
            "event_index": 0,
            "time_range_sec": "0 to 1.5",
            "event_description": "The gripper closes on the left rim of the lid shell, pinching its thin wall between the two finger pads. The lid stays flat on the table and every other object stays still.",
            "camera_motion": [{"time_range_sec": "0 to 1.5", "camera_motion": STATIC}],
            "contained_actions": [
                {"time_range_sec": "0 to 1.5", "action": "<|Subject1|> closes its gripper on the left rim of <|Subject2|>, one finger inside the shell and one outside."}
            ],
        },
        {
            "event_index": 1,
            "time_range_sec": "1.5 to 7",
            "event_description": "Holding the rim, the arm lifts and sweeps to the right in an arc. The lid shell pivots about the hinge it shares with the base shell: its left edge rises off the table, the shell stands upright on edge above the hinge, seen from above as a narrow dark sliver, then tips over to the right and comes down on top of the headphones. The base shell and headphones do not slide. The lid ends flat and aligned on the base, so the case is one closed black teardrop pod.",
            "camera_motion": [{"time_range_sec": "1.5 to 7", "camera_motion": STATIC}],
            "contained_actions": [
                {"time_range_sec": "1.5 to 4.5", "action": "<|Subject1|> raises the pinched rim and moves right, swinging <|Subject2|> up about its hinge until it stands vertical."},
                {"time_range_sec": "4.5 to 7", "action": "<|Subject1|> carries the rim over and down to the right, lowering <|Subject2|> onto the base shell until the case is shut."},
            ],
        },
        {
            "event_index": 2,
            "time_range_sec": "7 to 10",
            "event_description": "The gripper opens and lets go of the rim. The closed case stays shut and still on the table, its smooth outer shell facing up. The arm rises and draws back toward its base at the top of the frame, leaving the case fully visible.",
            "camera_motion": [{"time_range_sec": "7 to 10", "camera_motion": STATIC}],
            "contained_actions": [
                {"time_range_sec": "7 to 8", "action": "<|Subject1|> opens its gripper and releases <|Subject2|>, which stays closed on the base."},
                {"time_range_sec": "8 to 10", "action": "<|Subject1|> lifts away and retracts toward its base plate."},
            ],
        },
    ],
    "quality": QUALITY,
}

tape = {
    "video_style": STYLE,
    "lighting": LIGHTING,
    "perspective_view": VIEW,
    "subjects_lifecycle": [
        ROBOT,
        {
            "subject_id": "<|Subject2|>",
            "display_name": "Duct tape roll",
            "subject_attribute": "Rigid roll of gray duct tape about the width of a palm, with a brown cardboard core and a round open hole, lying flat.",
        },
    ],
    "init_frame_visual_description": "A light wood-laminate table seen from directly above. The robot arm stretches from its black base plate at the top center diagonally down to the left edge of the table, where its gripper is lowered over a roll of gray duct tape lying flat: one finger sits inside the roll's hole and the other outside its wall. A closed black teardrop-shaped headphone case rests right of center. A dark wicker basket with an open top stands at the right edge of the table, holding a white cup and a few bottles and leaving free space inside. A clear organizer box with colored parts and a few small colored plugs lie at the top right. The rest of the tabletop is empty. Dark wood flooring surrounds the table; a white surface and a black handle cross the bottom edge.",
    "events": [
        {
            "event_index": 0,
            "time_range_sec": "0 to 1.5",
            "event_description": "The gripper closes on the wall of the tape roll, one finger pad pressing from inside the hole and one from outside. The roll stays flat on the table until the grip is firm.",
            "camera_motion": [{"time_range_sec": "0 to 1.5", "camera_motion": STATIC}],
            "contained_actions": [
                {"time_range_sec": "0 to 1.5", "action": "<|Subject1|> closes its gripper on the wall of <|Subject2|>, one finger inside the hole and one outside."}
            ],
        },
        {
            "event_index": 1,
            "time_range_sec": "1.5 to 8",
            "event_description": "The arm lifts the roll straight off the table; the roll hangs level from the fingers and its shadow separates from it on the wood. The arm then swings to the right, carrying the roll across the tabletop, over the closed headphone case, which does not move, and on to the wicker basket. The arm links fold and turn about the fixed base as the gripper travels. The roll arrives above the open top of the basket, still held.",
            "camera_motion": [{"time_range_sec": "1.5 to 8", "camera_motion": STATIC}],
            "contained_actions": [
                {"time_range_sec": "1.5 to 3", "action": "<|Subject1|> lifts <|Subject2|> off the table, holding it level."},
                {"time_range_sec": "3 to 8", "action": "<|Subject1|> carries <|Subject2|> to the right across the table, over the headphone case, to above the basket."},
            ],
        },
        {
            "event_index": 2,
            "time_range_sec": "8 to 10",
            "event_description": "Over the basket the gripper opens. The roll drops a short distance into the basket and settles among the items inside, partly hidden by the wicker wall. The arm stays above the basket with open fingers.",
            "camera_motion": [{"time_range_sec": "8 to 10", "camera_motion": STATIC}],
            "contained_actions": [
                {"time_range_sec": "8 to 9", "action": "<|Subject1|> opens its gripper and releases <|Subject2|>, which falls into the basket and comes to rest."},
                {"time_range_sec": "9 to 10", "action": "<|Subject1|> holds its open gripper above the basket."},
            ],
        },
    ],
    "quality": QUALITY,
}

examples = json.loads((HERE / "examples.json").read_text())
negative = examples[0]["negative_prompt"]
assert negative.count("0 to 10") == 3

records = []
for name, caption, task in [
    ("lid", lid, "close the headphone case lid"),
    ("tape", tape, "pick up the tape roll and put it in the basket"),
]:
    records.append(
        {
            "name": f"{name}_pan2",
            "original_prompt": f"The robot arm does this task: {task}.",
            "prompt": json.dumps(caption, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            "negative_prompt": negative,
            "image": f"inputs/{name}_frame0.png",
            "seconds": 10,
            "height": 480,
            "width": 640,
        }
    )
(HERE / "prompts_robot.json").write_text(json.dumps(records, indent=1))

# Wrist camera: the camera rides on the gripper, so the fingertips stay fixed in the frame and the scene moves.
W_STYLE = "Realistic first-person footage from a small camera mounted on a robot gripper during a tabletop manipulation task, one uninterrupted continuous shot at real-time speed."
W_LIGHTING = "Soft cool indoor daylight, slightly bluish white balance, warm light wood grain, gentle shadows, no strong highlights."
W_VIEW = "Close wide-angle view from a camera fixed to the robot's wrist, looking down past the gripper at the table. The two black fingertips of the gripper stay at the bottom center of the frame for the whole shot because the camera moves rigidly with them; everything else shifts as the arm moves."
W_GRIPPER = {
    "subject_id": "<|Subject1|>",
    "display_name": "Gripper",
    "subject_attribute": "Two matte black wedge-shaped fingertips of a parallel robot gripper, seen from behind and above at the bottom center of the frame, rigidly attached to the camera.",
}
W_QUALITY = {
    "video_quality": "Sharp close-range webcam-like footage, rigid objects keeping their shape and texture, mild motion blur only while the camera moves.",
    "text_logos_overlays": None,
    "artifacts": None,
}

wlid = {
    "video_style": W_STYLE,
    "lighting": W_LIGHTING,
    "perspective_view": W_VIEW,
    "subjects_lifecycle": [
        W_GRIPPER,
        {
            "subject_id": "<|Subject2|>",
            "display_name": "Headphone case lid",
            "subject_attribute": "Rigid teardrop-shaped lid shell of a zippered headphone case: gray-blue molded felt lining inside, dark charcoal woven fabric outside, a black zipper band around the rim. A fabric hinge joins it to the matching base shell, which holds folded black over-ear headphones.",
        },
    ],
    "init_frame_visual_description": "A very close view down onto a light wood table. The left half of the frame is filled by the inside of an open headphone case lid lying flat: gray-blue felt lining with soft molded bumps, bounded by a curved black zipper rim that runs from the top left down to the bottom center. At the bottom center the two black gripper fingertips straddle that rim, one on the felt side and one on the table side. The right half of the frame is bare wood-grain tabletop. The base shell of the case, its headphones and the rest of the room are outside the frame.",
    "events": [
        {
            "event_index": 0,
            "time_range_sec": "0 to 2",
            "event_description": "The fingertips squeeze the lid rim between them and hold it. The lid stays flat on the table and the view barely changes.",
            "camera_motion": [{"time_range_sec": "0 to 2", "camera_motion": STATIC}],
            "contained_actions": [
                {"time_range_sec": "0 to 2", "action": "<|Subject1|> pinches the rim of <|Subject2|> and holds it firmly."}
            ],
        },
        {
            "event_index": 1,
            "time_range_sec": "2 to 6",
            "event_description": "The gripper lifts the pinched rim and the lid swings up about its hinge. The camera rises and tilts with the gripper, so the rim stays pinned at the bottom center while the table drops away. The felt interior turns edge-on and narrows. To the left the base shell comes into view, lying on the table with black over-ear headphones folded inside it. As the lid passes upright its dark charcoal fabric outside turns toward the camera. Farther back a dark wicker basket appears at the top left of the frame and a pale wall rises behind the far table edge.",
            "camera_motion": [{"time_range_sec": "2 to 6", "camera_motion": ["dolly_up", "tilt_up", "pan_left"]}],
            "contained_actions": [
                {"time_range_sec": "2 to 4", "action": "<|Subject1|> raises the rim of <|Subject2|>, swinging it up off the table about its hinge."},
                {"time_range_sec": "4 to 6", "action": "<|Subject1|> carries <|Subject2|> through upright, its fabric outside now facing the camera."},
            ],
        },
        {
            "event_index": 2,
            "time_range_sec": "6 to 10",
            "event_description": "The gripper carries the rim over and down toward the base shell. The lid's charcoal fabric outside grows to fill the center of the frame and covers the headphones from right to left; a last strip of the base shell with its zipper stays visible at the left, then disappears as the lid seats. The case ends as one closed dark fabric pod on the wood table directly in front of the fingertips, with the wicker basket behind it at the top left.",
            "camera_motion": [{"time_range_sec": "6 to 10", "camera_motion": ["dolly_down", "tilt_down"]}],
            "contained_actions": [
                {"time_range_sec": "6 to 9", "action": "<|Subject1|> lowers <|Subject2|> onto the base shell, covering the headphones."},
                {"time_range_sec": "9 to 10", "action": "<|Subject1|> holds <|Subject2|> shut on the base shell."},
            ],
        },
    ],
    "quality": W_QUALITY,
}

wcup = {
    "video_style": W_STYLE,
    "lighting": W_LIGHTING,
    "perspective_view": W_VIEW,
    "subjects_lifecycle": [
        W_GRIPPER,
        {
            "subject_id": "<|Subject2|>",
            "display_name": "Paper cup",
            "subject_attribute": "Empty white paper cup standing upright, glossy white inside with a rolled rim, its outer wall printed in green and orange, light and rigid.",
        },
    ],
    "init_frame_visual_description": "A view from above and behind an open robot gripper. A light wood table fills the left and center of the frame, its far edge running diagonally across the top. An empty white paper cup stands near the right table edge, seen from above so that its round open mouth and glossy white inside are visible. At the bottom center the two black gripper fingertips are spread apart, hovering just short of the cup's near rim. Beyond the table edge lies gray wood-plank flooring with a black robot vacuum on its dock against a white baseboard, a black power cable, a dark speaker at the top right and thin black chair legs at the right edge.",
    "events": [
        {
            "event_index": 0,
            "time_range_sec": "0 to 2.5",
            "event_description": "The gripper moves forward and to the right toward the cup, and the camera moves with it: the cup slides down and toward the bottom right of the frame and grows larger. One fingertip passes over the rim into the cup's mouth while the other stays outside the wall. The cup does not move.",
            "camera_motion": [{"time_range_sec": "0 to 2.5", "camera_motion": ["dolly_forward", "track_right"]}],
            "contained_actions": [
                {"time_range_sec": "0 to 2.5", "action": "<|Subject1|> advances onto the near rim of <|Subject2|>, one fingertip inside the cup and one outside."}
            ],
        },
        {
            "event_index": 1,
            "time_range_sec": "2.5 to 5",
            "event_description": "The fingertips close and pinch the cup's wall just below the rim. The thin paper wall flexes slightly under the grip and the cup shifts a little on the table as it is caught, then is held firmly. The printed green and orange outside of the cup shows at the left of the fingertips.",
            "camera_motion": [{"time_range_sec": "2.5 to 5", "camera_motion": STATIC}],
            "contained_actions": [
                {"time_range_sec": "2.5 to 5", "action": "<|Subject1|> closes on the wall of <|Subject2|> and grips it below the rim."}
            ],
        },
        {
            "event_index": 2,
            "time_range_sec": "5 to 10",
            "event_description": "The gripper lifts the cup off the table and starts to carry it. The cup stays fixed between the fingertips at the bottom right of the frame, its white mouth toward the camera, while the tabletop sinks and slides beneath it. The table's rounded near corner and edge come into view, and more of the gray plank floor appears beyond it, including a flat brown cardboard box with a white shipping label lying on the floor.",
            "camera_motion": [{"time_range_sec": "5 to 10", "camera_motion": ["dolly_up", "pan_left"]}],
            "contained_actions": [
                {"time_range_sec": "5 to 7", "action": "<|Subject1|> lifts <|Subject2|> clear of the table."},
                {"time_range_sec": "7 to 10", "action": "<|Subject1|> carries <|Subject2|> steadily above the table, holding it by the wall."},
            ],
        },
    ],
    "quality": W_QUALITY,
}

# Concatenated views: top camera on the left half, wrist camera on the right half, 384x288 each.
C_STYLE = "A single split-screen video: one frame divided into a left panel and a right panel that play together for the whole duration, showing one tabletop robot manipulation task recorded by two synchronized cameras. Realistic footage, one uninterrupted continuous shot at real-time speed. The split-screen layout never changes: the vertical seam stays fixed in the middle of the frame, neither panel cuts, fades, swaps or takes over the full frame."
C_VIEW = "Two panels of equal size showing the same moment. Left panel: static top-down view from a camera fixed above the table, the whole light wood tabletop in frame, the robot base at the top edge. Right panel: close wide-angle view from a camera fixed to the robot's wrist, looking down past the gripper, whose two black fingertips stay at the bottom center of the panel while the scene shifts as the arm moves. Every motion in the left panel has its matching close-up in the right panel."
C_ROBOT = dict(ROBOT, subject_attribute=ROBOT["subject_attribute"] + " In the right panel only its two matte black wedge-shaped fingertips are visible, from behind.")
C_QUALITY = dict(W_QUALITY, video_quality="Sharp webcam-like footage in both panels, rigid robot links and objects keeping their shape, the two panels consistent with each other, mild motion blur only on moving parts.")
C_CAM = ["static"]

catlid = {
    "video_style": C_STYLE,
    "lighting": LIGHTING,
    "perspective_view": C_VIEW,
    "subjects_lifecycle": [C_ROBOT, wlid["subjects_lifecycle"][1]],
    "init_frame_visual_description": "Left panel: a light wood table from directly above. In the center a headphone case lies fully open like a clamshell, its empty felt-lined lid shell on the left and its base shell with folded black over-ear headphones on the right. The robot arm reaches down from its base plate at the top; its gripper is at the upper left rim of the lid shell. A roll of gray duct tape, a white paper cup and a small white pill bottle sit left of the case, a dark wicker basket stands at the right edge, a clear organizer box lies at the top right. Right panel: a very close view of the same grasp. The gray-blue felt inside of the lid fills the left half, bounded by a curved black zipper rim; the two black fingertips straddle that rim at the bottom center; bare wood tabletop fills the right half.",
    "events": [
        {
            "event_index": 0,
            "time_range_sec": "0 to 2",
            "event_description": "The gripper squeezes the lid rim and holds it. In both panels the lid stays flat on the table and nothing else moves.",
            "camera_motion": [{"time_range_sec": "0 to 2", "camera_motion": C_CAM}],
            "contained_actions": [
                {"time_range_sec": "0 to 2", "action": "<|Subject1|> pinches the rim of <|Subject2|> and holds it firmly."}
            ],
        },
        {
            "event_index": 1,
            "time_range_sec": "2 to 6",
            "event_description": "Left panel: the arm lifts the pinched rim and sweeps to the right; the lid shell pivots about the hinge it shares with the base shell, its left edge rises and it stands upright on edge as a narrow dark sliver. The base shell and headphones do not slide. Right panel: the view rises and tilts with the gripper while the rim stays pinned at the bottom center; the felt interior turns edge-on, the base shell with its black headphones comes into view at the left, the lid's dark charcoal fabric outside turns toward the camera, and the wicker basket and a pale wall appear at the top.",
            "camera_motion": [{"time_range_sec": "2 to 6", "camera_motion": C_CAM}],
            "contained_actions": [
                {"time_range_sec": "2 to 4", "action": "<|Subject1|> raises the rim of <|Subject2|>, swinging it up off the table about its hinge."},
                {"time_range_sec": "4 to 6", "action": "<|Subject1|> carries <|Subject2|> through upright above the hinge."},
            ],
        },
        {
            "event_index": 2,
            "time_range_sec": "6 to 10",
            "event_description": "Left panel: the lid tips over to the right and comes down on top of the headphones; the case ends as one closed black teardrop pod with the gripper still on it. Right panel: the charcoal fabric outside of the lid grows to fill the center and covers the headphones; the last strip of base shell at the left disappears as the lid seats, leaving a closed dark fabric pod in front of the fingertips with the wicker basket behind it at the top left.",
            "camera_motion": [{"time_range_sec": "6 to 10", "camera_motion": C_CAM}],
            "contained_actions": [
                {"time_range_sec": "6 to 9", "action": "<|Subject1|> lowers <|Subject2|> onto the base shell, covering the headphones."},
                {"time_range_sec": "9 to 10", "action": "<|Subject1|> holds <|Subject2|> shut on the base shell."},
            ],
        },
    ],
    "quality": C_QUALITY,
}

cattape = {
    "video_style": C_STYLE,
    "lighting": LIGHTING,
    "perspective_view": C_VIEW,
    "subjects_lifecycle": [
        C_ROBOT,
        {
            "subject_id": "<|Subject2|>",
            "display_name": "Duct tape roll",
            "subject_attribute": "Rigid roll of gray duct tape about the width of a palm, with a slightly wavy outer edge, a white printed cardboard core and a round open hole, lying flat.",
        },
    ],
    "init_frame_visual_description": "Left panel: a light wood table from directly above. The robot arm stretches from its base plate at the top center diagonally to the left edge of the table, where its open gripper is lowered over a roll of gray duct tape lying flat. A closed black teardrop-shaped headphone case rests right of center, a dark wicker basket with bottles and tubes inside stands at the right edge, a clear organizer box lies at the top right. Right panel: a very close view of the same grasp from the wrist. Diagonal wood grain fills the panel; the gray tape roll lies at the bottom right, seen from above with the table visible through its hole. One black fingertip is inside the hole and the other is outside the roll's wall at the bottom center.",
    "events": [
        {
            "event_index": 0,
            "time_range_sec": "0 to 2.5",
            "event_description": "The gripper closes on the wall of the tape roll, one fingertip pressing from inside the hole and one from outside. The roll shifts slightly as it is caught, then is held firmly, still flat on the table. Nothing else moves in either panel.",
            "camera_motion": [{"time_range_sec": "0 to 2.5", "camera_motion": C_CAM}],
            "contained_actions": [
                {"time_range_sec": "0 to 2.5", "action": "<|Subject1|> closes its gripper on the wall of <|Subject2|>, one finger inside the hole and one outside."}
            ],
        },
        {
            "event_index": 1,
            "time_range_sec": "2.5 to 6",
            "event_description": "Left panel: the arm lifts the roll off the table and its links fold and turn about the fixed base as the gripper starts to travel right; the roll hangs level from the fingers. Right panel: the roll stays fixed at the bottom right between the fingertips while the tabletop sinks away and slides; the far table edge, a pale wall, a black cable and the foot of a black gooseneck stand come into view at the top left.",
            "camera_motion": [{"time_range_sec": "2.5 to 6", "camera_motion": C_CAM}],
            "contained_actions": [
                {"time_range_sec": "2.5 to 4", "action": "<|Subject1|> lifts <|Subject2|> off the table, holding it level."},
                {"time_range_sec": "4 to 6", "action": "<|Subject1|> swings to the right, carrying <|Subject2|> above the table."},
            ],
        },
        {
            "event_index": 2,
            "time_range_sec": "6 to 10",
            "event_description": "Left panel: the arm carries the roll across the tabletop and over the closed headphone case, which does not move, arriving next to the wicker basket with the roll still held. Right panel: the roll stays at the bottom right; the dark wicker basket enters from the bottom left and grows until its open top is seen from above, with bottles and tubes inside it, against the pale wall.",
            "camera_motion": [{"time_range_sec": "6 to 10", "camera_motion": C_CAM}],
            "contained_actions": [
                {"time_range_sec": "6 to 10", "action": "<|Subject1|> carries <|Subject2|> over the headphone case to the basket, still gripping it."}
            ],
        },
    ],
    "quality": C_QUALITY,
}

records = []
for name, caption, width in [("catlid", catlid, 768), ("cattape", cattape, 768), ("wlid", wlid, 384)]:
    records.append(
        {
            "name": f"{name}_pan2",
            "prompt": json.dumps(caption, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            "negative_prompt": negative,
            "image": f"inputs/{name}_frame0.png",
            "seconds": 10,
            "height": 288,
            "width": width,
        }
    )
(HERE / "prompts_cat.json").write_text(json.dumps(records, indent=1))
