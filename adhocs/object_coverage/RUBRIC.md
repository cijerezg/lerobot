# Object pose readout rubric

Rough readout of where an object lies on the table at the start of a grasp segment. Resolution is one grid cell and
15 degrees. Do not measure; estimate by eye and move on. Every image is the fixed top camera with a yellow grid drawn
on the table: columns A-F left to right, rows 1-4 from the robot side (top of the image) to the camera side (bottom).
The compass at the bottom left defines the angle: 0 = parallel to the long table edge (image horizontal),
90 = pointing toward the robot (image up). Angles are in [0, 180) because a long axis has no front.

Fields, one row per image, written as CSV with a header:

image,cell,angle,facing,state,extra_cell,confidence,note

- `cell`: grid cell containing the object's center, e.g. `C3`. If the center sits on a line, pick the cell holding most of the object.
- `angle`: long-axis angle of the object, multiple of 15, 0-165. Round objects: leave empty.
- `facing`: compass direction (N = toward robot, E = image right, S = toward camera, W = image left; also NE, SE, SW, NW) of the object's reference feature, defined per object below. Leave empty if the object has no reference feature.
- `state`: per object below.
- `extra_cell`: cell of the destination object named per object below.
- `confidence`: `sure` or `unsure`.
- `note`: short; say `occluded` if the gripper hides the object, `not found` if you cannot find it.

Per object:

- `nail_clipper` (the small nail clipper, a few cm long, metallic, usually lying flat): cell, angle. facing: direction of the lever end if visible, else empty. state: empty. extra_cell: the clipper case (a small flat box with two slots that the clippers go into).
- `large_clipper` (the larger nail clipper): same as nail_clipper.
- `headphone_case` (the rounded grey/black case with a hinged lid, lid open or partly open; there may be a second closed case or loose headphones, ignore those): cell = center of the case body. angle: of the case's long axis. facing: direction of the hinge, i.e. the side the open lid is attached to, seen from the case body. state: `open` (lid lying flat or more than halfway open), `half` (lid standing roughly upright), `closed`. extra_cell: empty.
- `bit_kit` (the blue-and-black fabric kit with bit holders; two outer flaps fold over the middle): cell = center of the whole kit as it lies. angle: of the kit's long axis. facing: direction, from the kit's center, of the flap that carries the bits (the flap with the rows of bits, not the white card). state: `flat` (both flaps open), `one_folded` (one flap already folded), `closed`. extra_cell: empty.
- `bit` (a small driver bit, a dark 2-3 cm rod): the clause in the batch file names which bit (e.g. `grasp the bit nearest the box`); when it is just `grasp the bit`, pick the loose bit on the table nearest the gripper, or the only loose bit. cell, angle. facing: empty. state: number of loose bits on the table in total, as an integer. extra_cell: the box the bits go into (the clear compartment box, or the bit kit if the clause says so).
