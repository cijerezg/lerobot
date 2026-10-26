# object_coverage

Question (2026-10-26): where on the table, and at what orientation, has the policy seen the nail clippers, the large
clipper, the headphone case, the bit kit and the bits in the internal ReBot data? Rough readout, meant to find empty
regions before recording more data, not a measurement.

## Method

One top-camera frame per grasp segment, at the segment's first frame (the pose the policy has to handle), from the 11
internal roots in `config_rl.yaml`. A table-aligned 6 x 4 grid is drawn on the frame (columns A-F left to right,
rows 1-4 from the robot side to the camera side, cells about 10 x 9 cm) plus a compass. Opus 5.5 vision agents
(medium effort, 14 images each) read cell, long-axis angle (0 = along the long table edge, 90 = toward the robot,
mod 180), a facing direction for the case (hinge side) and the kit (side of the flap with the bits), a state, and the
destination cell, following `RUBRIC.md`. Rows where the gripper hid the object got a second read on a frame 3 s
earlier; that read wins when it found the object.

Segment selection: `grasp the (small) nail clipper`, `grasp the large nail clipper`, first `grasp/close the headphone
case lid` per episode, first flap clause per episode for the kit, every `grasp ... bit` clause.

## Run

```
uv run python lerobot/adhocs/object_coverage/extract_frames.py   # frames/, manifest.csv
# split manifest into batches/, spawn readers per batch -> readouts/<batch>.csv (see RUBRIC.md)
uv run python lerobot/adhocs/object_coverage/aggregate.py        # coverage_<object>.png, qa_<object>.png, coverage.md
```

`coverage_<object>.png`: cell counts, one oriented segment per observation in its cell, angle histogram, facing rose,
state and destination counts. `qa_<object>.png`: every frame with its readout, for spot checks. `coverage.md`: the
angle tables and the empty and thin cells per object. `readouts_merged.csv`: all rows.

## Findings

| object | frames | read | unsure | occupied cells | empty |
|---|---|---|---|---|---|
| nail_clipper | 39 | 38 | 27 | rows 2-3, A-E, peak C3 | row 1 (except B1), row 4 (except D4), column F |
| large_clipper | 21 | 21 | 18 | rows 2-3, A-D, peaks B2 D2 | A2 C2 E2 E3, row 4 (except D4), column F |
| headphone_case | 32 | 32 | 19 | rows 2-3, B-E, peak E2 E3 | row 4 (except B4), A2 A3 C2 F2 |
| bit_kit | 18 | 17 | 7 | B2-E2, B3-D3, peak C3 (8 of 17) | rows 1 and 4, columns A and F |
| bit | 79 | 77 | 52 | rows 2-4, A-F, peaks A2 E2 C3 | row 1 (except E1), A4 E4 F4 |

All four target objects live in the middle band of the table, rows 2-3 and columns B-E. The strip nearest the robot
(row 1), the strip nearest the camera (row 4) and the right edge (column F) are empty for every one of them.

Orientation: the clippers are bimodal, along the table (0) or toward the robot (90), with 1-3 frames per diagonal bin.
The headphone case is 0 or 90 and its hinge faces W in most frames, NW or NE in the rest, never S and rarely E. The
bit kit lies with the bits flap to the E in most frames, W in a few, never N or S; 13 of 17 start flat, 4 with one
flap already folded. The bits are spread over all angles.

Caveats: readers marked most clipper rows unsure (small object, two clippers hard to tell apart, gripper often over the
case at the segment start); 6 nail clipper and 5 bit rows stayed occluded after the second read. Cell resolution is
one grid cell. The readout says where the object is in the image, which is what the policy sees.

What to record to fill the gaps: objects in row 1 and row 4 and in columns A and F; clippers at diagonal angles
(30, 60, 120, 150); the headphone case with the hinge toward the camera (S) and toward the right (E); the bit kit with
the bits flap to the W, N or S.
