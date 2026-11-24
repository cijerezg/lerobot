# vision_swap

Question: the internal checkpoint (`rebot-internal-only-20261021`, step 1200) keeps working when its image tokens
are replaced by those of the base model's vision path, although the two sets of tokens differ a lot (cosine per
token about 0.48). How much does the output change, and where in the network does the difference go?

Everything here is open loop on recorded validation frames (`outputs/data/rebot_2026-10-14-val-v1`). All commands
run from the workspace root.

## Layer by layer page

    uv run python lerobot/adhocs/vision_swap/layers.py --config_path=config_rl_validate.yaml

then open http://localhost:8772 (start-up about 2 minutes; 40 frames, `--frames_per_episode` changes that).
Two inputs A and B at the image positions of one model are compared through the 36 language-model blocks, the
keys and values the action expert reads, the 36 expert blocks and the output. An input is
`own tokens + fraction x (source tokens - own tokens)` on all image positions, on the long image tokens only
(own token longer than 10 x the frame's median) or on the others. Sources: own, base vision path, camera images
1 s later, camera images of another episode, mean token over frames, random direction of the length of
(base - own). The module docstring defines what is kept per run; the page defines every measure.

| Tab | Shows |
|---|---|
| Layers | one line per kind of position (ordinary / long image token, depth, state, text clauses, end of prompt) over the language-model layers, and the expert's action positions over its blocks; next to it the same for the camera images 1 s later and of another episode |
| Map | one measure at every layer and position of one frame; the image positions as a 14 x 14 grid over the camera pictures |
| Expert read | share of the expert's cross-attention per kind of position; change of the keys / values weighted by that attention; change of the cross-attention output |
| Output | tip distance (cm), KL of the action-token distributions, cross-entropy change, tip paths |
| Sweep | output and last-layer change against the fraction of the change |

A run over one frame takes about 3 s, over all 40 frames about 2 minutes.

First numbers (40 frames, base tokens at all image positions against own tokens; read off the page):

| | input B = base | camera images 1 s later | another episode |
|---|---|---|---|
| tip distance, same flow seed (cm); two flow seeds: 2.20 | 1.48 | 1.81 | 5.11 |
| KL of the action-token distributions | 0.080 | 0.123 | 0.516 |

Distance of base against own, divided by the distance of the camera images 1 s later: ordinary image positions
about 1.8 at the input of the language model and about 1.5 at its last block; state and end-of-prompt positions
about 1.1 to 1.2 from block 8 to the last; expert block output about 2.4 at block 1 and about 1.0 from block 17 on.
The two inputs do not come together at any layer; from the middle of the language model on, the difference at the
positions that write the action is the size of a 1 s change of the picture.

Measured through the page's endpoints (`/compare`, `/map`, `/tensor`; files in
`outputs/analysis/vision_swap_2026-11-19/explore`):

| input B, 40 frames | tip distance (cm) | KL |
|---|---|---|
| base, all image positions | 1.48 | 0.080 |
| base, the long image tokens only | 0.03 | 0.000 |
| base, the other image tokens only | 1.48 | 0.080 |
| random direction of the length of (base - own), the other image tokens only | 0.36 | 0.010 |
| mean token over frames, all image positions | 4.52 | 0.357 |

Block 17 of the language model (8 frames): at 20 to 40 of the ordinary image positions per frame it writes one
fixed vector (cosine 1.000 between positions, frames, and own / base runs; length about 7,000 against 100 for
the other image positions), which stays to the last block. The positions are patches without content (wall,
table top, picture margins). With base tokens about as many positions get it, but other ones (2 to 7 shared
out of about 25; 10 to 20 shared with the camera images 1 s later). The expert gives such a position about 6
times the attention of another image position in blocks 18 to 36 (about 11 % of its attention in total).
Consequences: the per-position distance at these positions is in the thousands without meaning anything, and
from block 17 on "distance / spread" and "cosine, mean removed" at image positions are distorted, because the
mean m over frames contains a share of the fixed vector. In the expert the difference base against own drops
from 1.5 times that of 1 s later (block 16) to 1.13 (block 17) and 1.05 (block 19).

## Discrete against continuous output

    uv run python lerobot/adhocs/vision_swap/discrete_swap.py --config_path=config_rl_validate.yaml --out=<new .pt>
    uv run python lerobot/adhocs/vision_swap/discrete_swap_report.py <that .pt>

160 frames, 4 flow seeds (`outputs/analysis/vision_swap_2026-11-19/discrete_swap_160.md`):

| | base | random direction, same length | 1 s later | another episode |
|---|---|---|---|---|
| continuous chunk, tip distance (cm) | 1.62 | 0.45 | 1.83 | 5.87 |
| discrete chunk (greedy), tip distance (cm) | 2.02 | 0.95 | 2.18 | 6.31 |
| token cross-entropy, change per frame | 0.082 | 0.025 | 0.102 | 0.313 |

The discrete output is as insensitive to the swap as the continuous one. The "random" column is not a fair
control: the length of (base - own) is dominated by about 7 very long image tokens (length about 150,000 against
76 for the others), so most of the random change goes where the model does not look.

## Gradient split

    uv run python lerobot/adhocs/vision_swap/gradient_split.py --config_path=config_rl_validate.yaml --out=<new .pt>
    uv run python lerobot/adhocs/vision_swap/gradient_split_report.py <that .pt>

Splits (base - own) into the part inside the 48 strongest gradient directions of the action-token
log-probabilities and the rest (40 frames, `gradient_split_per_frame.md`). KL of the full swap 0.080, of the part
inside 0.043, of the rest 0.086. Not conclusive: the gradient describes the model only for a few percent of the
step (error 0.11 at 1 % of the step, 1.06 at the full step), and the long tokens distort the split.
