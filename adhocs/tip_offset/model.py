"""The tip-offset adapter now lives in the policy package (it is co-trained with the policy); the standalone trial
(train.py) uses the same class at its default size."""
from lerobot.policies.molmoact2.tip_target import GRID, Adapter, to_cells, to_patches  # noqa: F401
