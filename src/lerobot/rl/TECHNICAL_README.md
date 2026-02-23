# Pi05 Technical README

This document is a more in-depth description of the files created for the Pi05 full RL pipeline.


## Key files

### `rl_pi05.py`
This file is the analogous of modeling_pi05.py in pi05_full. It inherits the class Pi05Pytorch from pi05_full and it also introduces the critic, which shares a similar architecture to the actor. The class is called PI05RLPolicy and it overwrites the forward pass taking in consideration the critic. For inference, this class still calls sample_actions() from the Pi05Pytorch class.


### `pi05_train_utils.py`
This file contains heplful functions for training. The reason for this file is that we need to have offline training and online training, which are very similar. So it was easier to condense the code to function that can be used in both cases. The key function here is pi05_update_step, which executes the update. The other functions are for preprocessing and postprocessing as well as logging to wandb.


### `offline_learner_pi05.py` and `learner_pi05.py`
These files are used to train the policy offline and online, respectively. The main difference is that `learner_pi05.py` has an offline buffer and an online buffer, where the former is static, and the latter is being fed more transitions constantly. When training offline, about 80% of the model is trained, unlike online when only 25% is trained. The reason for this is simply memory constraints and speed.

### `actor_pi05.py`
This file is used to run the online policy on the robot. It runs in conjunction with `learner_pi05.py`.


### Other modifications

`utils.py` contains helpful function for batch processing and dtype handling (the model is trained in bf16). Other files that were modifed from the original lerobot implementation are `processor_pi05.py` (to enable the advantage enriched prompt), the `so_leader.py` in the teleoperators, which was modified to add the intervention behavior, and probably other files that escape me now.

