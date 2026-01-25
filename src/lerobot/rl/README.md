# Pi05 RL Implementation

This README refers mainly to `actor_pi05.py`, `learner_pi05.py`, `rl_pi05.py`, and `offline_learner_pi05.py`, which are the key files for the Pi05 RL implementation. However, it is worth noting that other files were also modified or created such `rl/utils.py`, or teleoperation inside the `teleoperators` folder.


## Overview

This algorithm uses an actor-critic architecture. The actor is the Pi05 model in lerobot version 0.4.2 (no FAST tokens, no subtasks), and the critic shares a very similar architecture without the action expert. The critic learns to predict the value of a state, `V(s)`, and then it can be used to calculate the advantage of actions, as `A(s, a) = r(s,a) + V(s+1) - V(s)`. Intuitively, the advantage measures the return of action `a` relative to the expected value in state `s`. Good actions will have a high advantage, while bad actions will have a low advantage.

Since Pi05 is a VLA model, the output is not the action itself, but a velocity field $v_t$, therefore we cannot take the advantage and differentiate with respect to the actor weights. Instead, we condition the actor on the advantage, i.e., $\pi(v_t | s, a, A)$, and then we can use the traditional flow matching loss (Mean Squared on $v_t$). As a consequence, the policy can now discern good actions from bad actions, and when on inference, we fix the advantage to a high value.



## Code walkthrough and functionality

The policy object is defined in `rl_pi05.py`, where the critic architecture is also defined. As mentioned, the critic uses the same base architecture as Pi05, but with fewer layers. Moreover, the vision features are shared between the actor and the critic, although the critic gradient does not propagate to the vision tower. For the actor, the only difference with Pi05 is that now the prompt also contains the advantage. See line 132 in `policies/pi05/processor_pi05.py` for the new prompt. Note that the advantage is discretized into 5 bins.

`actor_pi05.py` and `learner_pi05.py` serve the same purposes as `actor.py` and `learner.py` in the original lerobot repo, but additional functionality was added.

For `actor_pi05.py`, one key difference is that the interventions are only done with the leader arm, specifically, when the policy starts running, the leader will follow the follower. To intervene, press `5`, and then the follower will follow the leader (normal teleoperation). To stop the intervention, press `5` again. Interventions are assumed to be good actions, so they automatically recieve a high advantage regardless of the critic output. Finally, the reward can be entered by pressing `1` if sucess, or `0` if fail. Then, to start the new episode you need to press `2`.

For `learner_pi05.py`, the difference is that it computes the losses as described in the overview. It also uses gradient accumulation, which helps with VRAM usage; if you have plenty of VRAM, just set `gradient_accumulation_steps = 1` in the config and set a larger batch size. It also logs several quantities to WandB, and creates a video of the episode with the corresponding critic value every N steps as specified in the variable at the top of the file `episode_logging_freq = 2`. Furthermore, `learner_pi05.py` will also save the online buffer as a LeRobot dataset periodically, and it also configured to read saved datasets.


Finally, the `offline_learner_pi05.py` scripts is very similar to `learner_pi05.py` with the key differences that the offline is not interacting with the actor and there is no data transmission. 


I found that the `offline_learner_pi05.py` is necessary to jump start the actor and critic before running `actor_pi05.py`, otherwise the actor is too erratic. Moreover, the data collectiong process (running `actor_pi05.py`) is much faster than learning, e.g., you might collect 5 episodes and only get 20 optimization steps. Therefore, `offline_learner_pi05.py` is helpful to really take advantage of the collected data.



Note: I only train certain layers of the models (see lines 334-345 of `learner_pi05.py`). If you remove those lines, then you train the entire model. Same applies for `offline_learner_pi05.py`.



## How to run

My config file is included in this folder `config-hiserl.json`. It is very similar to the original config-hiserl file. One key difference is the `dataset` section, where `repo_id` contains the offline dataset that you obtain when using `lerobot-record` and save. This dataset is assumed to have only successful episodes. The other datasets are buffers from other runs and they are passed under `"additional_offline_dataset_paths"` in a list. The list may be emtpy. 

