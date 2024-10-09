import os
import re
from copy import deepcopy

import numpy as np

from ptb_analysis.process import (create_session_trajectory_info,
                                  merge_trajectories)

# path to the folder where all the math files are located
path = "/Users/ebalzani/Desktop/firefly_data/"

# pattern for filtering sessions
patttern = "^m53s\d+.mat$"

# skip sessions
skip_list = ["m53s128.mat"]


# comment if you don't want to restart
session_list = np.zeros(0, dtype="U20")
cc = 0
for fh in os.listdir(path):

    # filter sessions
    if not re.match(patttern, fh):
        continue

    if fh in skip_list:
        continue

    # skip already processed
    if any(session_list == fh.split(".")[0]):
        continue

    print(fh)
    (
        trajectory,
        ptb_index,
        session_list,
        trial_ids,
        target_pos,
        is_rewarded,
        ptb_velocity,
        tot_velocity,
    ) = create_session_trajectory_info(path, fh.split(".")[0])

    if cc != 0:
        traj_merged = merge_trajectories(traj_merged, trajectory)
        ptb_index_merged = np.hstack((ptb_index_merged, ptb_index))
        session_list_merged = np.hstack((session_list_merged, session_list))
        trial_ids_merged = np.hstack((trial_ids_merged, trial_ids))
        target_pos_merged = np.vstack((target_pos_merged, target_pos))
        is_rewarded_merged = np.hstack((is_rewarded_merged, is_rewarded))
        ptb_velocity_merged = merge_trajectories(ptb_velocity_merged, ptb_velocity)
        tot_velocity_merged = merge_trajectories(tot_velocity_merged, tot_velocity)

    else:
        ptb_velocity_merged = deepcopy(ptb_velocity)
        tot_velocity_merged = deepcopy(tot_velocity)
        traj_merged = deepcopy(trajectory)
        ptb_index_merged = deepcopy(ptb_index)
        session_list_merged = deepcopy(session_list)
        trial_ids_merged = deepcopy(trial_ids)
        target_pos_merged = deepcopy(target_pos)
        is_rewarded_merged = deepcopy(is_rewarded)

    cc += 1


np.savez(
    "output/trajectory_and_info.npz",
    trajectory=traj_merged,
    ptb_index=ptb_index_merged,
    session_list=session_list_merged,
    trial_ids=trial_ids_merged,
    target_pos=target_pos_merged,
    is_rewarded=is_rewarded_merged,
    ptb_velocity=ptb_velocity_merged,
    tot_velocity=tot_velocity_merged,
)
