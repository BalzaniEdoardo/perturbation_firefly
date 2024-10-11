"""
This script extract the monkey trajectory and info related to each trial, and saves it into an array.

"""
import os
import re
from copy import deepcopy

import numpy as np

from ptb_analysis.process import (create_session_trajectory_info,
                                  merge_trajectories)
import matplotlib.pyplot as plt

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

# here:
#  - traj_merged has shape (n_trials, n_samples, 2), two is for x,y coord.
#    plt.plot(*traj_merged[100].T) will plot a trial
#  - ptb_index_merged: shape (n_trials) contains the index of the sample
#    corresponding to when the ptb was delivered.
#  - session_list_merged: an array (num_trials,) with the session label
#  - trial_ids_merged: (num_trials,) the id of the trial
#  - target_pos_merged: (num_trials, 2) the target position for each trial
#  - is_rewarded_merged: (num_trials,) bool if the trial was rewarded
#  - ptb_velocity_merged: (n_trials, n_samples, 2) velocity vx and vy of the ptb
#  - tot_velocity_merged: (n_trials, n_samples, 2) velocity vx and vy of the monkey (ptb + joystick displacement)

# plot two trials, rewarded and unrewarded
tr_rew = np.where(is_rewarded_merged)[0][0]
tr_unrew = np.where(~is_rewarded_merged)[0][0]


# compute rew zone (it shoould be 60cm, ask JP)
ang = np.linspace(0, np.pi*2,200)
rew_zone = 60 * np.c_[np.cos(ang), np.sin(ang)]

fig, axs = plt.subplots(1,2)
axs[0].plot(*traj_merged[tr_rew].T)
axs[0].plot(*target_pos_merged[tr_rew].T, "or", markersize=10)
axs[0].set_xlabel("cm")
axs[0].set_ylabel("cm")
axs[0].plot(*(rew_zone + target_pos_merged[tr_rew]).T, color="r")
plt.legend(["trajectory", "target", "reward zone"])
axs[0].set_aspect('equal')

axs[1].plot(*traj_merged[tr_unrew].T)
axs[1].plot(*target_pos_merged[tr_unrew].T, "or", markersize=10)
axs[1].set_xlabel("cm")
axs[1].set_ylabel("cm")
axs[1].plot(*(rew_zone + target_pos_merged[tr_unrew]).T, color="r")
plt.legend(["trajectory", "target", "reward zone"])
axs[1].set_aspect('equal')
plt.show()


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
