"""MPI parallelized code to pair trajectories according to how similar they are before the perturbation happens."""
import numpy as np
from ptb_analysis.process import compute_min_diff_by_trial
from mpi4py import MPI

# Initialize the MPI communicator, obtain the size and rank of the process
comm = MPI.COMM_WORLD
size = comm.Get_size()  # Total number of processes
rank = comm.Get_rank()  # Rank of the current process (0, 1, ..., size-1)

# PATH SETTINGS
path_to_trajectory_info = ""

# If the current process is the master (rank 0), load the trajectory data
if rank == 0:
    try:
        # Attempt to load the trajectory data from an external drive
        dat_traj = np.load('/Volumes/TOSHIBA EXT/TAME_paper/behavior/trajectory_and_info.npz')
    except:
        # If not found, try loading the data from the local directory
        dat_traj = np.load('trajectory_and_info.npz')

    # Extract relevant data arrays from the loaded file
    traj_merged = dat_traj['trajectory']
    ptb_index_merged = dat_traj['ptb_index']
    session_list_merged = dat_traj['session_list']
    trial_ids_merged = dat_traj['trial_ids']
    target_pos_merged = dat_traj['target_pos']
    is_rewarded_merged = dat_traj['is_rewarded']
    tot_velocity_merged = dat_traj['tot_velocity']
    ptb_velocity_merged = dat_traj['ptb_velocity']

    # Calculate the number of perturbation trials
    tot_ptb = (ptb_index_merged != -1).sum()
    idx_ptb = np.where(ptb_index_merged != -1)[0]  # Indices of perturbation trials
    tot_unptb = ptb_index_merged.shape[0] - tot_ptb  # Total number of unperturbed trials

    # Distribute the perturbation indices evenly across processes
    use_ptb_dict = {}
    num_trs = int(np.ceil(tot_ptb / size))  # Number of trials assigned to each process
    for k in range(size):
        use_ptb_dict[k] = idx_ptb[k * num_trs:(k + 1) * num_trs]  # Assign indices to each process
    print(use_ptb_dict)  # Print the distribution of trials across processes

else:
    # Initialize variables to None for other ranks
    traj_merged = None
    ptb_index_merged = None
    session_list_merged = None
    trial_ids_merged = None
    target_pos_merged = None
    is_rewarded_merged = None
    tot_velocity_merged = None
    ptb_velocity_merged = None
    use_ptb_dict = None
    tot_ptb = None
    tot_unptb = None

# Broadcast the data arrays and metadata from rank 0 to all processes
traj_merged = comm.bcast(traj_merged, root=0)
ptb_index_merged = comm.bcast(ptb_index_merged, root=0)
session_list_merged = comm.bcast(session_list_merged, root=0)
trial_ids_merged = comm.bcast(trial_ids_merged, root=0)
target_pos_merged = comm.bcast(target_pos_merged, root=0)
is_rewarded_merged = comm.bcast(is_rewarded_merged, root=0)
tot_velocity_merged = comm.bcast(tot_velocity_merged, root=0)
ptb_velocity_merged = comm.bcast(ptb_velocity_merged, root=0)
use_ptb_dict = comm.bcast(use_ptb_dict, root=0)
tot_ptb = comm.bcast(tot_ptb, root=0)
tot_unptb = comm.bcast(tot_unptb, root=0)

# %% Local processing for each rank

# Define time parameters
dt_ms = 6  # Time step in milliseconds
min_tp = int(np.ceil(500 / 6))  # Minimum number of timepoints for comparison

# Number of perturbation trials assigned to this rank
ptb_rank_num = len(use_ptb_dict[rank])

# Initialize arrays for storing trial labels, distances, and optimal matches
trial_lab = np.zeros((ptb_rank_num, tot_unptb), dtype=int)
dist_x_trial = np.zeros((ptb_rank_num, tot_unptb), dtype=np.float32) * np.nan
idx_optim_match = -1 * np.ones((ptb_rank_num, tot_unptb), dtype=int)

# Identify the last and first non-NaN indices for unperturbed trials
id_unpt_list = np.where(ptb_index_merged == -1)[0]
last_non_nan = np.zeros(id_unpt_list.shape, dtype=int)
first_non_nan = np.zeros(id_unpt_list.shape, dtype=int)
cc = 0
for ii in id_unpt_list:
    # Calculate relative trajectory positions to target
    other_traj = target_pos_merged[ii] - traj_merged[ii]
    if all(np.isnan(other_traj.sum(axis=1))):
        # If all values are NaN, set the first and last indices to 0
        last_non_nan[cc] = 0
        first_non_nan[cc] = 0
    else:
        # Otherwise, identify the indices of non-NaN values
        nnan = np.where(~np.isnan(other_traj.sum(axis=1)))[0]
        last_non_nan[cc] = nnan[-1]
        first_non_nan[cc] = nnan[0]
    cc += 1

# Compute distance metrics for each perturbation trial assigned to this rank
cnt_ptb = 0
for idx in use_ptb_dict[rank]:
    print(rank, cnt_ptb, ptb_rank_num)  # Print current status of the process
    ptb_idx = ptb_index_merged[idx]

    if ptb_idx < min_tp:
        # Skip if perturbation index is less than minimum timepoints
        dist_x_trial[cnt_ptb, :] = np.nan
    else:
        # Calculate the trajectory segment before the perturbation
        rel_traj_before_ptb = target_pos_merged[idx] - traj_merged[idx, ptb_idx - min_tp:ptb_idx]

        cnt_unptb = 0
        for idx_unpt in id_unpt_list:
            # If the unperturbed trial does not have enough timepoints, set NaN
            if last_non_nan[cnt_unptb] - first_non_nan[cnt_unptb] - 1 < min_tp:
                dist_x_trial[cnt_ptb, cnt_unptb] = np.nan
            else:
                # Calculate the trajectory segment for the unperturbed trial
                other_traj = target_pos_merged[idx_unpt] - traj_merged[idx_unpt, :last_non_nan[cnt_unptb]]
                # Compute the minimum distance between the perturbation segment and unperturbed trial
                id_opt, dst = compute_min_diff_by_trial(rel_traj_before_ptb, other_traj)
                dist_x_trial[cnt_ptb, cnt_unptb] = dst  # Store the distance
                idx_optim_match[cnt_ptb, cnt_unptb] = id_opt  # Store the index of optimal match
            cnt_unptb += 1
    cnt_ptb += 1

# Gather results from all processes to the root process (rank 0)
trial_lab_list = comm.gather(trial_lab, root=0)
dist_x_trial_list = comm.gather(dist_x_trial, root=0)
idx_optim_match_list = comm.gather(idx_optim_match, root=0)

# If this is the root process, concatenate the results from all processes
if rank == 0:
    for k in range(size):
        if k == 0:
            # Initialize final arrays with the results from the first process
            trial_lab_fin = trial_lab_list[0]
            dist_x_trial_fin = dist_x_trial_list[0]
            idx_optim_match_fin = idx_optim_match_list[0]
        else:
            # Append the results from subsequent processes
            trial_lab_fin = np.vstack((trial_lab_fin, trial_lab_list[k]))
            dist_x_trial_fin = np.vstack((dist_x_trial_fin, dist_x_trial_list[k]))
            idx_optim_match_fin = np.vstack((idx_optim_match_fin, idx_optim_match_list[k]))

    # Save the final results to a compressed .npz file
    np.savez('%d_mpi_distance_matrix_compute.npz' % size,
             dist_x_trial=dist_x_trial_fin, trial_lab=trial_lab_fin,
             ptb_idx=np.where(ptb_index_merged != -1)[0],
             unptb_idx=np.where(ptb_index_merged == -1)[0],
             idx_optim_match=idx_optim_match_fin)
