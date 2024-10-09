import os

import numpy as np

from ptb_analysis.io import fireFly_dataPreproc


# %% create the structures
def create_session_trajectory_info(path, session):
    """
    Create a stack of all the trajectories in a session and all the target locations.

    This function processes the data of a given session and returns several arrays
    containing information about the trajectories, perturbation indices, session
    identifiers, trial IDs, target positions, and reward status for each trial.

    Parameters
    ----------
    path : str
        The path to the directory containing the session data files.
    session : str
        The name of the session file (without the file extension).

    Returns
    -------
    trajectory : ndarray of shape (n_trials, max_timepoints, 2)
        The stack of 2D trajectories for each trial in the session. Each trajectory
        contains the (x, y) position of the subject over time.
    ptb_index : ndarray of shape (n_trials,)
        The index of the first perturbation time point for each trial.
        If no perturbation occurred, the value is set to -1.
    session_list : ndarray of shape (n_trials,)
        An array containing the session name for each trial.
    trial_ids : ndarray of shape (n_trials,)
        The unique trial identifiers for each trial in the session.
    target_pos : ndarray of shape (n_trials, 2)
        The (x, y) positions of the target location for each trial.
    is_rewarded : ndarray of shape (n_trials,)
        Boolean array indicating whether each trial was rewarded (`True`) or not (`False`).
    ptb_velocity : ndarray of shape (n_trials, max_timepoints, 2)
        The perturbation velocities (radial, angular) for each trial. If perturbation
        data is unavailable, the values are set to NaN.
    tot_velocity : ndarray of shape (n_trials, max_timepoints, 2)
        The total velocities (radial, angular) for each trial, computed from the
        processed data.

    """
    dat = fireFly_dataPreproc(os.path.join(path, session + ".mat"))
    dat.set_filters("all", True)
    dat.preProcPCA(
        binMs=6,
        init_event="t_flyON",
        final_event="t_stop",
        smooth=False,
        filt_window=None,
        preTrialMs=0,
        postTrialMs=0,
        add_events=True,
    )
    max_tp = 0
    num_trs = np.unique(dat.preProcessed.trialId).shape[0]
    for tr in np.unique(dat.preProcessed.trialId):
        sel = dat.preProcessed.trialId == tr
        max_tp = max(max_tp, sel.sum())

    trajectory = np.zeros((num_trs, max_tp, 2), dtype=np.float32) * np.nanco
    ptb_index = np.zeros((num_trs), dtype=int)  # set -1 in unptb
    session_list = np.zeros((num_trs), dtype="U20")
    trial_ids = np.zeros((num_trs), dtype=int)
    target_pos = np.zeros((num_trs, 2), dtype=np.float32)
    tot_velocity = np.zeros((num_trs, max_tp, 2), dtype=np.float32) * np.nan
    ptb_velocity = np.zeros((num_trs, max_tp, 2), dtype=np.float32) * np.nan
    is_rewarded = np.zeros((num_trs), dtype=bool)

    session_list[:] = session
    cnt = 0
    for tr in np.unique(dat.preProcessed.trialId):
        sel = dat.preProcessed.trialId == tr
        trajectory[cnt, : sel.sum(), 0] = dat.preProcessed.covariates["x_monk"][sel]
        trajectory[cnt, : sel.sum(), 1] = dat.preProcessed.covariates["y_monk"][sel]
        tot_velocity[cnt, : sel.sum(), 0] = dat.preProcessed.covariates["rad_vel"][sel]
        tot_velocity[cnt, : sel.sum(), 1] = dat.preProcessed.covariates["ang_vel"][sel]
        try:
            ptb_velocity[cnt, : sel.sum(), 0] = dat.preProcessed.covariates[
                "rad_vel_ptb"
            ][sel]
            ptb_velocity[cnt, : sel.sum(), 1] = dat.preProcessed.covariates[
                "ang_vel_ptb"
            ][sel]
        except:
            pass

        if sum(dat.preProcessed.covariates["t_ptb"][sel]):
            ptb_index[cnt] = np.where(dat.preProcessed.covariates["t_ptb"][sel])[0][0]
        else:
            ptb_index[cnt] = -1
        trial_ids[cnt] = tr
        target_pos[cnt, 0] = dat.preProcessed.covariates["x_fly"][cnt]
        target_pos[cnt, 1] = dat.preProcessed.covariates["y_fly"][cnt]

        is_rewarded[cnt] = all(~np.isnan(dat.behav.events.t_reward[tr]))

        cnt += 1

    return (
        trajectory,
        ptb_index,
        session_list,
        trial_ids,
        target_pos,
        is_rewarded,
        ptb_velocity,
        tot_velocity,
    )


def merge_trajectories(traj1, traj2):
    """
    Merge two sets of trajectories into a single array with consistent timepoints.

    This function combines two trajectory arrays into a single array, padding with NaNs
    where necessary to ensure that both trajectory sets have the same number of timepoints.

    Parameters
    ----------
    traj1 : ndarray of shape (n_trials_1, n_timepoints_1, 2)
        The first set of 2D trajectories, where each trajectory contains the (x, y)
        positions over time.
    traj2 : ndarray of shape (n_trials_2, n_timepoints_2, 2)
        The second set of 2D trajectories, where each trajectory contains the (x, y)
        positions over time.

    Returns
    -------
    traj_merged : ndarray of shape (n_trials_1 + n_trials_2, max(n_timepoints_1, n_timepoints_2), 2)
        The merged trajectory array, with dimensions sufficient to include all trials
        from both input arrays. Missing values are filled with NaNs.

    Examples
    --------
    >>> traj1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> traj2 = np.array([[[9, 10], [11, 12], [13, 14]]])
    >>> merged = merge_trajectories(traj1, traj2)
    >>> print(merged.shape)
    (3, 3, 2)
    """
    mx_tp = max(traj1.shape[1], traj2.shape[1])
    traj_merged = (
        np.zeros((traj1.shape[0] + traj2.shape[0], mx_tp, 2), dtype=np.float32) * np.nan
    )

    for cnt in range(traj1.shape[0]):
        traj_merged[cnt, : traj1.shape[1], :] = traj1[cnt]

    for cnt_2 in range(traj2.shape[0]):
        traj_merged[cnt + 1 + cnt_2, : traj2.shape[1], :] = traj2[cnt_2]
    return traj_merged


def compute_min_diff_by_trial(traj_before_ptb, other_traj):
    """
    Compute the minimum difference between a trajectory segment and multiple other trajectories.

    This function shifts a given trajectory segment across all possible sub-trajectories in
    another set of trajectories, computing the minimum Euclidean distance.

    Parameters
    ----------
    traj_before_ptb : ndarray of shape (n_timepoints_1, 2)
        The trajectory segment before the perturbation. This is a single trajectory
        containing the (x, y) positions over time.
    other_traj : ndarray of shape (n_timepoints_2, 2)
        The other set of trajectories to compare against. This is a single trajectory
        or a set of trajectories with the (x, y) positions over time.

    Returns
    -------
    imin : int
        The index of the segment in `other_traj` that has the minimum difference with `traj_before_ptb`.
        If `traj_before_ptb` has more timepoints than `other_traj`, returns -1.
    dmin : float
        The minimum Euclidean distance between `traj_before_ptb` and the closest segment in `other_traj`.
        If `traj_before_ptb` has more timepoints than `other_traj`, returns NaN.

    Notes
    -----
    - If `traj_before_ptb` has more timepoints than `other_traj`, the function returns -1 and NaN.
    - If there are no valid segments to compare, the function returns -1 and NaN.

    Examples
    --------
    >>> traj_before_ptb = np.array([[1, 2], [3, 4]])
    >>> other_traj = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> imin, dmin = compute_min_diff_by_trial(traj_before_ptb, other_traj)
    >>> print(imin, dmin)
    0 0.0
    """
    if traj_before_ptb.shape[0] > other_traj.shape[0]:
        return -1, np.nan
    mat_shifts = np.zeros(
        (other_traj.shape[0] - traj_before_ptb.shape[0], traj_before_ptb.shape[0], 2)
    )
    for i in range(other_traj.shape[0] - traj_before_ptb.shape[0]):
        mat_shifts[i, :] = other_traj[i : i + traj_before_ptb.shape[0]]
    try:
        dsts = np.linalg.norm(mat_shifts - traj_before_ptb, axis=(1, 2))
        imin = np.nanargmin(dsts)
    except ValueError:
        return -1, np.nan
    dmin = dsts[imin]
    return imin, dmin
