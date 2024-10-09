from copy import deepcopy

import numpy as np


def add_perturbation(
    traj_merged,
    ptb_velocity,
    ptb_i,
    unptb_i,
    velocity_merged,
    idx_match,
    tp_pair,
    tp_ptb,
    idx_ptb_start,
    bin_size=0.006,
    return_vel=False,
):
    """
    Add a perturbation to the trajectory of a specified unperturbed trial.

    This function simulates a perturbation by modifying the velocity profile of an unperturbed
    trial based on the velocity of a specified perturbation trial. It then integrates the modified
    velocity to generate a new trajectory for the unperturbed trial.

    Parameters
    ----------
    traj_merged : ndarray
        The array containing the merged trajectory data for all trials.
    ptb_velocity : ndarray
        The velocity profile for the perturbation trials, used to add perturbation effects.
    ptb_i : int
        The index of the perturbation trial to use.
    unptb_i : int
        The index of the unperturbed trial to which the perturbation will be added.
    velocity_merged : ndarray
        The velocity profile of the unperturbed trials.
    idx_match : int
        The index of the matching point in the unperturbed trial to align with the perturbation.
    tp_pair : int
        The number of timepoints before the perturbation.
    tp_ptb : int
        The duration (in timepoints) of the perturbation to be applied.
    idx_ptb_start : int
        The starting index of the perturbation in the perturbation trial.
    bin_size : float, optional
        The time step in seconds used for integration (default is 0.006 seconds).
    return_vel : bool, optional
        If True, the modified velocity profile will also be returned. Default is False.

    Returns
    -------
    pos_x : ndarray
        The x positions of the trajectory after adding the perturbation.
    pos_y : ndarray
        The y positions of the trajectory after adding the perturbation.
    final_pos : ndarray
        The final (x, y) position after the perturbation.
    idx_stop : int
        The index of the last timepoint used in the trajectory.
    sim_vel_ptb : ndarray, optional
        The simulated velocity profile after adding the perturbation. Returned only if `return_vel` is True.

    Notes
    -----
    - If the perturbation cannot be applied (e.g., index out of range), the function returns None values.
    """
    # Extract the unperturbed trial trajectory
    traj_unptb = traj_merged[unptb_i]

    # Create a copy of the velocity of the unperturbed trial for modification
    sim_vel_ptb = deepcopy(velocity_merged[unptb_i])
    # Zero out the velocities after the last valid timepoint
    sim_vel_ptb[np.where(~np.isnan(sim_vel_ptb[:, 0]))[0][-1] + 1 :] = 0

    try:
        # Add perturbation velocity to the corresponding segment in the unperturbed trial
        sim_vel_ptb[idx_match + tp_pair : idx_match + tp_pair + tp_ptb] += ptb_velocity[
            ptb_i, idx_ptb_start : idx_ptb_start + tp_ptb, :
        ]
    except:
        # If the perturbation cannot be applied, return None values
        if not return_vel:
            return (None,) * 4
        return (None,) * 5

    # Time vector for integration using the provided bin size
    ts = bin_size * np.arange(sim_vel_ptb.shape[0])

    # Integrate the angular velocity to get cumulative angle
    ang_cum = single_trial_angle_integrate(sim_vel_ptb[:, 1], bin_size, ts, flip=1)

    # Define a helper function for zero padding and cumulative summation
    zeropad_and_cumulative = lambda ts, vel: np.hstack(
        (np.zeros(np.sum(ts <= 0)), np.cumsum(vel[ts > 0] * bin_size))
    )

    # Compute x and y velocity components based on cumulative angle
    vx, vy = sim_vel_ptb[:, 0] * np.sin(ang_cum), sim_vel_ptb[:, 0] * np.cos(ang_cum)

    # Integrate velocities to get positions
    pos_x = (
        zeropad_and_cumulative(ts, vx)
        + traj_unptb[np.where(~np.isnan(traj_unptb[:, 1]))[0][0], 0]
    )
    pos_y = (
        zeropad_and_cumulative(ts, vy)
        + traj_unptb[np.where(~np.isnan(traj_unptb[:, 1]))[0][0], 1]
    )

    # Determine the stopping index of the trajectory
    idx_stop = max(
        np.where(~np.isnan(traj_unptb[:, 1]))[0][-1], idx_match + tp_pair + tp_ptb
    )
    final_pos = np.array([pos_x[idx_stop], pos_y[idx_stop]])

    if not return_vel:
        return pos_x, pos_y, final_pos, idx_stop

    return pos_x, pos_y, final_pos, idx_stop, sim_vel_ptb


def integrate_angle(behav, velocity, flip=1):
    """
    Integrate angular velocities to compute the cumulative angle over time for each trial.

    Parameters
    ----------
    behav : object
        An object containing behavior-related information, including time stamps.
    velocity : dict
        A dictionary containing the angular velocities for each trial.
    flip : int, optional
        A factor to flip the angular direction, by default 1.

    Returns
    -------
    int_ang : dict
        A dictionary with the cumulative angle integrated for each trial.
    """
    int_ang = {}
    for key in velocity.keys():
        # Extract the time stamps for the current trial
        ts = behav.time_stamps[key]
        # Compute the cumulative angle using the angular velocity
        cum_ang = single_trial_angle_integrate(velocity[key], behav.dt, ts, flip=flip)
        # Store the result in the dictionary
        int_ang[key] = cum_ang

    return int_ang


def single_trial_angle_integrate(angle, dt, ts, flip=1):
    """
    Compute the cumulative angle over time for a single trial using the angular velocities.

    Parameters
    ----------
    angle : ndarray
        The array of angular velocities (in degrees per second).
    dt : float
        The time step between each sample.
    ts : ndarray
        The array of time stamps for the trial.
    flip : int, optional
        A factor to flip the angular direction, by default 1.

    Returns
    -------
    cum_ang : ndarray
        The cumulative angle at each timepoint for the trial.
    """
    cum_ang = np.zeros(ts.shape)  # Initialize cumulative angle array
    i0 = np.where(ts > 0)[0][0]  # Find the first positive time index

    # Iterate over timepoints and integrate the angle
    for tt in range(i0, ts.shape[0]):
        cum_ang[tt] = (
            cum_ang[tt - 1] + flip * angle[tt] * np.pi / 180.0 * dt
        )  # Convert degrees to radians and integrate

    return cum_ang


def integrate_path(behav, velocity):
    """
    Integrate the velocity to compute the trajectory path over time for each trial.

    Parameters
    ----------
    behav : object
        An object containing behavior-related information, including time stamps.
    velocity : dict
        A dictionary containing the velocities (e.g., radial velocity) for each trial.

    Returns
    -------
    integr_path : dict
        A dictionary with the integrated path (positions) for each trial.
    """
    # Helper function for zero-padding and cumulative summation
    zeropad_and_cumulative = lambda ts, vel, dts: np.hstack(
        (np.zeros(np.sum(ts <= 0)), np.cumsum(vel[ts > 0] * dts[ts > 0]))
    )

    # Apply the helper function to each trial in the dictionary and store results
    dictFuct = lambda ts, vel: {
        key: zeropad_and_cumulative(
            ts[key], vel[key], np.hstack([[0], np.diff(behav.time_stamps[key])])
        )
        for key in vel.keys()
    }

    # Compute the integrated path for each trial and return as a dictionary
    integr_path = dictFuct(behav.time_stamps, velocity)

    return integr_path
