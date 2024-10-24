"""
Module for loading the matlab structure into a very similar python structure.
"""

from datetime import datetime

import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d

from .behav_class import behavior_experiment, load_trial_types
from .lfp_class import lfp_class
from .spike_times_class import spike_counts
from copy import deepcopy

def dict_to_vec(dictionary):
    return np.hstack(list(dictionary.values()))


def time_stamps_rebin(time_stamps, binwidth_ms=20):
    rebin = {}
    for tr in time_stamps.keys():
        ts = time_stamps[tr]
        tp_num = np.floor((ts[-1] - ts[0]) * 1000 / (binwidth_ms))
        rebin[tr] = ts[0] + np.arange(tp_num) * binwidth_ms / 1000.0
    return rebin


class data_handler(object):
    def __init__(
        self,
        dat,
        beh_key,
        spike_key,
        lfp_key,
        behav_stat_key,
        time_aligned_to_beh=True,
        dt=0.006,
        flyON_dur=0.3,
        pre_trial_dur=0.25,
        post_trial_dur=0.25,
        is_lfp_binned=True,
        extract_lfp_phase=True,
        lfp_beta=None,
        lfp_alpha=None,
        lfp_theta=None,
        use_eye=None,
        extract_fly_and_monkey_xy=False,
        extract_cartesian_eye_and_firefly=False,
        fhLFP="",
        skip_not_ok=True,
    ):

        self.info = load_trial_types(
            dat[behav_stat_key].flatten(),
            dat[beh_key].flatten(),
            skip_not_ok=skip_not_ok,
        )
        # import all data and trial info
        self.spikes = spike_counts(
            dat, spike_key, time_aligned_to_beh=time_aligned_to_beh
        )
        if lfp_key is None:
            self.lfp = None
        else:
            self.lfp = lfp_class(
                dat,
                lfp_key,
                binned=is_lfp_binned,
                lfp_beta=lfp_beta,
                lfp_alpha=lfp_alpha,
                lfp_theta=lfp_theta,
                compute_phase=extract_lfp_phase,
                fhLFP=fhLFP,
            )
        self.behav = behavior_experiment(
            dat,
            beh_key,
            behav_stat_key=behav_stat_key,
            dt=dt,
            flyON_dur=flyON_dur,
            pre_trial_dur=pre_trial_dur,
            post_trial_dur=post_trial_dur,
            info=self.info,
            use_eye=use_eye,
            extract_fly_and_monkey_xy=extract_fly_and_monkey_xy,
            extract_cartesian_eye_and_firefly=extract_cartesian_eye_and_firefly,
        )

        self.date_exp = datetime.strptime(dat["prs"]["sess_date"][0, 0][0], "%d-%b-%Y")

        # set the filter to trials in which the monkey worked
        self.filter = self.info.get_all(True)
        # save a dicitonary with the info regarding the selected trial
        self.filter_descr = {"all": True}

    def align_spike_times_to_beh(self):
        print("Method still empty")
        return

    def compute_train_and_test_filter(self, perc_train_trial=0.8, seed=None):
        if ~(seed is None):
            np.random.seed(seed)

        # compute how many of the selected trials will be in the training set
        num_selected = np.sum(self.filter)
        tot_train = int(perc_train_trial * num_selected)

        # make sure that the trial we select are in the filtered
        choiche_idx = np.arange(self.spikes.n_trials)[self.filter]
        # select the training set
        train = np.zeros(self.spikes.n_trials, dtype=bool)
        train_idx = np.random.choice(choiche_idx, size=tot_train, replace=False)
        train[train_idx] = True

        test = (~train) * self.filter

        return train, test

    def GPFA_YU_preprocessing(
        self,
        list_timepoints=None,
        var_list=[],
        pcaPrep=False,
        sqrtIfPCA=True,
        filt_window=None,
        smooth=True,
        event_list=[],
    ):
        if list_timepoints is None:
            list_timepoints = [("t_move", "t_stop", 75), ("t_stop", "t_reward", 15)]

        trial_use = np.arange(self.spikes.n_trials)[self.filter]  # self.spikes.n_trials
        n_trials = trial_use.shape[0]

        # check if the events are consecutive
        check_ev0 = []
        check_ev1 = []

        tot_tp = 0
        for ev0, ev1, tp in list_timepoints:
            check_ev0 += [ev0]
            check_ev1 += [ev1]
            tot_tp += tp - 1

        # tot_tp -= 1

        tp_matrix = np.zeros((n_trials, tot_tp)) * np.nan

        rate_tensor = np.zeros((n_trials, self.spikes.num_units, tot_tp)) * np.nan
        sm_traj = np.zeros((n_trials, 2, tot_tp)) * np.nan  # xy position
        raw_traj = np.zeros((n_trials, 2, tot_tp)) * np.nan
        fly_pos = np.zeros((n_trials, 2)) * np.nan

        tw_correlates = {}
        for var in var_list:
            tw_correlates[var] = np.zeros((n_trials, tot_tp)) * np.nan

        for var in event_list:
            tw_correlates[var] = np.zeros((n_trials, tot_tp))

        check_ev0 = np.array(check_ev0[1:])
        check_ev1 = np.array(check_ev1[:-1])
        assert all(check_ev0 == check_ev1)

        if pcaPrep:
            # smooth spikes
            ev0 = list_timepoints[0][0]
            ev1 = list_timepoints[-1][1]
            add = 0
            add_stop = 0
            if ev0 == "t_flyON":
                ev0 = "t_targ"
                add = 0

            elif ev0 == "t_flyOFF" or ev0 == "t_targ_off":
                ev0 = "t_targ"
                add = self.behav.flyON_dur

            if ev1 == "t_flyON":
                ev1 = "t_targ"
                add_stop = 0
            elif ev1 == "t_flyOFF" or ev1 == "t_targ_off":
                ev1 = "t_targ"
                add_stop = self.behav.flyON_dur

            if smooth:
                t_start = (
                    dict_to_vec(self.behav.events.__dict__[ev0])
                    + add
                    - self.behav.pre_trial_dur
                )
                t_stop = (
                    dict_to_vec(self.behav.events.__dict__[ev1])
                    + add_stop
                    + self.behav.pre_trial_dur
                )
                time_dict = self.spikes.bin_spikes(
                    self.behav.time_stamps, t_start=t_start, t_stop=t_stop
                )
                DT = time_dict[0][1] - time_dict[0][0]
                print("begin smoothing spikes for PCA")
                sm_spikes = np.zeros(self.spikes.binned_spikes.shape, dtype=object)
                for tr in range(self.spikes.binned_spikes.shape[1]):
                    for un in range(self.spikes.binned_spikes.shape[0]):
                        sm_spikes[un, tr] = np.convolve(
                            self.spikes.binned_spikes[un, tr] / DT,
                            filt_window,
                            mode="same",
                        )
                print("end smoothing spikes for PCA")

        if sqrtIfPCA and pcaPrep:
            transFun = lambda x: np.sqrt(x)
        elif pcaPrep:
            transFun = lambda x: x
        # loop over trials
        for indx_tr in range(n_trials):

            tr = trial_use[indx_tr]
            # spk_times = self.spikes.spike_times[:,tr]
            time_bins = []

            # extract smooth trajectories
            # trajectory_tr = np.zeros(len(self.behav.time_stamps[tr]))*np.nan
            traj_sele = (self.behav.time_stamps[tr] > self.behav.events.t_targ[tr]) * (
                self.behav.time_stamps[tr] <= self.behav.events.t_stop[tr]
            )

            Num = traj_sele.sum()
            valid_tr = any(traj_sele) and (Num > 20)
            if valid_tr:
                x_fly = self.behav.continuous.x_fly[tr]
                y_fly = self.behav.continuous.y_fly[tr]
                ts = self.behav.time_stamps[tr][traj_sele]
                x_monk = self.behav.continuous.x_monk[tr][traj_sele]
                y_monk = self.behav.continuous.y_monk[tr][traj_sele]

                fr = 20.0 / Num
                # print(tr,fr)
                non_nan = ~np.isnan(x_monk)
                x_smooth = np.nan * np.zeros((x_monk.shape[0], 2))
                y_smooth = np.nan * np.zeros((x_monk.shape[0], 2))
                x_smooth[non_nan, :] = sm.nonparametric.lowess(
                    x_monk, np.arange(x_monk.shape[0]), fr
                )
                y_smooth[non_nan, :] = sm.nonparametric.lowess(
                    y_monk, np.arange(y_monk.shape[0]), fr
                )
                x_smooth = x_smooth[:, 1]
                y_smooth = y_smooth[:, 1]

                fly_pos[indx_tr, 0] = x_fly
                fly_pos[indx_tr, 1] = y_fly

            skip_trial = False
            cc = 1
            for ev0, ev1, tp in list_timepoints:

                if ev0 == "t_flyON":
                    ev0 = "t_targ"

                elif ev0 == "t_flyOFF":
                    ev0 = "t_targ_off"

                if ev1 == "t_flyON":
                    ev1 = "t_targ"

                elif ev1 == "t_flyOFF":
                    ev1 = "t_targ_off"

                if ev0 != "t_targ_off":
                    t0 = self.behav.events.__dict__[ev0][tr][0]
                else:
                    t0 = (
                        self.behav.events.__dict__["t_targ"][tr][0]
                        + self.behav.flyON_dur
                    )

                if ev1 != "t_targ_off":
                    t1 = self.behav.events.__dict__[ev1][tr][0]
                else:
                    t1 = (
                        self.behav.events.__dict__["t_targ"][tr][0]
                        + self.behav.flyON_dur
                    )

                if any(np.isnan([t0, t1])):
                    skip_trial = True
                    break

                if t1 < t0:
                    skip_trial = True
                    break

                time_lst = np.linspace(t0, t1, tp)
                if cc != len(list_timepoints):
                    time_lst = time_lst[:-1]

                time_bins = np.hstack((time_bins, time_lst))
                cc += 1

            if skip_trial:
                print("skipping trial %d" % tr)
                continue

            tp_matrix[indx_tr, :] = 0.5 * (time_bins[:-1] + time_bins[1:])
            time_int_dur = np.diff(time_bins)

            if (not pcaPrep) or (not smooth):
                for unt in range(self.spikes.num_units):
                    rate_tensor[indx_tr, unt, :] = (
                        np.histogram(self.spikes.spike_times[unt, tr], bins=time_bins)[
                            0
                        ]
                        / time_int_dur
                    )
            else:
                # print('start interp smooth spike for PCA')
                for unt in range(self.spikes.num_units):
                    interp = interp1d(
                        time_dict[tr], transFun(sm_spikes[unt, tr]), bounds_error=False
                    )
                    rate_tensor[indx_tr, unt, :] = interp(tp_matrix[indx_tr, :])
                # print('end interp smooth spike for PCA')

            # compute linearly interp trajectory position
            sele_tp = (ts >= time_bins[0]) & (ts < time_bins[-1])
            if not any(sele_tp):
                continue
            # smooth interp
            intrp = interp1d(ts[sele_tp], x_smooth[sele_tp], bounds_error=False)
            sm_traj[indx_tr, 0] = intrp(tp_matrix[indx_tr, :])
            intrp = interp1d(ts[sele_tp], y_smooth[sele_tp], bounds_error=False)
            sm_traj[indx_tr, 1] = intrp(tp_matrix[indx_tr, :])

            # raw interp
            intrp = interp1d(ts[sele_tp], x_monk[sele_tp], bounds_error=False)
            raw_traj[indx_tr, 0] = intrp(tp_matrix[indx_tr, :])
            intrp = interp1d(ts[sele_tp], y_monk[sele_tp], bounds_error=False)
            raw_traj[indx_tr, 1] = intrp(tp_matrix[indx_tr, :])

            # interp variables
            for var in var_list:
                time_pts = self.behav.time_stamps[tr]
                y_val = self.behav.continuous.__dict__[var][tr]
                non_nan = ~np.isnan(y_val)
                intrp = interp1d(time_pts[non_nan], y_val[non_nan], bounds_error=False)
                tw_correlates[var][indx_tr, :] = intrp(tp_matrix[indx_tr, :])

            for var in event_list:
                # time_pts = self.behav.time_stamps[tr]
                y_val = self.behav.events.__dict__[var][tr]

                for val in y_val:
                    if np.isnan(val):
                        continue
                    t_iidx = (tp_matrix[indx_tr, :] >= val) & (
                        tp_matrix[indx_tr, :] < val
                    )
                    if t_iidx.sum() > 0:
                        tw_correlates[indx_tr, t_iidx] = 1

        return (
            tp_matrix,
            rate_tensor,
            sm_traj,
            raw_traj,
            fly_pos,
            tw_correlates,
            trial_use,
        )

    def GPFA_YU_preprocessing_noTW(self, t_start, t_stop, var_list=[], binwidth_ms=20):
        if binwidth_ms is None:
            bin_ts = self.behav.time_stamps
        else:
            bin_ts = time_stamps_rebin(self.behav.time_stamps, binwidth_ms=binwidth_ms)
        bin_list = self.spikes.bin_spikes(
            bin_ts, t_start=t_start, t_stop=t_stop, select=self.filter
        )
        trialId = {}
        spikes = {}
        tr_sel = np.array(np.arange(self.spikes.n_trials)[self.filter], dtype=int)
        ydim = self.spikes.binned_spikes.shape[0]

        sm_traj = np.zeros((tr_sel.shape[0], 2), dtype=object)
        raw_traj = np.zeros((tr_sel.shape[0], 2), dtype=object)
        fly_pos = np.zeros((tr_sel.shape[0], 2)) * np.nan

        tw_correlates = {}
        bbin_ts = {}

        for var in var_list:
            tw_correlates[var] = np.zeros((tr_sel.shape[0],), dtype=object)

        for cc in range(tr_sel.shape[0]):

            tr = tr_sel[cc]
            # extract smooth trajectories
            traj_sele = (self.behav.time_stamps[tr] > self.behav.events.t_targ[tr]) * (
                self.behav.time_stamps[tr] <= self.behav.events.t_stop[tr]
            )

            Num = traj_sele.sum()
            valid_tr = any(traj_sele) and (Num > 20)
            if valid_tr:
                x_fly = self.behav.continuous.x_fly[tr]
                y_fly = self.behav.continuous.y_fly[tr]
                x_monk = self.behav.continuous.x_monk[tr][traj_sele]
                y_monk = self.behav.continuous.y_monk[tr][traj_sele]

                fr = 20.0 / Num
                # print(tr,fr)
                non_nan = ~np.isnan(x_monk)
                x_smooth = np.nan * np.zeros((x_monk.shape[0], 2))
                y_smooth = np.nan * np.zeros((x_monk.shape[0], 2))
                x_smooth[non_nan, :] = sm.nonparametric.lowess(
                    x_monk, np.arange(x_monk.shape[0]), fr
                )
                y_smooth[non_nan, :] = sm.nonparametric.lowess(
                    y_monk, np.arange(y_monk.shape[0]), fr
                )
                x_smooth = x_smooth[:, 1]
                y_smooth = y_smooth[:, 1]

                fly_pos[cc, 0] = x_fly
                fly_pos[cc, 1] = y_fly

            tdim = self.spikes.binned_spikes[0, cc].shape[0]
            spikes[cc] = np.zeros((ydim, tdim))
            for i in range(ydim):
                spikes[cc][i, :] = self.spikes.binned_spikes[i, cc]
            trialId[cc] = tr_sel[cc]

            # smooth interp
            intrp = interp1d(
                self.behav.time_stamps[tr][traj_sele], x_smooth, bounds_error=False
            )
            sm_traj[cc, 0] = intrp(bin_list[tr])
            intrp = interp1d(
                self.behav.time_stamps[tr][traj_sele], y_smooth, bounds_error=False
            )
            sm_traj[cc, 1] = intrp(bin_list[tr])

            # raw interp
            intrp = interp1d(
                self.behav.time_stamps[tr][traj_sele], x_monk, bounds_error=False
            )
            raw_traj[cc, 0] = intrp(bin_list[tr])
            intrp = interp1d(
                self.behav.time_stamps[tr][traj_sele], y_monk, bounds_error=False
            )
            raw_traj[cc, 1] = intrp(bin_list[tr])

            # interp variables
            for var in var_list:
                time_pts = self.behav.time_stamps[tr]
                y_val = self.behav.continuous.__dict__[var][tr]
                non_nan = ~np.isnan(y_val)
                # try:
                intrp = interp1d(time_pts[non_nan], y_val[non_nan], bounds_error=False)
                # except:
                #     ccc=1
                tw_correlates[var][cc] = intrp(bin_list[tr])
            bbin_ts[cc] = bin_list[tr]
            # cc += 1
        # remove ts of other trials
        # cc = 1
        # for tr in tr_sel:
        #     bbin_ts[cc] = bin_ts[tr]
        #     cc+=1

        return bbin_ts, spikes, sm_traj, raw_traj, fly_pos, tw_correlates, trialId

    def rebin_time_stamps(self, bin_sec):
        time_stamps = {}
        for tr in self.behav.time_stamps:
            ts = self.behav.time_stamps[tr]
            ts0 = np.floor(ts[0] / bin_sec)
            ts1 = np.ceil(ts[-1] / bin_sec)
            time_stamps[tr] = ts0 * bin_sec + bin_sec * np.arange(
                0, int(ts1 + np.abs(ts0)) + 1
            )
        return time_stamps

    def concatenate_inputs(
        self, *varnames, t_start=None, t_stop=None, time_stamps=None
    ):
        if time_stamps is None:
            time_stamps = deepcopy(self.behav.time_stamps)
            rebin = False
        else:
            rebin = True

        self.spikes.bin_spikes(
            time_stamps, t_start=t_start, t_stop=t_stop, select=self.filter
        )

        edges_sel = np.arange(self.spikes.n_trials)[self.filter]

        spikes = self.spikes.binned_spikes

        # count the input data shape
        cc = 0
        for tr in range(spikes.shape[1]):
            cc += spikes[0, tr].shape[0]

        # stack all spike counts in a single vector per each unit
        tmp_spikes = np.zeros((spikes.shape[0], cc))
        trial_idx = np.zeros(cc, dtype=int)

        for unt in range(spikes.shape[0]):
            cc = 0
            for tr in range(spikes.shape[1]):
                d_idx = spikes[unt, tr].shape[0]
                tmp_spikes[unt, cc : cc + d_idx] = spikes[unt, tr]
                trial_idx[cc : cc + d_idx] = edges_sel[tr]
                cc += d_idx

        spikes = tmp_spikes

        event_names = list(self.behav.events.__dict__.keys())
        continuous_names = list(self.behav.continuous.__dict__.keys())
        var_dict = {}

        for var in varnames:

            if var in event_names:
                events = self.behav.events.__dict__[var]
                if events is None:
                    print("empty %s" % var)
                    continue
                var_dict[var] = self.behav.create_event_time_binned(
                    events,
                    time_stamps,
                    t_start=t_start,
                    t_stop=t_stop,
                    select=self.filter,
                )

            elif var in continuous_names:
                continuous = self.behav.continuous.__dict__[var]
                if continuous is None:
                    print("empty %s" % var)
                    continue
                var_dict[var] = self.behav.cut_continuous(
                    continuous,
                    time_stamps,
                    t_start=t_start,
                    t_stop=t_stop,
                    select=self.filter,
                    idx0=None,
                    idx1=None,
                    rebin=rebin,
                )
            elif var == "phase":
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials, dtype=bool)
                phase = self.lfp.extract_phase(
                    all_tr, self.spikes.channel_id, self.spikes.brain_area
                )
                var_dict[var] = self.lfp.cut_phase(
                    phase,
                    time_stamps,
                    t_start=t_start,
                    t_stop=t_stop,
                    select=self.filter,
                    idx0=None,
                    idx1=None,
                )

            elif var == "lfp_beta":
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials, dtype=bool)
                # assert (self.lfp.compute_phase)
                phase = self.lfp.extract_phase_x_unit(
                    self.lfp.lfp_beta,
                    all_tr,
                    self.spikes.channel_id,
                    self.spikes.brain_area,
                )
                var_dict[var] = self.lfp.cut_phase(
                    phase,
                    time_stamps,
                    t_start=t_start,
                    t_stop=t_stop,
                    select=self.filter,
                    idx0=None,
                    idx1=None,
                )
            elif var == "lfp_beta_power":
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials, dtype=bool)
                # assert (self.lfp.compute_phase)
                amplitude = self.lfp.extract_phase_x_unit(
                    self.lfp.lfp_beta_power,
                    all_tr,
                    self.spikes.channel_id,
                    self.spikes.brain_area,
                )
                var_dict[var] = self.lfp.cut_phase(
                    amplitude,
                    time_stamps,
                    t_start=t_start,
                    t_stop=t_stop,
                    select=self.filter,
                    idx0=None,
                    idx1=None,
                )
            elif var == "lfp_alpha":
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials, dtype=bool)
                # assert (self.lfp.compute_phase)
                phase = self.lfp.extract_phase_x_unit(
                    self.lfp.lfp_alpha,
                    all_tr,
                    self.spikes.channel_id,
                    self.spikes.brain_area,
                )
                var_dict[var] = self.lfp.cut_phase(
                    phase,
                    time_stamps,
                    t_start=t_start,
                    t_stop=t_stop,
                    select=self.filter,
                    idx0=None,
                    idx1=None,
                )
            elif var == "lfp_alpha_power":
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials, dtype=bool)
                # assert (self.lfp.compute_phase)
                amplitude = self.lfp.extract_phase_x_unit(
                    self.lfp.lfp_alpha_power,
                    all_tr,
                    self.spikes.channel_id,
                    self.spikes.brain_area,
                )
                var_dict[var] = self.lfp.cut_phase(
                    amplitude,
                    time_stamps,
                    t_start=t_start,
                    t_stop=t_stop,
                    select=self.filter,
                    idx0=None,
                    idx1=None,
                )
            elif var == "lfp_theta":
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials, dtype=bool)
                # assert (self.lfp.compute_phase)
                phase = self.lfp.extract_phase_x_unit(
                    self.lfp.lfp_theta,
                    all_tr,
                    self.spikes.channel_id,
                    self.spikes.brain_area,
                )
                var_dict[var] = self.lfp.cut_phase(
                    phase,
                    time_stamps,
                    t_start=t_start,
                    t_stop=t_stop,
                    select=self.filter,
                    idx0=None,
                    idx1=None,
                )
            elif var == "lfp_theta_power":
                # compute phase using hilbert transform in all trials (the filtering is applied with cut_continous)
                all_tr = np.ones(self.lfp.n_trials, dtype=bool)
                # assert (self.lfp.compute_phase)
                amplitude = self.lfp.extract_phase_x_unit(
                    self.lfp.lfp_theta_power,
                    all_tr,
                    self.spikes.channel_id,
                    self.spikes.brain_area,
                )
                var_dict[var] = self.lfp.cut_phase(
                    amplitude,
                    time_stamps,
                    t_start=t_start,
                    t_stop=t_stop,
                    select=self.filter,
                    idx0=None,
                    idx1=None,
                )
            else:
                raise ValueError("variable %s is unknown" % var)
            if not (
                var
                in [
                    "phase",
                    "lfp_beta",
                    "lfp_alpha",
                    "lfp_theta",
                    "lfp_beta_power",
                    "lfp_theta_power",
                    "lfp_alpha_power",
                ]
            ):
                var_dict[var] = dict_to_vec(var_dict[var])
            else:
                first = True
                for unit in range(var_dict[var].shape[0]):
                    phase = np.hstack(var_dict[var][unit, :])
                    if first:
                        first = False
                        phase_stack = np.zeros((var_dict[var].shape[0], phase.shape[0]))
                    phase_stack[unit, :] = phase
                var_dict[var] = phase_stack

            # check that the variables have same sizes
            if not (
                var
                in [
                    "phase",
                    "lfp_beta",
                    "lfp_alpha",
                    "lfp_theta",
                    "lfp_beta_power",
                    "lfp_theta_power",
                    "lfp_alpha_power",
                ]
            ):
                if var_dict[var].shape[0] != spikes.shape[1]:
                    raise ValueError(
                        "%s counts and spike counts have different sizes" % var
                    )
            else:
                if var_dict[var].shape[1] != spikes.shape[1]:
                    raise ValueError(
                        "%s counts and spike counts have different sizes" % var
                    )

        return spikes, var_dict, trial_idx

    def set_filters(self, *filter_settings):
        # check that the required input is even
        if len(filter_settings) % 2 != 0:
            raise ValueError("Must input a list of field names and input values")
        # list of acceptable field names
        trial_type_list = list(self.info.dytpe_names)
        print(trial_type_list)
        # number of trials
        n_trials = self.behav.n_trials
        filter = np.ones(n_trials, dtype=bool)
        descr = {}
        for k in range(0, len(filter_settings), 2):
            # get the name and check that is valid
            field_name = filter_settings[k]
            if not (field_name in trial_type_list):
                print('Filter not set. Invalid field name: "%s"' % field_name)
                return
            value = filter_settings[k + 1]
            func = self.info.__getattribute__("get_" + field_name)
            if np.isscalar(value):
                filter = filter * func(value)
            else:
                filter = filter * func(*value)

            descr[field_name] = value

        self.filter = filter
        self.filter_descr = descr
        print("Succesfully set filter")
