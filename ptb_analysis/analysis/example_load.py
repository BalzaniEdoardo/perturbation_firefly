from scipy.io import loadmat

from ptb_analysis.io import fireFly_dataPreproc, load_trial_types

# this loads the matlab struct into python
path = "/Users/ebalzani/Desktop/m53s36.mat"
data = fireFly_dataPreproc(path)

# to access the behavior data for a task variable
# data.behav.continuous.variable_name
rad_vel = data.behav.continuous.rad_vel


# to access the spike times
spike_times = data.spikes.spike_times
print(f"(num units, num trials): {spike_times.shape}")
print(f"Neuron 10, trial 123 - num spikes : {len(spike_times[10, 123])}")

# rad_vel is a dict with keys the trial number.

# if you want to pre-process for TAME-GP
data = data.preProcPGPFA(50)

# the output is stored in
preproc = data.preProcessed

# here the valid trial have been filtered
# the accepted trials are
valid_trials_id = preproc.trialId
print(f"tot valid trials: {len(valid_trials_id)}")

# behavior
proc_rad_vel = preproc.covariates["rad_vel"]

# spike counts (it is compatible with the Poisson GPFA implementation
counts = preproc.data

# this is an array with trials as entries, and values the time series
trial_num = 12
print(f"(time points, ): {proc_rad_vel[trial_num].shape}")
print(f"(num_neurons, time points): {counts[trial_num]['Y'].shape}")


# load the trial types using the code I have
dat = loadmat("/Users/ebalzani/Desktop/m53s36.mat")
info = load_trial_types(dat["behv_stats"].flatten(), dat["trials_behv"].flatten())
trial_type = info.trial_type

# filter valid
filter_trial_type = trial_type[data.filter]
print(filter_trial_type.shape)

print(f"Tot ptb trials: {filter_trial_type['ptb'].sum()}")
