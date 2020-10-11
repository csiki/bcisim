import load_local_neo_odml_elephant

import os, sys

import numpy as np
from scipy import stats
import quantities as pq
import matplotlib.pyplot as plt
import pickle

from matplotlib import gridspec, ticker

from reachgraspio import reachgraspio

import odml.tools

import neo_utils
import odml_utils


def get_monkey_datafile(monkey):
    if monkey == "Lilou":
        return "l101210-001"  # ns2 (behavior) and ns5 present
    elif monkey == "Nikos2":
        return "i140703-001"  # ns2 and ns6 present
    else:
        return ""


# Enter your dataset directory here
datasetdir = "../datasets/"

trialtype_colors = {
    'SGHF': 'MediumBlue', 'SGLF': 'Turquoise',
    'PGHF': 'DarkGreen', 'PGLF': 'YellowGreen',
    'LFSG': 'Orange', 'LFPG': 'Yellow',
    'HFSG': 'DarkRed', 'HFPG': 'OrangeRed',
    'SGSG': 'SteelBlue', 'PGPG': 'LimeGreen',
    'NONE': 'k', 'PG': 'k', 'SG': 'k', 'LF': 'k', 'HF': 'k'}

event_colors = {
    'TS-ON': 'Gray',  # 'TS-OFF': 'Gray',
    'WS-ON': 'Gray',  # 'WS-OFF': 'Gray',
    'CUE-ON': 'Gray',
    'CUE-OFF': 'Gray',
    'GO-ON': 'Gray',  # 'GO-OFF': 'Gray',
    #    'GO/RW-OFF': 'Gray',
    'SR': 'Gray',  # 'SR-REP': 'Gray',
    'RW-ON': 'Gray',  # 'RW-OFF': 'Gray',
    'STOP': 'Gray'}


def force_aspect(ax, aspect=1):
    ax.set_aspect(abs(
        (ax.get_xlim()[1] - ax.get_xlim()[0]) /
        (ax.get_ylim()[1] - ax.get_ylim()[0])) / aspect)


def get_arraygrid(blackrock_elid_list, chosen_el, rej_el=None):
    if rej_el is None:
        rej_el = []
    array_grid = np.zeros((10, 10))
    for m in range(10):
        for n in range(10):
            idx = (9 - m) * 10 + n
            bl_id = blackrock_elid_list[idx]
            if bl_id == -1:
                array_grid[m, n] = 0.7
            elif bl_id == chosen_el:
                array_grid[m, n] = -0.7
            elif bl_id in rej_el:
                array_grid[m, n] = -0.35
            else:
                array_grid[m, n] = 0
    return np.ma.array(array_grid, mask=np.isnan(array_grid))


# CHANGE this parameter to load data of the different monkeys
# monkey = 'Nikos2'
monkey = sys.argv[1] if len(sys.argv) > 1 else 'Lilou'

nsx_lfp = {'Lilou': 2, 'Nikos2': 2}
nsx_raw = {'Lilou': 5, 'Nikos2': 6}
datafile = get_monkey_datafile(monkey)

session = reachgraspio.ReachGraspIO(
    filename=os.path.join(datasetdir, datafile),
    odml_directory=datasetdir,
    verbose=False)

bl = session.read_block(
    index=None,
    name=None,
    description=None,
    nsx_to_load=nsx_lfp[monkey],
    n_starts=None,
    n_stops=None,
    # channels=[66] + [141, 143],  # for testing
    channels=range(1, 97) + [141, 143],
    units=range(1, 5),
    load_waveforms=False,
    load_events=True,
    scaling='voltage',
    lazy=False,
    cascade=True)

seg = bl.segments[0]

# get start and stop events of trials
start_events = neo_utils.get_events(
    seg, properties={
        'name': 'TrialEvents',
        'trial_event_labels': 'GO-ON',
        'performance_in_trial': session.performance_codes['correct_trial']})
stop_events = neo_utils.get_events(
    seg, properties={
        'name': 'TrialEvents',
        'trial_event_labels': 'RW-ON',
        'performance_in_trial': session.performance_codes['correct_trial']})

neo_utils.add_epoch(
    seg,
    start_events[0],
    stop_events[0],
    pre=0 * pq.ms,
    post=0 * pq.ms,
    segment_type='complete_trials',
    trialtype=start_events[0].annotations['belongs_to_trialtype'])

epochs = neo_utils.get_epochs(seg, properties={'segment_type': 'complete_trials'})
cut_segments = neo_utils.cut_segment_by_epoch(seg, epochs[0], reset_time=True)
# explicitely adding trial type annotations to cut segments
for i, cut_seg in enumerate(cut_segments):
    cut_seg.annotate(trialtype=epochs[0].annotations['trialtype'][i])

# store data in np arrays; analog behavioral signals are sampled at 1k, while spike time points refer to 30k Hz
annotations_2_save = {'channel_id', 'connector_aligned_id', 'unit_id', 'SNR', 'spike_amplitude',
                      'trial_event_labels', 'belongs_to_trialtype'}

spike_trains, spike_meta = [], []
analog_signals, analog_meta = [], []
events, event_meta = [], []
for i, cut_seg in enumerate(cut_segments):
    cut_spike_trains = [np.array(sp) for sp in cut_seg.spiketrains]
    cut_sp_meta = [{k: v for k, v in sp.annotations.items() if k in annotations_2_save} for sp in cut_seg.spiketrains]

    cut_analog = [np.array(an) for an in cut_seg.analogsignals]
    cut_an_meta = [{k: v for k, v in an.annotations.items() if k in annotations_2_save} for an in cut_seg.analogsignals]

    cut_events = [np.array(ev) for ev in cut_seg.events]
    cut_ev_meta = [{k: v for k, v in ev.annotations.items() if k in annotations_2_save} for ev in cut_seg.events]

    spike_trains.append(cut_spike_trains)
    spike_meta.append(cut_sp_meta)
    analog_signals.append(cut_analog)
    analog_meta.append(cut_an_meta)
    events.append(cut_events)
    event_meta.append(cut_ev_meta)

content = {'spike_trains': spike_trains, 'spike_meta': spike_meta,
           'analog_signals': analog_signals, 'analog_meta': analog_meta,
           'events': events, 'event_meta': event_meta,
           'blackrock_elid_list': bl.annotations['avail_electrode_ids']}

with open('{}.pckl'.format(monkey), mode='wb') as f:
    pickle.dump(content, f, protocol=2)
