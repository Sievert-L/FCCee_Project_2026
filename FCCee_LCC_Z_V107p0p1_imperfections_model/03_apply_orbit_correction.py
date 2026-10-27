#%%
import os
import numpy as np
import xtrack as xt
import json
import sys
helpers_path = f'../'
sys.path.insert(0, helpers_path)
from helpers_for_imperfections_model import generate_monitor_misalignments, apply_monitor_misalignments, create_elements_switch, apply_orbit_correction
from helpers_for_imperfections_model import compute_pseudo_inverse, compute_weighted_pseudo_inverse, build_row_weights, apply_optics_correction
from helpers_from_xutil import match_tune_chroma

# Markers to separate the different sections: see lattice_markers.py
from lattice_markers import arc_markers, DS_markers, straight_markers, IR_markers, TR_markers # markers separating the different sections


#%%
####################################
# Settings
seed = 19
line_version = "LCC_107-0-1_z"

# BPM errors on top of the misalignments inherited from the quadrupoles
bpm_error_sigmas = [10e-6, 10e-6, 500e-6] # [shift_x, shift_y, rot_s_rad]
# Corrector strength limit: with a limit, xtrack's limit-aware solver degrades the orbit from the 2nd iteration on (BPM rms 4 -> 36 um)
corr_limit = None 
# Number of singular values found empirically
num_sing_vals = 1000 # 2500 
# Threading step of the first orbit correction (5 km steps lose the trajectory across the IRs for some seeds)
ds_thread = 1000 # [m]
# SVD cut of the threading (relative to the largest singular value)
# With 1e-7 almost all singular values are kept: pairs of correctors ~0.6 m apart (e.g. hcor_qd20r.2 / hcor_qd10m.4)
# get large opposite kicks (+-300-400 urad) that cancel each other. They are invisible at the BPMs, so the later
# closed-orbit corrections (num_sing_vals) never remove them, and the IR corrector rms is ~56 urad instead of a few urad.
rcond_thread = 1e-4
# BPM reading error of the orbit measurement [m], new value at every orbit measurement (None: no reading error)
ORBIT_MEASUREMENT_ERROR = 1e-6

# Linear optics pre-correction (when sextupoles off): the first steps are done with the vertical tune
# moved up by pre_correction_dqy (tune split 0.04 -> 0.10), see the pre-correction cell. Set pre_correction_dqy = 0 to
# switch it off. Needed for seed 57 (|C-| = 0.068 after the threading, larger than the nominal tune split).
pre_correction_dqy = 0.06
pre_correction_steps_with_split = 3

# Sextupole ramp: (switch, ramp steps), applied in this order
sextupole_ramp = [
    ('on_powering_sext_arc',  [0.33, 0.66, 0.90, 1]),
    ('on_powering_sext_IR',   [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1]), # fine steps: each step adds a large coupling
    ('on_powering_sext_crab', [0.33, 0.66, 0.90, 1])
]

# Measurement errors for the linear optics corrections during the ramp (same values as in notebook 04)
use_measurement_errors = True
MEASUREMENT_ERRORS_SKEW = [12.5e-4, 12.5e-4, 12.5e-4, 12.5e-4, 10e-6] # f1001s, f1001c, f1010s, f1010c, dy
MEASUREMENT_ERRORS_PHASE = [1e-4, 1e-4] # mux, muy


#%%
####################################
# Load lines

# Reference line (with correctors): cycled to the rf cavity, no radiation, 4d twiss
line = xt.Line.from_json('lattices/reference_lattice/fccee_z_with_correctors.json')
line.cycle(name_first_element='rf400', inplace=True)
line.configure_radiation(model=None, model_beamstrahlung=None)
line.twiss_default['method'] = '4d'
tt = line.get_table()
tw_ref = line.twiss() # Reference twiss: for the orbit correction and tune matching
tw_ref_optics = line.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20) # Reference twiss with the coupling RDTs: for the linear optics corrections
for col in ['f1001', 'f1010']:
    tw_ref_optics[col + "c"] = np.real(tw_ref_optics[col])
    tw_ref_optics[col + "s"] = np.imag(tw_ref_optics[col])

# Line with the imperfection switches (already cycled to the rf cavity, no radiation, 4d twiss; see notebook 02)
line = xt.Line.from_json(f'lattices/lattices_with_imperfections/{line_version}_line_with_imperfections_switches_seed{seed}.json')


#%%
####################################
# Prepare the line for the correction

# BPM misalignments
# The BPMs inherit the misalignments of the quadrupoles they are attached to (incl. the girders the quadrupoles sit on):
# switch these on, read them, switch them off again
quad_and_girder_switches = ['on_misalignment_quad_arc', 'on_misalignment_quad_fd', 'on_misalignment_quad_ff',
                            'on_misalignment_quad_tr', 'on_misalignment_girder']
for switch_name in quad_and_girder_switches:
    line.vars[switch_name] = 1
monitor_alignment = generate_monitor_misalignments(line, pattern='bpm', attrs=['shift_x', 'shift_y', 'rot_s_rad'],
                                                   line_table=tt, element_type='Marker')
for switch_name in quad_and_girder_switches:
    line.vars[switch_name] = 0
# Additional BPM errors on top of the misalignments inherited from the quadrupoles
monitor_alignment = apply_monitor_misalignments(monitor_alignment, seed, sigmas=bpm_error_sigmas)

# Sextupole powering switches (for the ramp), all sextupoles switched off
create_elements_switch(line, line_table=tt, switch_name='on_powering_sext_arc', marker_pairs=[arc_markers, DS_markers], element_type='Sextupole')
create_elements_switch(line, line_table=tt, switch_name='on_powering_sext_IR', marker_pairs=[IR_markers], element_type='Sextupole', except_pattern=['scrab'])
create_elements_switch(line, line_table=tt, switch_name='on_powering_sext_crab', element_type='Sextupole', pattern='scrab')
line.vars['on_powering_sext_arc'] = 0
line.vars['on_powering_sext_IR'] = 0
line.vars['on_powering_sext_crab'] = 0

# Monitors and correctors used by the orbit correction (regex patterns)
line.steering_monitors_x = tt.rows['bpm.*'].name
line.steering_monitors_y = tt.rows['bpm.*'].name
line.steering_correctors_x = tt.rows['hcor.*'].name
line.steering_correctors_y = tt.rows['vcor.*'].name

# No chromatic properties in the twiss while the sextupoles are (partly) off
line.twiss_default['compute_chromatic_properties'] = False

# Switch on all the imperfections
imperfection_switches = [
    # arcs
    'on_misalignment_dip_arc', 'on_misalignment_quad_arc', 'on_misalignment_sext_arc',
    'on_field_error_dip_arc', 'on_field_error_quad_arc', 'on_field_error_sext_arc',
    'on_mult_error_dip_arc', 'on_mult_error_quad_arc',
    # straights
    'on_misalignment_dip_ir', 'on_misalignment_dip_tr', 'on_misalignment_quad_fd', 'on_misalignment_quad_ff',
    'on_misalignment_quad_tr', 'on_misalignment_sext_ir',
    'on_field_error_dip_ir', 'on_field_error_dip_tr', 'on_field_error_quad_fd', 'on_field_error_quad_ff',
    'on_field_error_quad_tr', 'on_field_error_sext_ir',
    # girders
    'on_misalignment_girder']
for switch_name in imperfection_switches:
    line.vars[switch_name] = 1


#%%
####################################
# Linear optics corrections used during the orbit correction and sextupole ramp
# With all imperfections, the machine after the first orbit correction has ~10-20% beta-beating and |C-| ~ 0.04,
# comparable to the fractional tune split (|0.16 - 0.20| = 0.04). Then the tunes cannot be matched and the orbit/optics
# response matrices are not valid. So the coupling and beta-beating are corrected at every step of the ramp.
# Only phases and coupling RDTs are used (no dispersion): until all sextupoles are on, the chromaticity is huge and the
# "dispersion" is dominated by the energy dependence of the orbit, which the linear RMs cannot describe.
# Dispersion is corrected in notebook 04.

OBSERVATION_POINTS = json.load(open('helpers_for_RM_generation/observation_points.json', 'r'))
RM_PHASE_ORIGIN = 'ipa' # first element of the line the RMs were built on

# Coupling: skew RM (Dy rows with weight 0), weighted and column-normalised pseudo-inverse
OBSERVABLES_SKEW = ['f1001s', 'f1001c', 'f1010s', 'f1010c', 'dy']
RM_skew = np.load(f'helpers_for_RM_generation/RM_ksl1_{"_".join(OBSERVABLES_SKEW)}.npy')
corr_knob_names_skew = json.load(open(f'helpers_for_RM_generation/corr_knob_names_ksl1_{"_".join(OBSERVABLES_SKEW)}.json'))
row_weights_skew = build_row_weights(OBSERVATION_POINTS, OBSERVABLES_SKEW, observable_weights={'dy': 0.0})
RM_inv_skew, S_skew, n_kept_skew = compute_weighted_pseudo_inverse(RM_skew, row_weights_skew, epsilon=1e-3)

# Beta-beating: phase rows of the normal RM (the rows are ordered mux block, muy block, dx block)
OBSERVABLES_PHASE = ['mux', 'muy']
RM_normal = np.load('helpers_for_RM_generation/RM_knl1_mux_muy_dx.npy')
corr_knob_names_normal = json.load(open('helpers_for_RM_generation/corr_knob_names_knl1_mux_muy_dx.json'))
RM_phase = RM_normal[:2*len(OBSERVATION_POINTS)]
RM_inv_phase, U, S_phase, VT = compute_pseudo_inverse(RM_phase, full_matrices=False, epsilon=1e-3)
del RM_normal, U, VT

# Each correction call gets its own measurement-noise seed (a fixed seed would repeat the same noise at every call,
# acting as a systematic error). Unique per machine seed, per call, and per correction type (offset).
meas_call_counter = 0
def next_meas_seed(offset):
    global meas_call_counter
    meas_call_counter += 1
    return 1000000 * seed + 10 * meas_call_counter + offset

def correct_coupling(line, nloops=2):
    c_minus_before = line.twiss4d().c_minus
    apply_optics_correction(line, RM_inv_skew, corr_knob_names_skew, tw_ref=tw_ref_optics,
                            OBSERVATION_POINTS=OBSERVATION_POINTS, OBSERVABLES=OBSERVABLES_SKEW,
                            nloops=nloops, row_weights=row_weights_skew,
                            measurement_errors=MEASUREMENT_ERRORS_SKEW if use_measurement_errors else None,
                            meas_seed=next_meas_seed(offset=1))
    c_minus_after = line.twiss4d().c_minus
    print(f'Coupling correction: |C-| {c_minus_before:.4f} -> {c_minus_after:.4f}')

def correct_beta(line, nloops=2):
    apply_optics_correction(line, RM_inv_phase, corr_knob_names_normal, tw_ref=tw_ref_optics,
                            OBSERVATION_POINTS=OBSERVATION_POINTS, OBSERVABLES=OBSERVABLES_PHASE,
                            nloops=nloops, phase_origin=RM_PHASE_ORIGIN,
                            measurement_errors=MEASUREMENT_ERRORS_PHASE if use_measurement_errors else None,
                            meas_seed=next_meas_seed(offset=2))

def correct_linear_optics(line, nloops=2):
    # alternate coupling and beta-beating corrections
    for _ in range(nloops):
        correct_coupling(line, nloops=1)
        correct_beta(line, nloops=1)


#%%
####################################
# Initial orbit correction (sextupoles off): threading only
# Threading only: the closed-orbit correction is done after the linear optics pre-correction (next cell), because with
# the optics still far from the model (10-20% beta-beating, |C-|~0.04) the design response matrix makes it diverge.
# In the past, in a  few cases the threading fails because of the strong perturbations --> then retry without the monitor misalignments.
# Not real, and does not happen with the current model; kept for safety.
trial_line = line.copy()
try:
    apply_orbit_correction(trial_line, tw_ref, monitor_alignment, corr_limit=corr_limit, ds_thread=ds_thread, thread_only=True,
                           rcond_short=rcond_thread, rcond_long=rcond_thread) # SVD cut of the threading
    line = trial_line
except Exception:
    print('Threading failed. Retrying without monitor alignment...')
    trial_line = line.copy()
    apply_orbit_correction(trial_line, tw_ref, corr_limit=corr_limit, ds_thread=ds_thread, thread_only=True,
                           rcond_short=rcond_thread, rcond_long=rcond_thread) # SVD cut of the threading
    line = trial_line


#%%
####################################
# Linear optics pre-correction (sextupoles off) and initial tune matching
# 5 steps of coupling + beta-beating correction, each followed by a tune matching.
# The first steps are done with the vertical tune moved up (tune split 0.04 -> 0.10): right after
# the threading, |C-| can already be larger than the nominal split (seed 57: 0.068). Then the tunes cannot be matched, the coupling RDTs saturate
# (f1001 ~ 0.5) and the coupling correction diverges. With the larger split the tunes can be matched, the RDTs are smaller
# than in the RM model, and the correction converges (it undershoots instead of overshooting).
for k in range(5):
    qy_target = tw_ref.qy + (pre_correction_dqy if k < pre_correction_steps_with_split else 0) # [Change 14.10.2026]
    try: # [Change 14.10.2026] tune matching also before the correction (to the split tunes in the first steps)
        match_tune_chroma(line, {'qx': tw_ref.qx, 'qy': qy_target}, match_quantities='tune', method='4d')
    except Exception as e:
        print(f'Pre-correction step {k+1}: tune matching (before) failed due to: {e}')
    correct_linear_optics(line, nloops=1)
    try:
        match_tune_chroma(line, {'qx': tw_ref.qx, 'qy': qy_target}, match_quantities='tune', method='4d') # [Change 14.10.2026] target qy_target
    except Exception as e:
        print(f'Pre-correction step {k+1}: tune matching failed due to: {e}')

# Closed-orbit correction, now that the optics is close to the model
apply_orbit_correction(line, tw_ref, monitor_alignment, num_sing_vals=num_sing_vals, corr_limit=corr_limit,
                       bpm_noise=ORBIT_MEASUREMENT_ERROR, noise_seed=next_meas_seed(offset=3),
                       rcond_short=rcond_thread, rcond_long=rcond_thread)  # [Change 14.10.2026] same SVD cut if it falls back to threading

# Initial tune matching
try:
    correct_linear_optics(line, nloops=2)
    match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d')
except Exception as e:
    print(f'Initial tune matching failed due to: {e}')


#%%
####################################
# Sextupole ramp
# At every step: orbit correction, tune matching, linear optics correction, tune matching
# The tunes are matched right after the orbit correction, before the coupling correction: the beam
# passes the IR sextupoles (beta_y ~ 9 km) ~100 um off-centre, and the feed-down of each IR step can shift Qy by up to ~-0.03
# (seen with seed 9). The tune split |Qx-Qy| (0.04) then drops to <0.01, the coupling RDTs blow up and the coupling RM
# is no longer valid: the coupling correction diverges. With the tunes back at the nominal values the RM is valid.
for switch_name, ramp_steps in sextupole_ramp:
    for step in ramp_steps:
        print(f'---------- {switch_name} = {step}')
        line.vars[switch_name] = step
        apply_orbit_correction(line, tw_ref, monitor_alignment, num_sing_vals=num_sing_vals, corr_limit=corr_limit,
                               bpm_noise=ORBIT_MEASUREMENT_ERROR, noise_seed=next_meas_seed(offset=3),
                               rcond_short=rcond_thread, rcond_long=rcond_thread)  # [Change 14.10.2026] same SVD cut if it falls back to threading
        try: # [Change 14.10.2026] tune matching right after the orbit correction (see the comment above)
            match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d')
        except Exception as e:
            print(f'Tune matching before the optics correction failed at {switch_name} = {step} due to: {e}')
        try:
            correct_linear_optics(line, nloops=2)
            match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d')
        except Exception as e:
            print(f'Tune matching failed at {switch_name} = {step} due to: {e}')


#%%
####################################
# Save the orbit-corrected line
line.twiss_default['compute_chromatic_properties'] = True # All sextupoles on: chromatic properties back in the twiss
if not os.path.exists(f'lattices/lattices_with_corrected_imperfections/01_orbit_corrected_only/'):
    os.makedirs(f'lattices/lattices_with_corrected_imperfections/01_orbit_corrected_only/')
line.to_json(f'lattices/lattices_with_corrected_imperfections/01_orbit_corrected_only/{line_version}_line_orbit_corrected_seed{seed}.json')