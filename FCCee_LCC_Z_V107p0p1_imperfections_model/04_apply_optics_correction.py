#%%
import os
import numpy as np
import xtrack as xt
import json
import sys
helpers_path = f'../'
sys.path.insert(0, helpers_path)
from helpers_for_imperfections_model import compute_pseudo_inverse, compute_weighted_pseudo_inverse, build_row_weights, apply_optics_correction
from helpers_for_imperfections_model import generate_monitor_misalignments, apply_monitor_misalignments, apply_orbit_correction # [Change 14.10.2026] for the orbit feedback
from helpers_from_xutil import add_chroma_knobs, match_tune_chroma
from plotting_helpers import plot_ring_optics, plot_ip_optics


#%%
####################################
# Settings
seed = 19
line_version = "LCC_107-0-1_z"

# Observables (and measurement errors) of the normal and skew corrections
OBSERVATION_POINTS = json.load(open(f'helpers_for_RM_generation/observation_points.json', 'r'))
OBSERVABLES_NORMAL = ['mux', 'muy', 'dx']
OBSERVABLES_SKEW = ['f1001s', 'f1001c', 'f1010s', 'f1010c', 'dy']
use_measurement_errors = True
MEASUREMENT_ERRORS_NORMAL = [1e-4, 1e-4, 10e-6]
MEASUREMENT_ERRORS_SKEW = [12.5e-4, 12.5e-4, 12.5e-4, 12.5e-4, 10e-6]

# Thresholds for the singular values of the RM pseudo-inverses
# (epsilon_normal = 1e-4 is safe only because notebook 03 delivers a pre-corrected machine, ~2% beta-beating)
epsilon_normal = 1e-4
epsilon_skew = 1e-4

# Number of correction passes (normal + skew), and of iterations per correction
npasses = 2
nloops = 5
# Final normal correction after the last skew correction (removes the beta-beating brought back by the orbit feedback)
final_normal_correction = True

# The RMs were built on the un-cycled reference line (helpers_for_RM_generation), which starts at 'ipa'.
# The phases (mux, muy) depend on where the line starts, so they are re-measured from 'ipa' inside the correction
# (beta, dispersion and the coupling RDTs do not depend on the starting point).
# To be improved...
RM_PHASE_ORIGIN = 'ipa'

# 'Orbit feedback' during the optics correction (same BPM errors and orbit correction settings as in notebook 03).
# Before every optics measurement, the orbit is corrected again if the rms BPM reading has drifted above the tolerance.
bpm_error_sigmas = [10e-6, 10e-6, 500e-6] # [shift_x, shift_y, rot_s_rad], on top of the quadrupole misalignments
ORBIT_MEASUREMENT_ERROR = 1e-6 # BPM reading error of the orbit measurement [m]
num_sing_vals = 1000 #2500
ORBIT_FEEDBACK_TOLERANCE = 2e-6 # [m] rms BPM reading above which the orbit is corrected again


#%%
####################################
# Load the reference line and the orbit-corrected line

# Reference line: cycled to the rf cavity, no radiation, 4d twiss
line0 = xt.Line.from_json('lattices/reference_lattice/fccee_z_with_correctors.json')
line0.cycle(name_first_element='rf400', inplace=True)
line0.configure_radiation(model=None, model_beamstrahlung=None)
line0.twiss_default['method'] = '4d'

# Reference twiss with the coupling RDTs
tw_ref = line0.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20)
for col in ['f1001', 'f1010']:
    tw_ref[col + "c"] = np.real(tw_ref[col])
    tw_ref[col + "s"] = np.imag(tw_ref[col])

# Orbit-corrected line (from notebook 03), same settings
line = xt.Line.from_json(f'lattices/lattices_with_corrected_imperfections/01_orbit_corrected_only/{line_version}_line_orbit_corrected_seed{seed}.json')
line.cycle(name_first_element='rf400', inplace=True)
line.configure_radiation(model=None, model_beamstrahlung=None)
line.twiss_default['method'] = '4d'
tw_orbit_corrected = line.twiss(coupling_edw_teng=True)


#%%
####################################
# Orbit feedback during the optics correction
# The optics correctors kick the beam where it passes off-centre (skew trims in sextupoles with a horizontal offset
# kick vertically, normal trims in misaligned quadrupoles kick in both planes), so every correction step moves the orbit.
# In the IR sextupoles (beta_y ~ 9 km) a few um of orbit change already changes the coupling (|C-| ~ 2.5e-4 per um)
# and Dy through feed-down. Without re-correcting the orbit, the optics iterations can diverge (seeds 20 and 30: the
# vertical orbit drifts by 100s of um at the IR sextupoles, f1001 and Dy grow at every iteration until the twiss fails).
# The orbit is corrected only when the BPM readings have drifted (dead band): re-correcting at every iteration would
# move the orbit at the IR sextupoles randomly (1 um BPM reading error), which changes the coupling by ~1e-3 each time.

# BPM misalignments, as in notebook 03 (all imperfection switches are on in this line)
tt = line.get_table()
monitor_alignment = generate_monitor_misalignments(line, pattern='bpm', attrs=['shift_x', 'shift_y', 'rot_s_rad'],
                                                   line_table=tt, element_type='Marker')
monitor_alignment = apply_monitor_misalignments(monitor_alignment, seed, sigmas=bpm_error_sigmas)
bpm_names = list(monitor_alignment.keys())
bpm_shift_x = np.array([monitor_alignment[name]['shift_x'] for name in bpm_names])
bpm_shift_y = np.array([monitor_alignment[name]['shift_y'] for name in bpm_names])

# Monitors and correctors used by the orbit correction
line.steering_monitors_x = tt.rows['bpm.*'].name
line.steering_monitors_y = tt.rows['bpm.*'].name
line.steering_correctors_x = tt.rows['hcor.*'].name
line.steering_correctors_y = tt.rows['vcor.*'].name
tw_ref_orbit = line0.twiss() # reference for the orbit response matrix (4d, as in notebook 03)

orbit_feedback_calls = 0
def orbit_feedback(line):
    global orbit_feedback_calls
    tw = line.twiss(compute_chromatic_properties=False)
    bpm_reading_x = tw.rows[bpm_names].x - bpm_shift_x
    bpm_reading_y = tw.rows[bpm_names].y - bpm_shift_y
    rms_x = np.sqrt(np.mean(bpm_reading_x**2))
    rms_y = np.sqrt(np.mean(bpm_reading_y**2))
    if max(rms_x, rms_y) > ORBIT_FEEDBACK_TOLERANCE:
        orbit_feedback_calls += 1
        print(f'Orbit feedback: rms BPM reading x/y = {rms_x*1e6:.1f}/{rms_y*1e6:.1f} um -> orbit correction')
        apply_orbit_correction(line, tw_ref_orbit, monitor_alignment, num_sing_vals=num_sing_vals,
                               bpm_noise=ORBIT_MEASUREMENT_ERROR, noise_seed=1000000*seed + 500000 + orbit_feedback_calls)


#%%
####################################
# Response matrices and their pseudo-inverses

# Normal correctors (quadrupole trims)
RM_normal = np.load(f'helpers_for_RM_generation/RM_knl1_{"_".join(OBSERVABLES_NORMAL)}.npy')
corr_knob_names_normal = json.load(open(f'helpers_for_RM_generation/corr_knob_names_knl1_{"_".join(OBSERVABLES_NORMAL)}.json'))
RM_inv_normal, U, S_normal, VT = compute_pseudo_inverse(RM_normal, full_matrices=False, epsilon=epsilon_normal)
del U, VT

# Skew correctors (skew trims in the sextupoles): weighted, column-normalised pseudo-inverse
RM_skew = np.load(f'helpers_for_RM_generation/RM_ksl1_{"_".join(OBSERVABLES_SKEW)}.npy')
corr_knob_names_skew = json.load(open(f'helpers_for_RM_generation/corr_knob_names_ksl1_{"_".join(OBSERVABLES_SKEW)}.json'))
# Row weights: Dy rows x10 relative to the coupling RDT rows.
# IP rows NOT up-weighted (ip_weight=1): with measurement errors, the few IP rows (2-3 per IP, no real BPM there)
# would be fitted almost exactly, i.e. the IP coupling would be pinned to the noise of those rows.
row_weights_skew = build_row_weights(OBSERVATION_POINTS, OBSERVABLES_SKEW, observable_weights={'dy': 10.0}, ip_weight=1.0)
RM_inv_skew, S_skew, n_kept_skew = compute_weighted_pseudo_inverse(RM_skew, row_weights_skew, epsilon=epsilon_skew)


#%%
####################################
# Optics correction

# Knobs for the chromaticity matching
add_chroma_knobs(line, optics_type='LCC')

def match_tune_and_chroma(line):
    # try/except: the matching may fail if the optics is still too distorted
    try:
        match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d')
        match_tune_chroma(line, tw_ref, match_quantities='chroma', method='4d')
    except Exception as e:
        print(f'Tune/chroma matching failed due to: {e}')

# Each pass: normal correction, tune/chroma matching, skew correction, tune/chroma matching
for ii in range(npasses):
    apply_optics_correction(line, RM_inv_normal, corr_knob_names_normal, tw_ref=tw_ref,
                            OBSERVATION_POINTS=OBSERVATION_POINTS, OBSERVABLES=OBSERVABLES_NORMAL, nloops=nloops,
                            measurement_errors=MEASUREMENT_ERRORS_NORMAL if use_measurement_errors else None,
                            meas_seed=10000*seed + 10*ii + 1,
                            phase_origin=RM_PHASE_ORIGIN, orbit_feedback=orbit_feedback) # [Change 14.10.2026] orbit feedback
    match_tune_and_chroma(line)

    apply_optics_correction(line, RM_inv_skew, corr_knob_names_skew, tw_ref=tw_ref,
                            OBSERVATION_POINTS=OBSERVATION_POINTS, OBSERVABLES=OBSERVABLES_SKEW, nloops=nloops,
                            measurement_errors=MEASUREMENT_ERRORS_SKEW if use_measurement_errors else None,
                            meas_seed=10000*seed + 10*ii + 2,
                            row_weights=row_weights_skew,
                            phase_origin=RM_PHASE_ORIGIN, orbit_feedback=orbit_feedback) # [Change 14.10.2026] orbit feedback
    match_tune_and_chroma(line)
    print(f'Done correction pass {ii+1}/{npasses}.')

# Final normal correction: the orbit feedback during the last skew correction moves the orbit in the IR sextupoles,
# and the feed-down brings back some beta-beating (seed 20: 1.8% rms in y and beta*_y 3% low, instead of 0.45%).
# Without orbit feedback: the steps are small here, and an orbit correction would change the coupling and Dy again
# (with it, seed 30 ended with dDy = 4.9 mm instead of 0.3 mm).
if final_normal_correction:
    apply_optics_correction(line, RM_inv_normal, corr_knob_names_normal, tw_ref=tw_ref,
                            OBSERVATION_POINTS=OBSERVATION_POINTS, OBSERVABLES=OBSERVABLES_NORMAL, nloops=nloops,
                            measurement_errors=MEASUREMENT_ERRORS_NORMAL if use_measurement_errors else None,
                            meas_seed=10000*seed + 10*npasses + 1,
                            phase_origin=RM_PHASE_ORIGIN)
    match_tune_and_chroma(line)
    print('Done final normal correction.')


#%%
###################################
# Save the imperfections and the corrections (json), so that the corrected machine can be rebuilt from the ideal line
#
# To rebuild: load the ideal line (reference_lattice/fccee_z_with_correctors.json), cycle it to 'rf400', add_chroma_knobs(line, 'LCC'), then
#   corrections json:
#     'other_knobs'       : line.vars[name] = value (tune and chroma knobs, sextupole powering switches)
#     'optics_correctors' : line.vars[name] = value (knl1_<quad>: normal trims, ksl1_<sext>: skew trims)
#     'orbit_correctors'  : kicks of hcor_<quad> (knl[0]) and vcor_<quad> (ksl[0]); in the ideal line they are driven by
#                           line.vars['knl_<quad>'] and line.vars['ksl_<quad>']
#   imperfections json (values with all the imperfection switches as in 'switches'):
#     'misalignments'         : element attributes set to these values (shift_x, shift_y, shift_s, rot_s_rad_no_frame)
#     'field_errors_relative' : k -> k * (1 + value) (k0, k1, k2)
#     'multipole_errors'      : added to the element knl[n] and ksl[n]

errors_and_corrections_dir = 'lattices/lattices_with_corrected_imperfections/02_orbit_and_optics_corrected/imperfections_and_corrections'
os.makedirs(errors_and_corrections_dir, exist_ok=True)
tt = line.get_table()

# Imperfections
# Switches of the error families (created in 02)
imperfection_switches = [nn for nn in line.vars.get_table().name if nn.startswith(('on_misalignment', 'on_field_error', 'on_mult_error'))]
switch_values = {nn: line.vars[nn]._value for nn in imperfection_switches}
# Elements that can carry imperfections (magnets) and their attributes
magnet_names = [nn for nn, element_type in zip(tt.name, tt.element_type) if element_type in ['Bend', 'RBend', 'Quadrupole', 'Sextupole', 'Octupole']]
MISALIGNMENT_ATTRS = ['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame']
FIELD_ATTRS = ['k0', 'k1', 'k2']
def read_magnet_values(line, names):
    values = {}
    for nn in names:
        element = line[nn]
        values[nn] = {'misalignment': {aa: float(getattr(element, aa)) for aa in MISALIGNMENT_ATTRS},
                      'field': {aa: float(getattr(element, aa)) for aa in FIELD_ATTRS if hasattr(element, aa)},
                      'knl': np.array(element.knl, dtype=float),
                      'ksl': np.array(element.ksl, dtype=float)}
    return values
# The imperfections are the difference between the magnets with the imperfection switches on and off
values_on = read_magnet_values(line, magnet_names)
for nn in imperfection_switches:
    line.vars[nn] = 0
values_off = read_magnet_values(line, magnet_names)
for nn in imperfection_switches:
    line.vars[nn] = switch_values[nn] # restore the switches

imperfections = {'seed': seed, 'line_version': line_version, 'switches': switch_values,
                 'misalignments': {}, 'field_errors_relative': {}, 'multipole_errors': {}}
for nn in magnet_names:
    on = values_on[nn]
    off = values_off[nn]
    misalignment = {aa: on['misalignment'][aa] - off['misalignment'][aa] for aa in MISALIGNMENT_ATTRS
                    if on['misalignment'][aa] != off['misalignment'][aa]}
    if len(misalignment) > 0:
        imperfections['misalignments'][nn] = misalignment
    field_error = {aa: on['field'][aa] / off['field'][aa] - 1 for aa in on['field']
                   if on['field'][aa] != off['field'][aa]}
    if len(field_error) > 0:
        imperfections['field_errors_relative'][nn] = field_error
    dknl = on['knl'] - off['knl']
    dksl = on['ksl'] - off['ksl']
    if np.any(dknl != 0) or np.any(dksl != 0):
        imperfections['multipole_errors'][nn] = {'knl': dknl.tolist(), 'ksl': dksl.tolist()}

# Corrections
corrections = {'seed': seed, 'line_version': line_version,
               'orbit_correctors': {}, 'optics_correctors': {}, 'other_knobs': {}}

# Orbit correctors: kicks of the corrector elements
for nn in tt.name:
    if nn.startswith('hcor_') and line[nn].knl[0] != 0:
        corrections['orbit_correctors'][nn] = float(line[nn].knl[0])
    if nn.startswith('vcor_') and line[nn].ksl[0] != 0:
        corrections['orbit_correctors'][nn] = float(line[nn].ksl[0])

# Optics correctors and other knobs: all the variables that differ from the ideal line (line0)
tv0 = line0.vars.get_table()
values_ideal = dict(zip(tv0.name, tv0.value))
tv = line.vars.get_table()
for nn, value in zip(tv.name, tv.value):
    if not isinstance(value, (int, float, np.integer, np.floating)):
        continue
    if nn in imperfection_switches:
        continue # in the imperfections json
    if nn.startswith(('knl_', 'ksl_', 'orbit_corr_')):
        continue # orbit correctors, already saved as kicks above
    if nn in values_ideal and values_ideal[nn] == value:
        continue # unchanged
    if nn.startswith(('knl1_', 'ksl1_')):
        corrections['optics_correctors'][nn] = float(value)
    else:
        corrections['other_knobs'][nn] = float(value)

json.dump(imperfections, open(f'{errors_and_corrections_dir}/{line_version}_imperfections_seed{seed}.json', 'w'), indent=1)
json.dump(corrections, open(f'{errors_and_corrections_dir}/{line_version}_corrections_seed{seed}.json', 'w'), indent=1)
print(f"Saved {len(imperfections['misalignments'])} misaligned, {len(imperfections['field_errors_relative'])} field-error and "
      f"{len(imperfections['multipole_errors'])} multipole-error magnets; {len(corrections['orbit_correctors'])} orbit correctors, "
      f"{len(corrections['optics_correctors'])} optics correctors, other knobs: {list(corrections['other_knobs'])}")


#%%
###################################
# Save the optics-corrected lines and the final Twiss (6d, with radiation)

lines_dir = 'lattices/lattices_with_corrected_imperfections/02_orbit_and_optics_corrected'
twiss_dir = f'{lines_dir}/twiss'
os.makedirs(twiss_dir, exist_ok=True)

# Without radiation (4d Twiss kept only for the summary of run_all_seeds.py)
line.twiss_default['method'] = '4d'
tw_after_optics_corr = line.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20)
#line.to_json(f'{lines_dir}/{line_version}_line_optics_corrected_seed{seed}_without_radiation.json')

# With radiation and tapering
line.twiss_default['method'] = '6d'
line.configure_radiation(model='mean', model_beamstrahlung=None)
line.compensate_radiation_energy_loss()
line.to_json(f'{lines_dir}/{line_version}_line_optics_corrected_seed{seed}.json')

# Final Twiss: 6d, with radiation, with the equilibrium emittances (eneloss_and_damping=True)
tw_after_optics_corr_6d = line.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20, eneloss_and_damping=True)

# Saved as HDF5 (read with xt.TwissTable.from_hdf5; needs h5py), with the columns below (~8 MB).
# The scalars (qx, qy, dqx, dqy, c_minus, eq_gemitt_x, eq_gemitt_y, ...) are always saved.
TWISS_COLUMNS = ['s', 'x', 'px', 'y', 'py', 'zeta', 'delta',
                 'betx', 'bety', 'alfx', 'alfy', 'mux', 'muy',
                 'dx', 'dy', 'dpx', 'dpy', 'ddx', 'ddy', 'ddpx', 'ddpy',
                 'f1001', 'f1010', 'betx1', 'bety1', 'betx2', 'bety2',
                 'wx_chrom', 'wy_chrom', 'ax_chrom', 'ay_chrom', 'bx_chrom', 'by_chrom']
tw_after_optics_corr_6d.cols[TWISS_COLUMNS].to_hdf5(f'{twiss_dir}/{line_version}_twiss_optics_corrected_seed{seed}_6d_with_radiation.h5')
print(f'Equilibrium emittances: eps_x = {tw_after_optics_corr_6d.eq_gemitt_x*1e9:.3f} nm, eps_y = {tw_after_optics_corr_6d.eq_gemitt_y*1e12:.3f} pm')


#%%
####################################
# Plots

tws = {
    #'Orbit': tw_orbit_corrected,
    #'Optics': tw_after_optics_corr,
    'Optics (6D)': tw_after_optics_corr_6d,
}
fig, axs = plot_ring_optics(tws, tw_ref=tw_ref)
# fig.savefig('orbit_optics_correction.png', dpi=400)

fig, axs = plot_ip_optics(tws, tw_ref=tw_ref,
                          ips=['ip.6', 'ip.0', 'ip.2', 'ip.4'],
                          ip_labels=['IP1', 'IP2', 'IP3', 'IP4'])