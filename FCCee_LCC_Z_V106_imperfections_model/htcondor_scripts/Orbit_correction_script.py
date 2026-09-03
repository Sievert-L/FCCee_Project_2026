# ToDos:
# choose correct path (local or htcondor submission)
# choose which seed to load and define number of singular values to use in the orbit correction
# define output directory name
# check that output directory does not overwrite files
# check the path towards the reference line is correct
# make sure xutil and xsuite_optics_imperfections are up to date
# check whether using old or updated error tolerances

#path = '/home/larasievert/cernbox'
path = '/eos/user/l/lsievert' 

# Load all the packages
import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import sys
import os
# Ensure Python loads the correct xsuite_optics_imperfections
xsuite_optics_path = f'{path}/12_LCC_V106'
sys.path.insert(0, xsuite_optics_path)
from xsuite_optics_imperfections import find_elements, apply_errors, apply_girder_misalignments, generate_monitor_misalignments, apply_monitor_misalignments, create_elements_switch, apply_orbit_correction
# Ensure Python loads the correct xsuite_utilities
xutil_path = f'{path}/08_LCC_V105'
sys.path.insert(0, xutil_path)
from xsuite_utilities import match_tune_chroma 

#seed = 2 # CHOOSE SEED HERE
seed = int(sys.argv[1]) # CHOOSE SEED HERE when running with htcondor submission
num_sing_vals = 2500 # Choose the number of singular values to use in the orbit correction
corr_limit = 5e-4 # Threshold for the orbit corrector strengths (None or a numerical value)

line_version = "LCC_106-2-3_z"
output_dir_name = f'OUTPUT_orbit_correction' # to save the orbit-corrected lines

# Create the full path to the output directory
output_dir = os.path.join(path, "12_LCC_V106", output_dir_name)
# Create the output directory
os.makedirs(output_dir, exist_ok=True)

print(f'Starting the process for seed {seed}!')

# Load reference line with correctors installed
line = xt.Line.from_json(f'{path}/12_LCC_V106/line_fccee_p_ring_{line_version}_merged_dipoles_with_correctors.json')
line.cycle(name_first_element='rf400', inplace=True) # cycle to rf cavity
line.configure_radiation(model=None, model_beamstrahlung=None) # disable radiation 
line.twiss_default['method'] = '4d' # switch to 4d Twiss method
print(f'Loaded the {line_version} line, cycled it to the RF cavity, disabled radiation, and switched to 4D Twiss.')

tt = line.get_table()
tw_ref = line.twiss()

arc_markers = [
    ('end_ds_start_arc_ipa', 'end_arc_start_ds_ipb'),
    ('end_ds_start_arc_ipb', 'end_arc_start_ds_ipd'),
    ('end_ds_start_arc_ipd', 'end_arc_start_ds_ipf'),
    ('end_ds_start_arc_ipf', 'end_arc_start_ds_ipg'),
    ('end_ds_start_arc_ipg', 'end_arc_start_ds_iph'),
    ('end_ds_start_arc_iph', 'end_arc_start_ds_ipj'),
    ('end_ds_start_arc_ipj', 'end_arc_start_ds_ipl'),
    ('end_ds_start_arc_ipl', 'end_arc_start_ds_ipa')]

DS_markers = [
    ('end_straight_start_ds_ipa', 'end_ds_start_arc_ipa'),
    ('end_arc_start_ds_ipb', 'end_ds_start_straight_ipb'),
    ('end_straight_start_ds_ipb', 'end_ds_start_arc_ipb'),
    ('end_arc_start_ds_ipd', 'end_ds_start_straight_ipd'),
    ('end_straight_start_ds_ipd', 'end_ds_start_arc_ipd'),
    ('end_arc_start_ds_ipf', 'end_ds_start_straight_ipf'),
    ('end_straight_start_ds_ipf', 'end_ds_start_arc_ipf'),
    ('end_arc_start_ds_ipg', 'end_ds_start_straight_ipg'),
    ('end_straight_start_ds_ipg', 'end_ds_start_arc_ipg'),
    ('end_arc_start_ds_iph', 'end_ds_start_straight_iph'),
    ('end_straight_start_ds_iph', 'end_ds_start_arc_iph'),
    ('end_arc_start_ds_ipj', 'end_ds_start_straight_ipj'),
    ('end_straight_start_ds_ipj', 'end_ds_start_arc_ipj'),
    ('end_arc_start_ds_ipl', 'end_ds_start_straight_ipl'),
    ('end_straight_start_ds_ipl', 'end_ds_start_arc_ipl'),
    ('end_arc_start_ds_ipa', 'end_ds_start_straight_ipa')]

straight_markers = [
    ('end_ds_start_straight_ipb', 'end_straight_start_ds_ipb'),
    ('end_ds_start_straight_ipd', 'end_straight_start_ds_ipd'),
    ('end_ds_start_straight_ipf', 'end_straight_start_ds_ipf'),
    ('end_ds_start_straight_ipg', 'end_straight_start_ds_ipg'),
    ('end_ds_start_straight_iph', 'end_straight_start_ds_iph'),
    ('end_ds_start_straight_ipj', 'end_straight_start_ds_ipj'),
    ('end_ds_start_straight_ipl', 'end_straight_start_ds_ipl'),
    ('end_ds_start_straight_ipa', 'end_straight_start_ds_ipa')]

# Split the straight markers into IR and TR markers
IR_markers = [
    ('end_ds_start_straight_ipa', 'end_straight_start_ds_ipa'),
    ('end_ds_start_straight_ipd', 'end_straight_start_ds_ipd'),
    ('end_ds_start_straight_ipg', 'end_straight_start_ds_ipg'),
    ('end_ds_start_straight_ipj', 'end_straight_start_ds_ipj')]

TR_markers = [
    ('end_ds_start_straight_ipb', 'end_straight_start_ds_ipb'),
    ('end_ds_start_straight_ipf', 'end_straight_start_ds_ipf'),
    ('end_ds_start_straight_iph', 'end_straight_start_ds_iph'),
    ('end_ds_start_straight_ipl', 'end_straight_start_ds_ipl')]


arc_quad_names = find_elements(tt, marker_pairs=[arc_markers, DS_markers], element_type='Quadrupole').name # note: include the ds elements with the arc elements
arc_dipole_names = find_elements(tt, marker_pairs=[arc_markers, DS_markers], element_type='RBend').name
arc_sext_names = find_elements(tt, marker_pairs=[arc_markers, DS_markers], element_type='Sextupole').name

# Misaligments in arcs
apply_errors(line=line, pattern=None, seed=seed, sigmas=[50e-6, 50e-6, 100e-6, 50e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_quad_arc',
             element_names=arc_quad_names)

line.vars['on_misalignment_quad_arc'] = 0

apply_errors(line=line, pattern=None, seed=seed, sigmas=[1e-3, 1e-3, 0.5e-3, 1e-3], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_dip_arc',
             element_names=arc_dipole_names)

line.vars['on_misalignment_dip_arc'] = 0

apply_errors(line=line, pattern=None, seed=seed, sigmas=[50e-6, 50e-6, 100e-6, 50e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_sext_arc',
             element_names=arc_sext_names)

line.vars['on_misalignment_sext_arc'] = 0

# Field errors in arcs
_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[1e-3], attrs=['k0'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_dip_arc',
                    element_names=arc_dipole_names)

line.vars['on_field_error_dip_arc'] = 0

_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[2e-4], attrs=['k1'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_quad_arc',
                    element_names=arc_quad_names)

line.vars['on_field_error_quad_arc'] = 0

_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[2e-4], attrs=['k2'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_sext_arc',
                    element_names=arc_sext_names)

line.vars['on_field_error_sext_arc'] = 0

# Girder misalignments
groups = apply_girder_misalignments(line=line, seed=seed, sigmas=[0.15e-3, 0.15e-3, 0.5e-3, 0.15e-3],
                                        attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'],
                                        switch_name='on_misalignment_girder',
                                        line_table=tt, marker_pairs_arcs=[arc_markers, DS_markers])
line.vars['on_misalignment_girder'] = 0


IR_dipole_names = find_elements(tt, marker_pairs=[IR_markers], element_type='RBend').name
TR_dipole_names = find_elements(tt, marker_pairs=[TR_markers], element_type='RBend').name
IR_quad_names = find_elements(tt, marker_pairs=[IR_markers], element_type='Quadrupole').name
FD_quad_names = find_elements(tt, pattern=['qd0a', 'qd0b', 'qd0cr', 'qd0cl', 'qf1a', 'qf1b', 'qf1cr', 'qf1cl', 'qf1dr', 'qf1dl'], element_type='Quadrupole').name
FF_quad_names = find_elements(tt, marker_pairs=[IR_markers], element_type='Quadrupole', except_names=FD_quad_names).name
TR_quad_names = find_elements(tt, marker_pairs=[TR_markers], element_type='Quadrupole').name
IR_sext_names = find_elements(tt, marker_pairs=[IR_markers], element_type='Sextupole').name

# Misaligments in straights
apply_errors(line=line, pattern=None, seed=seed, sigmas=[1e-3, 1e-3, 0.1e-3, 1e-3], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_dip_ir',
             element_names=IR_dipole_names)

line.vars['on_misalignment_dip_ir'] = 0

apply_errors(line=line, pattern=None, seed=seed, sigmas=[1e-3, 1e-3, 0.5e-3, 1e-3], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_dip_tr',
             element_names=TR_dipole_names)

line.vars['on_misalignment_dip_tr'] = 0

apply_errors(line=line, pattern=None, seed=seed, sigmas=[30e-6, 30e-6, 100e-6, 30e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_quad_fd',
             element_names=FD_quad_names)

line.vars['on_misalignment_quad_fd'] = 0

apply_errors(line=line, pattern=None, seed=seed, sigmas=[30e-6, 30e-6, 100e-6, 30e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_quad_ff',
             element_names=FF_quad_names)

line.vars['on_misalignment_quad_ff'] = 0

apply_errors(line=line, pattern=None, seed=seed, sigmas=[100e-6, 100e-6, 100e-6, 100e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_quad_tr',
             element_names=TR_quad_names)

line.vars['on_misalignment_quad_tr'] = 0

apply_errors(line=line, pattern=None, seed=seed, sigmas=[30e-6, 30e-6, 100e-6, 30e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_sext_ir',
             element_names=IR_sext_names)

line.vars['on_misalignment_sext_ir'] = 0

# Field errors in straights
_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[1e-3], attrs=['k0'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_dip_ir',
                    element_names=IR_dipole_names)

line.vars['on_field_error_dip_ir'] = 0

_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[1e-3], attrs=['k0'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_dip_tr',
                    element_names=TR_dipole_names)

line.vars['on_field_error_dip_tr'] = 0

_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[0.5e-4], attrs=['k1'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_quad_fd',
                    element_names=FD_quad_names)

line.vars['on_field_error_quad_fd'] = 0

_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[1e-4], attrs=['k1'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_quad_ff',
                    element_names=FF_quad_names)

line.vars['on_field_error_quad_ff'] = 0

_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[2e-4], attrs=['k1'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_quad_tr',
                    element_names=TR_quad_names)

line.vars['on_field_error_quad_tr'] = 0

_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[1e-4], attrs=['k2'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_sext_ir',
                    element_names=IR_sext_names)

line.vars['on_field_error_sext_ir'] = 0

# Save the line with imperfections switches installed
line.to_json(f'{output_dir}/{line_version}_line_with_imperfections_switches_seed{seed}.json')


# Generating the monitor misalignment dictionary
# Remember to set all the quadrupole misalignment switches to 1 before calling the function!
line.vars['on_misalignment_quad_arc'] = 1
line.vars['on_misalignment_quad_fd'] = 1
line.vars['on_misalignment_quad_ff'] = 1
line.vars['on_misalignment_quad_tr'] = 1
line.vars['on_misalignment_girder'] = 1
monitor_alignment = generate_monitor_misalignments(line, pattern='bpm', attrs=['shift_x', 'shift_y', 'rot_s_rad'], 
                                                    line_table=tt, element_type='Marker')
line.vars['on_misalignment_quad_arc'] = 0
line.vars['on_misalignment_quad_fd'] = 0
line.vars['on_misalignment_quad_ff'] = 0
line.vars['on_misalignment_quad_tr'] = 0
line.vars['on_misalignment_girder'] = 0

# Apply imperfections to the bpms
apply_monitor_misalignments(monitor_alignment, seed, sigmas=[10e-6, 10e-6, 100e-6])

# Set switches for powering the sextupoles and deactivate them

# Arc sextupoles:
create_elements_switch(line, line_table=tt, switch_name='on_powering_sext_arc', marker_pairs=[arc_markers, DS_markers], element_type='Sextupole')
line.vars['on_powering_sext_arc'] = 0

# IR sextupoles excluding crab:
create_elements_switch(line, line_table=tt, switch_name='on_powering_sext_IR', marker_pairs=[IR_markers], element_type='Sextupole', except_pattern=['scrab'])
line.vars['on_powering_sext_IR'] = 0

# Crab sextupoles ('scrab'):
create_elements_switch(line, line_table=tt, switch_name='on_powering_sext_crab', element_type='Sextupole', pattern='scrab')
line.vars['on_powering_sext_crab'] = 0

print('Switches for powering the sextupoles created successfully!')

# Load the steering correctors and monitors required for the correction procedure
steering_monitors=tt.rows['bpm.*']
line.steering_monitors_x = steering_monitors.name
line.steering_monitors_y = steering_monitors.name
print(f'Found', len(line.steering_monitors_x), 'BPMs in the line.')

line.steering_correctors_x = tt.rows['hcor.*'].name
line.steering_correctors_y = tt.rows['vcor.*'].name
print(f'Found', len(line.steering_correctors_x), 'horizontal correctors in the line.')
print(f'Found', len(line.steering_correctors_y), 'vertical correctors in the line.')

# Set chromatic properties to False
line.twiss_default['compute_chromatic_properties'] = False
print('Switched to compute_chromatic_properties=False.')

print('Finished preparing the line with imperfections switches for the correction procedure.')

# Apply all the imperfections

# ARCS
line.vars['on_misalignment_dip_arc'] = 1
line.vars['on_misalignment_quad_arc'] = 1
line.vars['on_misalignment_sext_arc'] = 1

line.vars['on_field_error_dip_arc'] = 1
line.vars['on_field_error_quad_arc'] = 1
line.vars['on_field_error_sext_arc'] = 1

# STRAIGHTS
line.vars['on_misalignment_dip_ir'] = 1
line.vars['on_misalignment_dip_tr'] = 1
line.vars['on_misalignment_quad_fd'] = 1
line.vars['on_misalignment_quad_ff'] = 1
line.vars['on_misalignment_quad_tr'] = 1
line.vars['on_misalignment_sext_ir'] = 1

line.vars['on_field_error_dip_ir'] = 1
line.vars['on_field_error_dip_tr'] = 1
line.vars['on_field_error_quad_fd'] = 1
line.vars['on_field_error_quad_ff'] = 1
line.vars['on_field_error_quad_tr'] = 1
line.vars['on_field_error_sext_ir'] = 1

# GIRDERS
line.vars['on_misalignment_girder'] = 1

print('All imperfections applied.')

# Initial orbit correction (with threading) and tune matching
trial_line = line.copy()
try: 
    apply_orbit_correction(trial_line, tw_ref, monitor_alignment, num_sing_vals=num_sing_vals, corr_limit=corr_limit)
    print(f'Initial orbit correction completed successfully!')
    line = trial_line
except Exception:
    print(f'Initial orbit correction failed.')
    print('Retrying without monitor alignment as argument...')
    trial_line = line.copy()
    apply_orbit_correction(trial_line, tw_ref, num_sing_vals=num_sing_vals, corr_limit=corr_limit)
    line = trial_line

try: 
    match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d')
    print(f'Initial tune matching completed successfully!')
except Exception as e:
    print(f'Initial tune matching failed due to: {e}')

# Arc sextupole ramping
for i in [0.33, 0.66, 0.90, 1]:
    print(f'Ramping up arc sextupoles to {i*100}%...')
    line.vars['on_powering_sext_arc'] = i

    apply_orbit_correction(line, tw_ref, monitor_alignment, num_sing_vals=num_sing_vals, corr_limit=corr_limit)
    print(f'Orbit corrected successfully for arc sextupole ramping up to {i*100}% !')

    try:
        match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d')
        print(f'Tune matched successfully after orbit correction with arc sextupole ramping up to {i*100}% !')
    except Exception as e:
        print(f'Tune matching failed at i={i*100}% during arc sextupole ramping due to: {e}')

# IR sextupole ramping
for i in [0.25, 0.50, 0.75, 0.85, 0.90, 0.95, 1]:
    print(f'Ramping up IR sextupoles to {i*100}%...')
    line.vars['on_powering_sext_IR'] = i
    #line.vars['on_powering_sext_crab'] = i

    apply_orbit_correction(line, tw_ref, monitor_alignment, num_sing_vals=num_sing_vals, corr_limit=corr_limit)
    print(f'Orbit corrected successfully for IR sextupole ramping up to {i*100}% !')

    try: 
        match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d') # tune matching can become difficult for i > 0.90
        print(f'Tune matched successfully after orbit correction with IR sextupole ramping up to {i*100}% !')
    except Exception as e:
        print(f'Tune matching failed at i={i*100}% during IR sextupole ramping due to: {e}')

# Crab sextupole ramping
for i in [0.33, 0.66, 0.90, 1]:
    print(f'Ramping up crab sextupoles to {i*100}%...')
    #line.vars['on_powering_sext_IR'] = i
    line.vars['on_powering_sext_crab'] = i

    apply_orbit_correction(line, tw_ref, monitor_alignment, num_sing_vals=num_sing_vals, corr_limit=corr_limit)
    print(f'Orbit corrected successfully for crab sextupole ramping up to {i*100}% !')

    try: 
        match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d')
        print(f'Tune matched successfully after orbit correction with crab sextupole ramping up to {i*100}% !')
    except Exception as e:
        print(f'Tune matching failed at i={i*100}% during crab sextupole ramping due to: {e}')

# Reintroduce the chromatic properties
line.twiss_default['compute_chromatic_properties'] = True
print('Switched to compute_chromatic_properties=True since all sextupoles are now fully ramped up!')

# Save the orbit-corrected line
line.to_json(f'{output_dir}/{line_version}_line_orbit_corrected_seed{seed}.json')
