#%%
import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import sys
helpers_path = f'../'
sys.path.insert(0, helpers_path)
from helpers_for_imperfections_model import find_elements, apply_errors, apply_girder_misalignments

# Markers to separate the different sections: see lattice_markers.py
from lattice_markers import arc_markers, DS_markers, straight_markers, IR_markers, TR_markers # markers separating the different sections


#%%
####################################
# Settings
seed = 22


#%%
####################################
# Load reference line with correctors installed
print(f'Starting the process for seed {seed}!')
line_version = "LCC_107-0-1_z"
line = xt.Line.from_json('lattices/reference_lattice/fccee_z_with_correctors.json')
line.cycle(name_first_element='rf400', inplace=True) # cycle to rf cavity
line.configure_radiation(model=None, model_beamstrahlung=None) # disable radiation 
line.twiss_default['method'] = '4d' # switch to 4d Twiss method
print(f'Loaded the {line_version} line, cycled it to the RF cavity, disabled radiation, and switched to 4D Twiss.')
tt = line.get_table()
tw_ref = line.twiss() #tw_ref has 4D Twiss


#%%
####################################
# Group all element types based on the markers defined above

arc_quad_names = find_elements(tt, marker_pairs=[arc_markers, DS_markers], element_type='Quadrupole').name # note: include the ds elements with the arc elements
arc_dipole_names = find_elements(tt, marker_pairs=[arc_markers, DS_markers], element_type='RBend').name
arc_sext_names = find_elements(tt, marker_pairs=[arc_markers, DS_markers], element_type='Sextupole').name

IR_dipole_names = find_elements(tt, marker_pairs=[IR_markers], element_type='RBend').name # interaction region dipoles
TR_dipole_names = find_elements(tt, marker_pairs=[TR_markers], element_type='RBend').name # technical region dipoles

IR_quad_names = find_elements(tt, marker_pairs=[IR_markers], element_type='Quadrupole').name # interaction region quadrupoles
FD_quad_names = find_elements(tt, pattern=['qd0a', 'qd0b', 'qd0cr', 'qd0cl', 'qf1a', 'qf1b', 'qf1cr', 'qf1cl', 'qf1dr', 'qf1dl'], element_type='Quadrupole').name # final doublet quadrupoles
FF_quad_names = find_elements(tt, marker_pairs=[IR_markers], element_type='Quadrupole', except_names=FD_quad_names).name # final focusing quadrupoles
TR_quad_names = find_elements(tt, marker_pairs=[TR_markers], element_type='Quadrupole').name # technical region quadrupoles

IR_sext_names = find_elements(tt, marker_pairs=[IR_markers], element_type='Sextupole').name # interaction region sextupoles


#%%
####################################
# Imperfections to arc elements

# Misaligments
# ARC DIPOLES
apply_errors(line=line, pattern=None, seed=seed, sigmas=[1e-3, 1e-3, 0.5e-3, 1e-3], # <-- chosen error tolerances
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], # <-- attributes to apply the errors to
             switch_name='on_misalignment_dip_arc', # <-- switch to turn on/off the errors
             element_names=arc_dipole_names) # <-- lattice elements to apply the errors to
line.vars['on_misalignment_dip_arc'] = 0
# ARC QUADRUPOLES
apply_errors(line=line, pattern=None, seed=seed, sigmas=[50e-6, 50e-6, 100e-6, 50e-6], # <-- chosen error tolerances
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], # <-- attributes to apply the errors to
             switch_name='on_misalignment_quad_arc', # <-- switch to turn on/off the errors
             element_names=arc_quad_names) # <-- lattice elements to apply the errors to
line.vars['on_misalignment_quad_arc'] = 0
# ARC SEXTUPOLES
apply_errors(line=line, pattern=None, seed=seed, sigmas=[50e-6, 50e-6, 100e-6, 50e-6], # <-- chosen error tolerances
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], # <-- attributes to apply the errors to
             switch_name='on_misalignment_sext_arc', # <-- switch to turn on/off the errors
             element_names=arc_sext_names) # <-- lattice elements to apply the errors to
line.vars['on_misalignment_sext_arc'] = 0

# Field errors
# ARC DIPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, sigmas=[1e-3], attrs=['k0'], # <-- chosen error tolerance and attribute to apply the errors to
                    wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_dip_arc', # <-- switch to turn on/off the errors
                    element_names=arc_dipole_names) # <-- lattice elements to apply the errors to
line.vars['on_field_error_dip_arc'] = 0
# ARC QUADRUPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, sigmas=[7*2e-4], attrs=['k1'], # <-- chosen error tolerance and attribute to apply the errors to
                    wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_quad_arc', # <-- switch to turn on/off the errors
                    element_names=arc_quad_names) # <-- lattice elements to apply the errors to
line.vars['on_field_error_quad_arc'] = 0
# ARC SEXTUPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, sigmas=[2e-4], attrs=['k2'], # <-- chosen error tolerance and attribute to apply the errors to
                    wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_sext_arc', # <-- switch to turn on/off the errors
                    element_names=arc_sext_names) # <-- lattice elements to apply the errors to
line.vars['on_field_error_sext_arc'] = 0

# Higher order field errors
# ARC DIPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, sigmas=[10e-4, 5e-4, 2e-4, 0.5e-4, 0.5e-4, 0.1e-4], attrs=['a1', 'b2', 'a2', 'b3', 'b4', 'a4'],
                 reference_radius=1e-2,
                 switch_name='on_mult_error_dip_arc',
                 element_names=arc_dipole_names)
line.vars['on_mult_error_dip_arc'] = 0
# ARC QUADRUPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, sigmas=[61e-4, 23e-4, 12e-4, 4e-4], attrs=['b1', 'a1', 'a2', 'b3'],
                 reference_radius=1e-2,
                 switch_name='on_mult_error_quad_arc',
                 element_names=arc_quad_names)
line.vars['on_mult_error_quad_arc'] = 0

print(f'Done arc imperfections for seed {seed}.')


#%%
####################################
# Imperfections to girders

# Misalignments
groups = apply_girder_misalignments(line=line, seed=seed, sigmas=[0.15e-3, 0.15e-3, 0.5e-3, 0.15e-3], # <-- chosen error tolerances
                                        attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], # <-- attributes to apply the errors to
                                        switch_name='on_misalignment_girder', # <-- switch to turn on/off the errors
                                        line_table=tt, marker_pairs_arcs=[arc_markers]) # <-- filtering via marker pairs to identify the arcs
line.vars['on_misalignment_girder'] = 0

print(f'Done girder imperfections for seed {seed}.')


#%%
####################################
# Imperfections straight sections

# Misaligments in straights
# INTERACTION REGION (IR) DIPOLES ---------------------------> counts as FD -> CS element? For now counting it as a straight dipole
apply_errors(line=line, pattern=None, seed=seed, sigmas=[1e-3, 1e-3, 0.1e-3, 1e-3], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_dip_ir',
             element_names=IR_dipole_names)
line.vars['on_misalignment_dip_ir'] = 0
# TECHNICAL REGION (TR) DIPOLES
apply_errors(line=line, pattern=None, seed=seed, sigmas=[1e-3, 1e-3, 0.1e-3, 1e-3], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_dip_tr',
             element_names=TR_dipole_names)
line.vars['on_misalignment_dip_tr'] = 0
# FINAL DOUBLET (FD) QUADRUPOLES
apply_errors(line=line, pattern=None, seed=seed, sigmas=[30e-6, 30e-6, 100e-6, 30e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_quad_fd',
             element_names=FD_quad_names)
line.vars['on_misalignment_quad_fd'] = 0
# FINAL FOCUSING (FF) QUADRUPOLES
apply_errors(line=line, pattern=None, seed=seed, sigmas=[50e-6, 50e-6, 100e-6, 50e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_quad_ff',
             element_names=FF_quad_names)
line.vars['on_misalignment_quad_ff'] = 0
# TECHNICAL REGION (TR) QUADRUPOLES
apply_errors(line=line, pattern=None, seed=seed, sigmas=[100e-6, 100e-6, 100e-6, 100e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_quad_tr',
             element_names=TR_quad_names)
line.vars['on_misalignment_quad_tr'] = 0
# INTERACTION REGION (IR) SEXTUPOLES ---------------------------> counting as FD->CS element
apply_errors(line=line, pattern=None, seed=seed, sigmas=[50e-6, 50e-6, 100e-6, 50e-6], 
             attrs=['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame'], 
             switch_name='on_misalignment_sext_ir',
             element_names=IR_sext_names)
line.vars['on_misalignment_sext_ir'] = 0

# Field errors in straights
# INTERACTION REGION (IR) DIPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[1e-3], attrs=['k0'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_dip_ir',
                    element_names=IR_dipole_names)
line.vars['on_field_error_dip_ir'] = 0
# TECHNICAL REGION (TR) DIPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[1e-3], attrs=['k0'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_dip_tr',
                    element_names=TR_dipole_names)
line.vars['on_field_error_dip_tr'] = 0
# FINAL DOUBLET (FD) QUADRUPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[0.5e-4], attrs=['k1'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_quad_fd',
                    element_names=FD_quad_names)
line.vars['on_field_error_quad_fd'] = 0
# FINAL FOCUSING (FF) QUADRUPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[1e-4], attrs=['k1'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_quad_ff',
                    element_names=FF_quad_names)
line.vars['on_field_error_quad_ff'] = 0
# TECHNICAL REGION (TR) QUADRUPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[2e-4], attrs=['k1'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_quad_tr',
                    element_names=TR_quad_names)
line.vars['on_field_error_quad_tr'] = 0
# INTERACTION REGION (IR) SEXTUPOLES
_ = apply_errors(line=line, pattern=None, seed=seed, 
                    sigmas=[1e-4], attrs=['k2'], wrt_current_expr=True, # <-- careful, do not apply it twice!
                    apply_relative=True, # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                    switch_name='on_field_error_sext_ir',
                    element_names=IR_sext_names)
line.vars['on_field_error_sext_ir'] = 0

print(f'Done straight section imperfections for seed {seed}.')


#%%
####################################
# Save line with imperfections switches installed
line.to_json(f'lattices/lattices_with_imperfections/{line_version}_line_with_imperfections_switches_seed{seed}.json')
print(f'Line with `imperfections switches saved to {line_version}_line_with_imperfections_switches_seed{seed}.json')

# # Check the switch names
# for switch_name in line.vars.keys():
#     if 'on_' in switch_name:
#         print(line.vars[switch_name])