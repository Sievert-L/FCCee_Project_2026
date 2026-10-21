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
line_version = "LCC_107-0-1_z"


#%%
####################################
# ERROR TABLE  (edit only here)
# Misalignments: sigma_x, sigma_y, sigma_s in [um], sigma_rot in [urad]  (rot_s_rad_no_frame)
# Field error:   relative error on the main strength (k0 / k1 / k2)
# The row name sets the switch names, e.g. 'quad_arc' -> 'on_misalignment_quad_arc', 'on_field_error_quad_arc'
ERROR_TABLE = {
#                 sigma_x  sigma_y  sigma_s  sigma_rot    field   field error
#                 [um]     [um]     [um]     [urad]       attr    [rel.]
    # ARC
    'dip_arc':  [  1000,    1000,    500,     1000,       'k0',   1e-3     ],
    'quad_arc': [    50,      50,    100,       50,       'k1',   7 * 2e-4 ],
    'sext_arc': [    50,      50,    100,       50,       'k2',   2e-4     ],
    'girder':   [   150,     150,    500,      150,       None,   None     ],
    # STRAIGHT SECTION
    'dip_ir':   [  1000,    1000,    100,     1000,       'k0',   1e-3     ],  # FD -> CS element? For now counted as a straight dipole
    'dip_tr':   [  1000,    1000,    100,     1000,       'k0',   1e-3     ],
    'quad_fd':  [    30,      30,    100,       30,       'k1',   0.5e-4   ],
    'quad_ff':  [    50,      50,    100,       50,       'k1',   1e-4     ],  # FF excluding FD
    'quad_tr':  [   100,     100,    100,      100,       'k1',   2e-4     ],
    'sext_ir':  [    50,      50,    100,       50,       'k2',   1e-4     ],  # incl. crab sextupoles
}
 
# Higher-order multipole errors (relative, at the reference radius)
MULTIPOLE_REFERENCE_RADIUS = 1e-2  # [m]
MULTIPOLE_TABLE = {
    'dip_arc':  {'a1': 10e-4, 'b2': 5e-4,  'a2': 2e-4,  'b3': 0.5e-4, 'b4': 0.5e-4, 'a4': 0.1e-4},
    'quad_arc': {'b1': 61e-4, 'a1': 23e-4, 'a2': 12e-4, 'b3': 4e-4},
}
 
# Global scaling factors (1.0 = nominal table)
SCALE_MISALIGNMENT = 1.0
SCALE_FIELD_ERROR  = 1.0
SCALE_MULTIPOLE    = 1.0


#%%
####################################
# Load reference line with correctors installed
print(f'Starting the process for seed {seed}!')
line = xt.Line.from_json('lattices/reference_lattice/fccee_z_with_correctors.json')
line.cycle(name_first_element='rf400', inplace=True) # cycle to rf cavity
line.configure_radiation(model=None, model_beamstrahlung=None) # disable radiation 
line.twiss_default['method'] = '4d' # switch to 4d Twiss method
print(f'Loaded the {line_version} line, cycled it to the RF cavity, disabled radiation, and switched to 4D Twiss.')
tt = line.get_table()
tw_ref = line.twiss() #tw_ref has 4D Twiss


#%%
####################################
# Element names for each row of the table (girder is handled separately, via the arc markers)
 
FD_quad_names = find_elements(tt, pattern=['qd0a', 'qd0b', 'qd0cr', 'qd0cl', 'qf1a', 'qf1b', 'qf1cr', 'qf1cl', 'qf1dr', 'qf1dl'], element_type='Quadrupole').name
 
ELEMENT_NAMES = {
    # ARC (DS elements included with the arc elements)
    'dip_arc':  find_elements(tt, marker_pairs=[arc_markers, DS_markers], element_type='RBend').name, # note: include the ds elements with the arc elements
    'quad_arc': find_elements(tt, marker_pairs=[arc_markers, DS_markers], element_type='Quadrupole').name,
    'sext_arc': find_elements(tt, marker_pairs=[arc_markers, DS_markers], element_type='Sextupole').name,
    # STRAIGHT SECTION
    'dip_ir':   find_elements(tt, marker_pairs=[IR_markers], element_type='RBend').name, # interaction region dipoles
    'dip_tr':   find_elements(tt, marker_pairs=[TR_markers], element_type='RBend').name, # technical region dipoles
    'quad_fd':  FD_quad_names, # interaction region quadrupoles
    'quad_ff':  find_elements(tt, marker_pairs=[IR_markers], element_type='Quadrupole', except_names=FD_quad_names).name, # final doublet quadrupoles
    'quad_tr':  find_elements(tt, marker_pairs=[TR_markers], element_type='Quadrupole').name, # technical region quadrupoles
    'sext_ir':  find_elements(tt, marker_pairs=[IR_markers], element_type='Sextupole').name, # interaction region sextupoles
}


#%%
####################################
# Apply misalignments
um = 1e-6
urad = 1e-6
MISALIGNMENT_ATTRS = ['shift_x', 'shift_y', 'shift_s', 'rot_s_rad_no_frame']
for group, row in ERROR_TABLE.items():
    sigma_x, sigma_y, sigma_s, sigma_rot, field_attr, field_sigma = row
    sigmas = [sigma_x * um * SCALE_MISALIGNMENT,
              sigma_y * um * SCALE_MISALIGNMENT,
              sigma_s * um * SCALE_MISALIGNMENT,
              sigma_rot * urad * SCALE_MISALIGNMENT]
    switch_name = f'on_misalignment_{group}'
    if group == 'girder':
        apply_girder_misalignments(line=line, seed=seed, sigmas=sigmas, 
                                   attrs=MISALIGNMENT_ATTRS, 
                                   switch_name=switch_name, 
                                   line_table=tt, marker_pairs_arcs=[arc_markers]) # <-- filtering via marker pairs to identify the arcs
    else:
        apply_errors(line=line, pattern=None, seed=seed, sigmas=sigmas, # <-- chosen error tolerances
                     attrs=MISALIGNMENT_ATTRS, # <-- attributes to apply the errors to
                     switch_name=switch_name, # <-- switch to turn on/off the errors
                     element_names=ELEMENT_NAMES[group]) # <-- lattice elements to apply the errors to
    line.vars[switch_name] = 0 
print(f'Done misalignments for seed {seed}.')


#%%
####################################
# Apply main field errors (relative)
for group, row in ERROR_TABLE.items():
    sigma_x, sigma_y, sigma_s, sigma_rot, field_attr, field_sigma = row
    if field_attr is None:
        continue
    switch_name = f'on_field_error_{group}'
    apply_errors(line=line, pattern=None, seed=seed,
                 sigmas=[field_sigma * SCALE_FIELD_ERROR], attrs=[field_attr],  # <-- chosen error tolerance and attribute to apply the errors to
                 wrt_current_expr=True,  # <-- careful, do not apply it twice!
                 apply_relative=True,    # <-- to apply the value as a relative change (i.e., multiplied by the existing value)
                 switch_name=switch_name, # <-- switch to turn on/off the errors
                 element_names=ELEMENT_NAMES[group]) # <-- lattice elements to apply the errors to
    line.vars[switch_name] = 0
print(f'Done field errors for seed {seed}.')


#%%
####################################
# Apply higher-order multipole errors
for group, multipoles in MULTIPOLE_TABLE.items():
    attrs = list(multipoles.keys())
    sigmas = [multipoles[attr] * SCALE_MULTIPOLE for attr in attrs]
    switch_name = f'on_mult_error_{group}'
    apply_errors(line=line, pattern=None, seed=seed, sigmas=sigmas, attrs=attrs,
                 reference_radius=MULTIPOLE_REFERENCE_RADIUS,
                 switch_name=switch_name,
                 element_names=ELEMENT_NAMES[group])
    line.vars[switch_name] = 0 
print(f'Done multipole errors for seed {seed}.')


#%%
####################################
# Save line with imperfections switches installed
line.to_json(f'lattices/lattices_with_imperfections/{line_version}_line_with_imperfections_switches_seed{seed}.json')
print(f'Line with `imperfections switches saved to {line_version}_line_with_imperfections_switches_seed{seed}.json')

# # Check the switch names
# for switch_name in line.vars.keys():
#     if 'on_' in switch_name:
#         print(line.vars[switch_name])