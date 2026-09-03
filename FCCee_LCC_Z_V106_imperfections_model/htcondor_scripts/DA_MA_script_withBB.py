# ToDos:
# choose correct path (local or htcondor submission)
# choose which seed to load
# define output directory name
# check that output directory does not overwrite files
# specify number of turns for DA and MA
# make sure xutil is up to date
# check that the desired line is loaded
# check the parameters are correctly defined for the case with or without collisions

#path = '/home/larasievert/cernbox'
path = '/eos/user/l/lsievert'  # for htcondor submission

import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import sys
import os
import pickle
# Ensure Python loads the correct xsuite_utilities
xutil_path = f'{path}/08_LCC_V105'
sys.path.insert(0, xutil_path)
import xsuite_utilities as xutil

#seed = 4 # CHOOSE SEED HERE
seed = int(sys.argv[1]) # CHOOSE SEED HERE when running with htcondor submission

line_version = "LCC_106-2-3_z"
output_dir_name = f'OUTPUT_performance_scans/Performance_scans_corrected_lattice_withBB'

# Create the full path to the output directory
output_dir = os.path.join(path, "12_LCC_V106", output_dir_name)
# Create the output directory
os.makedirs(output_dir, exist_ok=True)


# ----------------------------- Define parameters ---------------------------- #
reference_parameters = {
    'normalized_emittance_x': 0.7e-9 * 89236, # tw1.eq_nemitt_x, relativistic gamma factor is 89236
    'normalized_emittance_y': 1.4e-12 * 89236, #tw1.eq_nemitt_y,
    'bunch_length': 16.7e-3, #5.1e-3 when no collisions, 16.7e-3 with collisions
    'bunch_population': 2.02e11}
beam_beam_parameters = {
    'collisions' : True,
    'num_IPs' : 4,
    'half_xing_angle' : 15*1e-3, # half-crossing angle in radians
    'xing_plane' : 0,
    'num_slices' : 251,
    'beamstrahlung_on' : True,
    'binning_mode' : 'unicharge'}

study_param = {}
study_param['number_of_turns'] = 2500 # CHOOSE NUMBER OF TURNS HERE
study_param['ini_cond_nemittance_x'] = 0.7e-9 * 89236 # normalised emittance is geometric emittance times relativistic gamma
study_param['ini_cond_nemittance_y'] = 1.4e-12 * 89236
study_param['ini_cond_type'] = 'grid_MA'
study_param['ini_cond_energy_spread'] = 1.34e-3 # 0.39e-3 when no collisions otherwise 1.34e-3
study_param['ini_cond_bunch_length'] = 16.7e-3 # 5.1mm when no collisions otherwise 16.7mm


# -------------------------------- Corrected line -------------------------------- #
line = xt.Line.from_json(f'{path}/12_LCC_V106/output_optics_correction_old_tolerances_with_corr_limits/{line_version}_line_optics_corrected_seed{seed}.json')
line.cycle(name_first_element='rf400', inplace=True)
line.twiss_default['method'] = '6d'
line.configure_radiation(model='mean', model_beamstrahlung=None)
line.compensate_radiation_energy_loss()

tt = line.get_table()
ttips = tt.rows['ip.*']
ttips = ttips.rows[(ttips.name == 'ipj') | (ttips.name == 'ipa') | (ttips.name == 'ipd') | (ttips.name == 'ipg')] # hardcoded - careful!

bb_elem_names = xutil.install_beam_beam_elements(line, reference_parameters, beam_beam_parameters, ip_list=list(ttips.name))
# Ensure the BB elems are correctly shifted at the IPs (very important!)
tw = line.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20)
xutil.set_BBelem_shift(line, tw)


# ------------ Create context and build tracker and particle grid ------------ #
print('Building tracker and generating particle grid...')
context = xo.ContextCpu() # For CPU
#context_tracking = xo.ContextCpu(omp_num_threads=4)#'auto') # For CPU with activate multi-core CPU parallelisation
line.build_tracker(_context=context)

# Generate particle grid for MA
if study_param['ini_cond_type'] == 'grid_MA':
    delta_initial_values = np.linspace(-0.020, 0.020, 100) # 100 is the number of points along x
    particles_MA, grid_details_MA = xutil.generate_particle_grid(line, study_param, delta_initial_values=delta_initial_values, max_r_y=25)

# Generate particle grid for DA
study_param['ini_cond_type'] = 'grid_DA'
if study_param['ini_cond_type'] == 'grid_DA':
    particles_DA, grid_details_DA = xutil.generate_particle_grid(line, study_param, cartesian_polar='cartesian', max_r_y=70)


# -------------------------------- Turn on BB -------------------------------- #
xutil.set_beam_beam_scale(line, 1, ip_list = list(ttips.name))
print('Beam-beam elements turned on at full strength.')


# -------------------------- Perform the DA/MA scans ------------------------- #
# Set the radiation to 'quantum' for tracking
line.configure_radiation(model='quantum', model_beamstrahlung=None)

line.discard_tracker()
line.build_tracker(_context=context)#_tracking)

# Track particles for DA scan
line.track(particles_DA, num_turns=study_param['number_of_turns'], turn_by_turn_monitor=True, time=True, with_progress=10) 
particles_DA.sort(interleave_lost_particles=True)
if study_param['ini_cond_type'] == 'grid_DA': # num_theta_x_points is the number of points along x
    x_DA, y_DA, where_min_DA=xutil.DA_vs_turns(particles_DA, grid_details_DA['num_r_y_points'], grid_details_DA['num_theta_x_points'], grid_details_DA['x_normalized'], grid_details_DA['y_normalized'], grid_details_DA['delta_init'], output_dir=output_dir, seed=seed)
    
    # Save these values in a pickle file
    DA_results = {
        'x_DA': x_DA.tolist(),
        'y_DA': y_DA.tolist(),
        'where_min_DA': where_min_DA.tolist(),
        'particles_DA': particles_DA.to_dict(),
        'grid_details_DA': grid_details_DA}
    with open(f'{output_dir}/DA_results_seed{seed}.pkl', 'wb') as f:
        import pickle
        pickle.dump(DA_results, f)

print('DA scan completed. Starting MA scan...')

line.track(particles_MA, num_turns=study_param['number_of_turns'], turn_by_turn_monitor=True, time=True, with_progress=10) 
particles_MA.sort(interleave_lost_particles=True)
study_param['ini_cond_type'] = 'grid_MA'

if study_param['ini_cond_type'] == 'grid_MA':
    x_MA, delta_MA, where_min_MA=xutil.MA_vs_turns(particles_MA, grid_details_MA['num_r_y_points'], grid_details_MA['num_delta'], grid_details_MA['x_normalized'], grid_details_MA['y_normalized'], grid_details_MA['delta_init'], output_dir=output_dir, seed=seed)
    
    # Save these values in a pickle file
    MA_results = {
        'x_MA': x_MA.tolist(),
        'delta_MA': delta_MA.tolist(),
        'where_min_MA': where_min_MA.tolist(),
        'particles_MA': particles_MA.to_dict(),
        'grid_details_MA': grid_details_MA}
    with open(f'{output_dir}/MA_results_seed{seed}.pkl', 'wb') as f:
        import pickle
        pickle.dump(MA_results, f)

print('MA scan completed.')