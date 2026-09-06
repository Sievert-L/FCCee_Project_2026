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
output_dir_name = f'output_performance_scans/Performance_scans_corrected_lattice_withBB_andPC'

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


# -------------------------------- Ideal line -------------------------------- #
line0 = xt.Line.from_json(f'{path}/12_LCC_V106/line_fccee_p_ring_{line_version}_merged_dipoles_with_correctors.json')

line0.cycle(name_first_element='rf400', inplace=True)
line0.twiss_default['method'] = '6d'
line0.configure_radiation(model='mean')
line0.compensate_radiation_energy_loss()

tt0 = line0.get_table()
ttips0 = tt0.rows['ip.*']
ttips0 = ttips0.rows[(ttips0.name == 'ip.0') | (ttips0.name == 'ip.2') | (ttips0.name == 'ip.4') | (ttips0.name == 'ip.6')] # hardcoded - careful!

#Install beam-beam elements
bb_elem_names0 = xutil.install_beam_beam_elements(line0, reference_parameters, beam_beam_parameters, ip_list=list(ttips0.name))
xutil.set_beam_beam_scale(line0, 0.0, ip_list = list(ttips0.name))
# Ensure the BB elems are correctly shifted at the IPs (very important!)
tw0 = line0.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20)
xutil.set_BBelem_shift(line0, tw0)

ip_optics0 = tw0.rows[ttips0.name].cols['betx', 'bety', 'dx', 'dy', 'x', 'y', 'mux', 'muy', 'alfx', 'alfy', 'f1001', 'f1010']


# -------------------------------- Corrected line -------------------------------- #
line = xt.Line.from_json(f'{path}/12_LCC_V106/output_optics_correction_old_tolerances_with_corr_limits/{line_version}_line_optics_corrected_seed{seed}.json')
line.cycle(name_first_element='rf400', inplace=True)
line.twiss_default['method'] = '6d'
line.configure_radiation(model='mean', model_beamstrahlung=None)
line.compensate_radiation_energy_loss()

tt = line.get_table()
ttips = tt.rows['ip.*']
ttips = ttips.rows[(ttips.name == 'ip.0') | (ttips.name == 'ip.2') | (ttips.name == 'ip.4') | (ttips.name == 'ip.6')] # hardcoded - careful!

bb_elem_names = xutil.install_beam_beam_elements(line, reference_parameters, beam_beam_parameters, ip_list=list(ttips.name))
# Ensure the BB elems are correctly shifted at the IPs (very important!)
tw = line.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20)
xutil.set_BBelem_shift(line, tw)

ip_optics_beforePC = tw.rows[ttips.name].cols['betx', 'bety', 'dx', 'dy', 'x', 'y', 'mux', 'muy', 'alfx', 'alfy', 'f1001', 'f1010']


# -------------------- Phase correction matching procedure ------------------- #
print('Starting phase correction matching...')
# Install additional variables in line
beta_matching_quads = ['qd10m', 'qf9m', 'qd8m', 'qd6m', 'qf5m', 'qd4m', 'qf3m', 'qd2m', 'qf1m']
quad_numbers_ordered = {'.6': ttips0.name[0]+'_l', '': ttips0.name[0]+'_r', 
                        '.0': ttips0.name[1]+'_l', '.1': ttips0.name[1]+'_r', 
                        '.2': ttips0.name[2]+'_l', '.3': ttips0.name[2]+'_r', 
                        '.4': ttips0.name[3]+'_l', '.5': ttips0.name[3]+'_r'}
all_new_vars = []
all_quad_vals = []
all_initial_expr = []
for quad in beta_matching_quads:
    for num,val in quad_numbers_ordered.items():
        #print('Quad ', quad+num, ' value is ', quad_value)
        new_var = 'k'+quad+'_'+val
        #print(new_var)
        quad_value = line.element_refs[quad+num].k1._value
        quad_expr = line.element_refs[quad+num].k1._expr
        all_new_vars.append(new_var)
        all_quad_vals.append(quad_value)
        all_initial_expr.append(quad_expr)
        line.vars[new_var] = quad_value
        line.element_refs[quad+num].k1 = line.vars[new_var]

line.configure_radiation(model=None) #<-- disable radiation since below we use 4d twiss; don't forget to enable it after and call compensate_radiation_energy()

# Do the phase correction matching
opt = line.match(
    method = '4d',# <-- careful: 4d ! Remove radiation first
    solve = True,
    restore_if_fail = False, # beamline is restored to its initial state if the matching fails
    vary=[
        xt.VaryList([
            # All magnets independent (9 x 2 x 4 = 72 variables)
            'kqd10m_ip.6_l', 'kqf9m_ip.6_l', 'kqd8m_ip.6_l', 'kqd6m_ip.6_l', 'kqf5m_ip.6_l', 'kqd4m_ip.6_l', 'kqf3m_ip.6_l', 'kqd2m_ip.6_l', 'kqf1m_ip.6_l', # left of ip.6
            'kqd10m_ip.6_r', 'kqf9m_ip.6_r', 'kqd8m_ip.6_r', 'kqd6m_ip.6_r', 'kqf5m_ip.6_r', 'kqd4m_ip.6_r', 'kqf3m_ip.6_r', 'kqd2m_ip.6_r', 'kqf1m_ip.6_r', # right of ip.6
            'kqd10m_ip.0_l', 'kqf9m_ip.0_l', 'kqd8m_ip.0_l', 'kqd6m_ip.0_l', 'kqf5m_ip.0_l', 'kqd4m_ip.0_l', 'kqf3m_ip.0_l', 'kqd2m_ip.0_l', 'kqf1m_ip.0_l', # left of ip.0
            'kqd10m_ip.0_r', 'kqf9m_ip.0_r', 'kqd8m_ip.0_r', 'kqd6m_ip.0_r', 'kqf5m_ip.0_r', 'kqd4m_ip.0_r', 'kqf3m_ip.0_r', 'kqd2m_ip.0_r', 'kqf1m_ip.0_r', # right of ip.0
            'kqd10m_ip.2_l', 'kqf9m_ip.2_l', 'kqd8m_ip.2_l', 'kqd6m_ip.2_l', 'kqf5m_ip.2_l', 'kqd4m_ip.2_l', 'kqf3m_ip.2_l', 'kqd2m_ip.2_l', 'kqf1m_ip.2_l', # left of ip.2
            'kqd10m_ip.2_r', 'kqf9m_ip.2_r', 'kqd8m_ip.2_r', 'kqd6m_ip.2_r', 'kqf5m_ip.2_r', 'kqd4m_ip.2_r', 'kqf3m_ip.2_r', 'kqd2m_ip.2_r', 'kqf1m_ip.2_r', # right of ip.2
            'kqd10m_ip.4_l', 'kqf9m_ip.4_l', 'kqd8m_ip.4_l', 'kqd6m_ip.4_l', 'kqf5m_ip.4_l', 'kqd4m_ip.4_l', 'kqf3m_ip.4_l', 'kqd2m_ip.4_l', 'kqf1m_ip.4_l', # left of ip.4
            'kqd10m_ip.4_r', 'kqf9m_ip.4_r', 'kqd8m_ip.4_r', 'kqd6m_ip.4_r', 'kqf5m_ip.4_r', 'kqd4m_ip.4_r', 'kqf3m_ip.4_r', 'kqd2m_ip.4_r', 'kqf1m_ip.4_r', # right of ip.4
            ],
            step=1e-8, tag='phase_corr'),
    ],
    targets = [
        # --- Inter-IP phase advances (your 8 targets) ---
        xt.Target('mux', tw0.rows[ttips0.name[0]].mux[0], at=ttips0.name[0], tol=1e-6),
        xt.Target('mux', tw0.rows[ttips0.name[1]].mux[0], at=ttips0.name[1], tol=1e-6),
        xt.Target('mux', tw0.rows[ttips0.name[2]].mux[0], at=ttips0.name[2], tol=1e-6),
        xt.Target('mux', tw0.rows[ttips0.name[3]].mux[0], at=ttips0.name[3], tol=1e-6),
        xt.Target('muy', tw0.rows[ttips0.name[0]].muy[0], at=ttips0.name[0], tol=1e-6),
        xt.Target('muy', tw0.rows[ttips0.name[1]].muy[0], at=ttips0.name[1], tol=1e-6),
        xt.Target('muy', tw0.rows[ttips0.name[2]].muy[0], at=ttips0.name[2], tol=1e-6),
        xt.Target('muy', tw0.rows[ttips0.name[3]].muy[0], at=ttips0.name[3], tol=1e-6),

        # --- beta* at the IPs (protect luminosity) ---
        xt.Target('betx', tw0.rows[ttips0.name[0]].betx[0], at=ttips0.name[0], tol=1e-4),
        xt.Target('betx', tw0.rows[ttips0.name[1]].betx[0], at=ttips0.name[1], tol=1e-4),
        xt.Target('betx', tw0.rows[ttips0.name[2]].betx[0], at=ttips0.name[2], tol=1e-4),
        xt.Target('betx', tw0.rows[ttips0.name[3]].betx[0], at=ttips0.name[3], tol=1e-4),
        xt.Target('bety', tw0.rows[ttips0.name[0]].bety[0], at=ttips0.name[0], tol=1e-6),
        xt.Target('bety', tw0.rows[ttips0.name[1]].bety[0], at=ttips0.name[1], tol=1e-6),
        xt.Target('bety', tw0.rows[ttips0.name[2]].bety[0], at=ttips0.name[2], tol=1e-6),
        xt.Target('bety', tw0.rows[ttips0.name[3]].bety[0], at=ttips0.name[3], tol=1e-6),

        # --- alfx, alfy = 0 at IPs (waist condition) ---
        xt.Target('alfx', tw0.rows[ttips0.name[0]].alfx[0], at=ttips0.name[0], tol=1e-4),
        xt.Target('alfx', tw0.rows[ttips0.name[1]].alfx[0], at=ttips0.name[1], tol=1e-4),
        xt.Target('alfx', tw0.rows[ttips0.name[2]].alfx[0], at=ttips0.name[2], tol=1e-4),
        xt.Target('alfx', tw0.rows[ttips0.name[3]].alfx[0], at=ttips0.name[3], tol=1e-4),
        xt.Target('alfy', tw0.rows[ttips0.name[0]].alfy[0], at=ttips0.name[0], tol=1e-4),
        xt.Target('alfy', tw0.rows[ttips0.name[1]].alfy[0], at=ttips0.name[1], tol=1e-4),
        xt.Target('alfy', tw0.rows[ttips0.name[2]].alfy[0], at=ttips0.name[2], tol=1e-4),
        xt.Target('alfy', tw0.rows[ttips0.name[3]].alfy[0], at=ttips0.name[3], tol=1e-4),

        # --- dx, dy = 0 at IPs (waist condition) ---
        # xt.Target('dx', tw0.rows[ttips0.name[0]].dx[0], at=ttips0.name[0], tol=1e-4),
        # xt.Target('dx', tw0.rows[ttips0.name[1]].dx[0], at=ttips0.name[1], tol=1e-4),
        # xt.Target('dx', tw0.rows[ttips0.name[2]].dx[0], at=ttips0.name[2], tol=1e-4),
        # xt.Target('dx', tw0.rows[ttips0.name[3]].dx[0], at=ttips0.name[3], tol=1e-4),
        # xt.Target('dy', tw0.rows[ttips0.name[0]].dy[0], at=ttips0.name[0], tol=1e-4),
        # xt.Target('dy', tw0.rows[ttips0.name[1]].dy[0], at=ttips0.name[1], tol=1e-4),
        # xt.Target('dy', tw0.rows[ttips0.name[2]].dy[0], at=ttips0.name[2], tol=1e-4),
        # xt.Target('dy', tw0.rows[ttips0.name[3]].dy[0], at=ttips0.name[3], tol=1e-4),

        # --- Global tunes (prevent tune drift) ---
        xt.Target('qx', tw0.qx, tol=1e-6),
        xt.Target('qy', tw0.qy, tol=1e-6),
])
print('Phase correction matching finished.')


# ------------------------- Finish setup of the line ------------------------- #
# Reintroduce radiation
line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

# Check twiss after the rematching 
# Optics corrections should have degraded only by a bit
tw_afterPC = line.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20)
ip_optics_afterPC = tw_afterPC.rows[ttips0.name].cols['betx', 'bety', 'dx', 'dy', 'x', 'y', 'mux', 'muy', 'alfx', 'alfy', 'f1001', 'f1010'] 

# Save the ip optics in a pickle file
ip_optics_results = {
    'ip_optics0': ip_optics0.to_dict(),
    'ip_optics_beforePC': ip_optics_beforePC.to_dict(), 
    'ip_optics_afterPC': ip_optics_afterPC.to_dict(),
}
with open(f'{output_dir}/ip_optics_results_seed{seed}.pkl', 'wb') as f:
    import pickle
    pickle.dump(ip_optics_results, f)

# Save line after phase correction
line.to_json(f'{output_dir}/{line_version}_line_phase_corrected_seed{seed}.json')


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