# ToDos:
# choose correct path (local or htcondor submission)
# choose which seed to load
# define output directory names (orbit directory needs to agree with orbit correction script)
# check that output directory does not overwrite files
# check the path towards the reference line and the orbit-corrected line is correct
# make sure xutil and xsuite_optics_imperfections are up to date

#path = '/home/larasievert/cernbox'
path = '/eos/user/l/lsievert' 

# Load all the packages
import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
# Ensure Python loads the correct xsuite_optics_imperfections
xsuite_optics_path = f'{path}/12_LCC_V106'
sys.path.insert(0, xsuite_optics_path)
from xsuite_optics_imperfections import add_correctors, compute_pseudo_inverse, apply_optics_correction
# Ensure Python loads the correct xsuite_utilities
xutil_path = f'{path}/08_LCC_V105'
sys.path.insert(0, xutil_path)
from xsuite_utilities import add_chroma_knobs, match_tune_chroma

#seed = 26 # CHOOSE SEED HERE
seed = int(sys.argv[1]) # CHOOSE SEED HERE when running with htcondor submission

line_version = "LCC_106-2-3_z"
orbit_dir = os.path.join(path, "12_LCC_V106/OUTPUT_orbit_correction")
# Define the output directory for the optics-corrected lines
optics_dir = os.path.join(path, "12_LCC_V106/OUTPUT_optics_correction")
# Create the output directory
os.makedirs(optics_dir, exist_ok=True)

nloops = 5 # Number of iterations for optics correction
epsilon = 0.001 # Threshold for singular values in pseudo-inverse computation

try:
 
    print(f'Starting procedure for seed {seed}...')

    # Load reference line
    line0 = xt.Line.from_json(f'{path}/12_LCC_V106/line_fccee_p_ring_{line_version}_merged_dipoles_with_correctors.json')
    # need to use the reference line with orbit correctors!!!
    line0.cycle(name_first_element='rf400', inplace=True) # cycle to rf cavity
    line0.configure_radiation(model=None, model_beamstrahlung=None) #disable radiation
    line0.twiss_default['method'] = '4d' # switch to 4d Twiss method
    
    tw_ref = line0.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20)
    for col in ['f1001', 'f1010']:
        tw_ref[col + "c"] = np.real(tw_ref[col])
        tw_ref[col + "s"] = np.imag(tw_ref[col])

    # Load orbit-corrected line (cycled to rf cavity, radiation disabled, 4d method)
    line = xt.Line.from_json(f'{orbit_dir}/{line_version}_line_orbit_corrected_seed{seed}.json')
    # just to be safe: 
    line.cycle(name_first_element='rf400', inplace=True) # cycle to rf cavity
    line.configure_radiation(model=None, model_beamstrahlung=None) #disable radiation
    line.twiss_default['method'] = '4d' # switch to 4d Twiss method

    # --------------------------- Add optics correctors -------------------------- #
    tt = line.get_table() # using the table from the orbit-corrected line

    # Filter out normal quads
    ttquad = tt.rows[tt.element_type=='Quadrupole']
    mask = [quad for quad in ttquad.name if line.element_dict[quad].k1s==0 and abs(line.element_dict[quad].k1)>0]
    ttquadno = ttquad.rows[np.isin(ttquad.name, mask)]

    # Filter out normal sexts
    ttsext = tt.rows[tt.element_type=='Sextupole']
    mask = [sext for sext in ttsext.name if line.element_dict[sext].k2s==0 and abs(line.element_dict[sext].k2)>0]
    ttsextno = ttsext.rows[np.isin(ttsext.name, mask)]

    # Add optics correctors to normal quads and normal sexts
    add_correctors(line, ttquadno.name, type='normal', order=1, switch_name='on_qno_corrector')
    add_correctors(line, ttsextno.name, type='skew', order=1, switch_name='on_qsk_corrector')

    # ----------------- Define observation points and observables ---------------- #
    OBSERVATION_POINTS = json.load(open(f'{path}/12_LCC_V106/RM_generation/observation_points.json', 'r'))
    OBSERVABLES_NORMAL = ['mux', 'muy', 'dx']
    OBSERVABLES_SKEW = ['f1001s', 'f1001c', 'f1010s', 'f1010c', 'dy']

    # ----------------------------- Normal correctors ---------------------------- #
    # Load response matrix for normal correctors
    corr_type = 'knl1'
    RM_normal = np.load(f'{path}/12_LCC_V106/RM_generation/RM_{corr_type}_{"_".join(OBSERVABLES_NORMAL)}.npy')
    corr_knob_names_normal = json.load(open(f'{path}/12_LCC_V106/RM_generation/corr_knob_names_{corr_type}_{"_".join(OBSERVABLES_NORMAL)}.json'))

    # Invert RM
    RM_inv_normal, U, S_normal, VT = compute_pseudo_inverse(RM_normal, full_matrices=False, epsilon=epsilon)

    # ------------------------------ Skew correctors ----------------------------- #
    # Load response matrix for skew correctors
    corr_type = 'ksl1'
    RM_skew = np.load(f'{path}/12_LCC_V106/RM_generation/RM_{corr_type}_{"_".join(OBSERVABLES_SKEW)}.npy')
    corr_knob_names_skew = json.load(open(f'{path}/12_LCC_V106/RM_generation/corr_knob_names_{corr_type}_{"_".join(OBSERVABLES_SKEW)}.json'))

    # Invert RM
    RM_inv_skew, U, S_skew, VT = compute_pseudo_inverse(RM_skew, full_matrices=False, epsilon=epsilon)

    # ----------------------- Apply the optics corrections ----------------------- #
    
    add_chroma_knobs(line, optics_type='LCC')

    for _ in range(2):  # Repeat the process twice for better convergence
        #  NORMAL
        apply_optics_correction(line, RM_inv_normal, corr_knob_names_normal, tw_ref=tw_ref, OBSERVATION_POINTS=OBSERVATION_POINTS, OBSERVABLES=OBSERVABLES_NORMAL, nloops=nloops)
        print('Applied optics correction with normal correctors.')

        try: 
            match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d')
            print('Tune matched successfully.')
            
            match_tune_chroma(line, tw_ref, match_quantities='chroma', method='4d')
            print('Chroma matched successfully.')
        except Exception as e:
            print(f'Tune/chroma matching failed due to: {e}')

        # SKEW
        apply_optics_correction(line, RM_inv_skew, corr_knob_names_skew, tw_ref=tw_ref, OBSERVATION_POINTS=OBSERVATION_POINTS, OBSERVABLES=OBSERVABLES_SKEW, nloops=nloops)
        print('Applied optics correction with skew correctors.')

        try: 
            match_tune_chroma(line, tw_ref, match_quantities='tune', method='4d')
            print('Tune matched successfully.')
            
            match_tune_chroma(line, tw_ref, match_quantities='chroma', method='4d')
            print('Chroma matched successfully.')
        except Exception as e:
            print(f'Tune/chroma matching failed due to: {e}')

    # Save the optics-corrected line
    line.to_json(f'{optics_dir}/{line_version}_line_optics_corrected_seed{seed}_without_radiation.json')

    line.twiss_default['method'] = '6d'
    line.configure_radiation(model='mean', model_beamstrahlung=None)
    line.compensate_radiation_energy_loss()

    # Save the optics-corrected line with radiation
    line.to_json(f'{optics_dir}/{line_version}_line_optics_corrected_seed{seed}.json')


except Exception as e:
    print(f'Error occurred for seed {seed}: {e}')            


