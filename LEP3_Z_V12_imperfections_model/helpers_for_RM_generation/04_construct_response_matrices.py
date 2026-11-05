import json
import xtrack as xt
import numpy as np
import sys
import glob
sys.path.append('../../')
from helpers_for_imperfections_model import _load_twiss, _construct_vector, build_response_matrix

OBSERVATION_POINTS = json.load(open(f'observation_points.json', 'r'))
OBSERVABLES_NORMAL = ['mux', 'muy', 'dx']
OBSERVABLES_SKEW = ['f1001s', 'f1001c', 'f1010s', 'f1010c', 'dy']

# We need to generate vector0 that is used for the construction of the response matrix
# Reference line with the optics correctors (from notebook 01), cycled to rf800.0
line = xt.Line.from_json(f'../lattices/reference_lattice/LEP3_Z_v12_wMarkers_wBPMs_wCorr_wTrims.json')
tw = line.twiss4d(coupling_edw_teng=True, matrix_stability_tol=0.20)
for col in ['f1001', 'f1010']:
    tw[col + "c"] = np.real(tw[col])
    tw[col + "s"] = np.imag(tw[col])
tw = tw.cols[["name", 's', 'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy', 'mux', 'muy', 'f1001c', 'f1001s', 'f1010c', 'f1010s']]

# Build response matrix (for normal correctors)
tw0 = _load_twiss(twiss=tw, OBSERVABLES=OBSERVABLES_NORMAL, OBSERVATION_POINTS=OBSERVATION_POINTS)
vector0 = _construct_vector(tw0, OBSERVABLES=OBSERVABLES_NORMAL)
twiss_outputs_files = sorted(glob.glob(f'twiss_outputs/twiss*knl1*.csv'))
RM, corr_knob_names, dk_list = build_response_matrix(twiss_outputs_files, OBSERVATION_POINTS, OBSERVABLES_NORMAL, vector0, print_progress=True)
np.save(f'RM_knl1_{"_".join(OBSERVABLES_NORMAL)}.npy', RM)
with open(f'corr_knob_names_knl1_{"_".join(OBSERVABLES_NORMAL)}.json', 'w') as f:
    json.dump(corr_knob_names, f)
with open(f'dk_list_knl1_{"_".join(OBSERVABLES_NORMAL)}.json', 'w') as f:
    json.dump(dk_list, f)
print(f'Normal RM: {RM.shape}')

# Build response matrix (for skew correctors)
tw0 = _load_twiss(twiss=tw, OBSERVABLES=OBSERVABLES_SKEW, OBSERVATION_POINTS=OBSERVATION_POINTS)
vector0 = _construct_vector(tw0, OBSERVABLES=OBSERVABLES_SKEW)
twiss_outputs_files = sorted(glob.glob(f'twiss_outputs/twiss*ksl1*.csv'))
RM, corr_knob_names, dk_list = build_response_matrix(twiss_outputs_files, OBSERVATION_POINTS, OBSERVABLES_SKEW, vector0, print_progress=True)
np.save(f'RM_ksl1_{"_".join(OBSERVABLES_SKEW)}.npy', RM)
with open(f'corr_knob_names_ksl1_{"_".join(OBSERVABLES_SKEW)}.json', 'w') as f:
    json.dump(corr_knob_names, f)
with open(f'dk_list_ksl1_{"_".join(OBSERVABLES_SKEW)}.json', 'w') as f:
    json.dump(dk_list, f)
print(f'Skew RM: {RM.shape}')
