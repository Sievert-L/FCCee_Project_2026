import xtrack as xt
import numpy as np
import json
import xobjects as xo

path2output = 'twiss_outputs/'
line_version = 'LCC_106-2-3_z'

line = xt.Line.from_json(f'line_fccee_{line_version}_with_qno_qsk.json')
line.build_tracker(_context=xo.ContextCpu(omp_num_threads=8))

DELTAK = 1e-5
OBSERVATION_POINTS = json.load(open(f'observation_points.json', 'r'))

vartable = line.vars.get_table()
qno_vars = list(vartable.rows['knl1_.*'].name)
qsk_vars = list(vartable.rows['ksl1_.*'].name)
VARS2SET = qno_vars + qsk_vars
total_vars = len(VARS2SET)

for i, VAR2SET in enumerate(VARS2SET, 1):
    print(f"Processing variable {i}/{total_vars}: {VAR2SET}")

    line[VAR2SET] = DELTAK

    tw = line.twiss4d(coupling_edw_teng=True, matrix_stability_tol=0.20)
    tw = tw.cols[("name", 's', 'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy', 'mux', 'muy', 'f1001', 'f1010')] # <-- for the twiss outputs I keep all observables
    for col in ['f1001', 'f1010']:
        tw[col + "c"] = np.real(tw[col])
        tw[col + "s"] = np.imag(tw[col])
    tw = tw.cols[["name", 's', 'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy', 'mux', 'muy', 'f1001c', 'f1001s', 'f1010c', 'f1010s']]

    tw = tw.rows[OBSERVATION_POINTS]

    tw_name = f'twiss_{int(DELTAK*1e7)}e-7_{VAR2SET}'
    tw.to_csv(path2output+tw_name+'.csv')
    #tw.to_csv(tw_name+'.csv')
    
    line[VAR2SET] = 0