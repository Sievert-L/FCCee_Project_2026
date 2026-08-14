import xtrack as xt
import json

line_version = 'LCC_106-2-3_z'
line = xt.Line.from_json(f'../lattices/reference_lattice_LCC_V106/line_fccee_{line_version}_with_qno_qsk.json')

tt = line.get_table()
ttobs = tt.rows['bpm.*|ip.*']
obs_points = list(ttobs.name)

with open(f'observation_points.json', 'w') as f:
    json.dump(obs_points, f)
print('Observation points saved to observation_points.json')