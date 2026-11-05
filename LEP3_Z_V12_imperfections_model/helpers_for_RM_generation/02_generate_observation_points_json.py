import xtrack as xt
import json

# Reference line with the optics correctors (from notebook 01), cycled to rf800.0
line = xt.Line.from_json(f'../lattices/reference_lattice/LEP3_Z_v12_wMarkers_wBPMs_wCorr_wTrims.json')

tt = line.get_table()
ttobs = tt.rows['bpm.*|ip.*']
obs_points = list(ttobs.name)

with open(f'observation_points.json', 'w') as f:
    json.dump(obs_points, f)
print(f'{len(obs_points)} observation points saved to observation_points.json')
