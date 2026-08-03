import xtrack as xt
import json

line = xt.Line.from_json('line_fccee_LCC_105-0-0_z_with_qno_qsk.json')
tt = line.get_table()
ttobs = tt.rows['bpm.*|ip.*']
obs_points = list(ttobs.name)
with open('observation_points.json', 'w') as f:
    json.dump(obs_points, f)
print('Observation points saved to observation_points.json')
obs_points = json.load(open('observation_points.json', 'r'))