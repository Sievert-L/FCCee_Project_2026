#%%
import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import sys
helpers_path = f'../' # specify the path for helper functions file
sys.path.insert(0, helpers_path)
from helpers_for_imperfections_model import install_orbit_correctors, add_correctors


#%%
####################################
# Load reference line
env = xt.Environment.from_json(f"lattices/reference_lattice/fccee_z.json")
line = env.lines['fccee_p_ring']
tt = line.get_table()


#%%
####################################
# Add orbit correctors 

# Get all quads
ttquad = tt.rows[tt.element_type == 'Quadrupole'] 

# Install orbit correctors (H & V) at all the quadrupoles which do not already have an orbit corrector attached to them
# Orbit correctors: xtrack.Multipole elements of order 0
hcor_names, vcor_names = install_orbit_correctors(line, ttquad.name)
print('Number of horizontal correctors:', len(hcor_names))
print('Number of vertical correctors:', len(vcor_names))


#%%
####################################
# Add optics correctors (trims)

# Get normal quads
mask = [quad for quad in ttquad.name if line.element_dict[quad].k1s==0 and abs(line.element_dict[quad].k1)>0]
ttquadno = ttquad.rows[np.isin(ttquad.name, mask)]
# Normal quadrupole corrector trims are added to normal quadrupoles
# The knl[1] of the quadrupole is added as: knl + order + '_' + element_name (e.g., knl1_q1l1.1)
add_correctors(line, ttquadno.name, type='normal', order=1, switch_name='on_qno_corrector')

# Get normal sexts
ttsext = tt.rows[tt.element_type=='Sextupole']
mask = [sext for sext in ttsext.name if line.element_dict[sext].k2s==0 and abs(line.element_dict[sext].k2)>0]
ttsextno = ttsext.rows[np.isin(ttsext.name, mask)]
# Skew quadrupole corrector trims to normal sextupoles
# The ksl[1] of the sextupole is added as: ksl + order + '_' + element_name (e.g., ksl1_s1l1.1)
add_correctors(line, ttsextno.name, type='skew', order=1, switch_name='on_qsk_corrector')


#%%
####################################
# Save the line with correctors
line.to_json(f"lattices/reference_lattice/fccee_z_with_correctors.json")


#%%
# ####################################
# # Check the corrector installation
# # by investigating the number of elements
# tt = line.get_table()
# ttdip = tt.rows[tt.element_type=='RBend']
# ttquad = tt.rows[tt.element_type=='Quadrupole']
# ttsext = tt.rows[tt.element_type=='Sextupole']
# ttbpms = tt.rows['bpm.*']
# tthcor = tt.rows['hcor.*']
# ttvcor = tt.rows['vcor.*']
# print('Number of dipoles:', len(ttdip))
# print('Number of quadrupoles:', len(ttquad))
# print('Number of sextupoles:', len(ttsext))
# print('Number of BPMs:', len(ttbpms))
# print('Number of H orbit correctors:', len(tthcor))
# print('Number of V orbit correctors:', len(ttvcor))