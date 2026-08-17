path = '/home/larasievert/cernbox'

import xtrack as xt
import sys
import numpy as np
import matplotlib.pyplot as plt

line = xt.Line.from_json(f"{path}/09_LCC_V106/line_fccee_p_ring_LCC_106-2-3_z_merged_dipoles.json")
tt = line.get_table()

tthcor = tt.rows['hcor.*']
ttvcor = tt.rows['vcor.*']
ttquad = tt.rows[tt.element_type == 'Quadrupole']

def install_orbit_correctors(line, elements, prefix=('hcor_', 'vcor_')):
    tt = line.get_table()

    for element in elements:
        s_insert = tt.rows[element].s[0]

        # Create variables
        line.vars[f'knl_{element}'] = 0
        line.vars[f'ksl_{element}'] = 0

        # Corrector names
        hcor_name = f'{prefix[0]}{element}'
        vcor_name = f'{prefix[1]}{element}'

        # Horizontal corrector
        if hcor_name not in line.element_dict:
            line.insert(hcor_name, xt.Multipole(knl=np.array([0.])), at=s_insert)
            line[hcor_name].knl = line.vars[f'knl_{element}']

        # Vertical corrector
        if vcor_name not in line.element_dict:
            line.insert(vcor_name, xt.Multipole(ksl=np.array([0.])), at=s_insert)
            line[vcor_name].ksl = line.vars[f'ksl_{element}']

install_orbit_correctors(line, ttquad.name) # install orbit correctors at all quadrupoles which have no corrector attached to them

line.to_json(f"{path}/09_LCC_V106/line_fccee_p_ring_LCC_106-2-3_z_merged_dipoles_with_correctors.json")