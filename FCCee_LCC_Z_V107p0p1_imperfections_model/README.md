# Lattices

Lattices and assumed error tolerances are stored on EOS:

- CERNBox: https://cernbox.cern.ch/s/pQB9J1q8RWWl4kH
- EOS: `/eos/user/t/tprebiba/Xsuite_Imperfections_Model_Lattices/lattices_FCCee_LCC_Z_V107.0.1`

```
lattices_FCCee_LCC_Z_V107.0.1/
├── lattices_with_imperfections/                 # Lattices with imperfections only (no corrections)
├── lattices_with_corrected_imperfections/
│   ├── 01_orbit_corrected_only/                 # After orbit correction
│   └── 02_orbit_and_optics_corrected/           # Final lattices: orbit and optics corrected, with radiation
│       └── twiss/                               # Twiss for all seeds, with radiation
```

## Local lattices

The local `lattices/` folder (~580 MB) does not store full lattices per seed. It holds the ideal line
plus, for each seed, the imperfections and corrections as JSON, from which the corrected machine is rebuilt:

```
lattices/
├── reference_lattice/                  # Ideal line: fccee_z_with_correctors.json
└── imperfections_and_corrections/      # Per seed: imperfections json + corrections json
```

## Rebuilding the corrected machine

1. Load the ideal line `reference_lattice/fccee_z_with_correctors.json`.
2. Cycle it to start at `'rf400'`.
3. Call `add_chroma_knobs(line, 'LCC')`.
4. Apply the **corrections** json:

   | Key | How to apply |
   |---|---|
   | `other_knobs` | `line.vars[name] = value` (tune and chroma knobs, sextupole powering switches) |
   | `optics_correctors` | `line.vars[name] = value` (`knl1_<quad>`: normal trims, `ksl1_<sext>`: skew trims) |
   | `orbit_correctors` | Kicks of `hcor_<quad>` (`knl[0]`) and `vcor_<quad>` (`ksl[0]`); in the ideal line these are driven by `line.vars['knl_<quad>']` and `line.vars['ksl_<quad>']` |

5. Apply the **imperfections** json (values generated with the imperfection switches stored under `'switches'`):

   | Key | How to apply |
   |---|---|
   | `misalignments` | Set element attributes to these values (`shift_x`, `shift_y`, `shift_s`, `rot_s_rad_no_frame`) |
   | `field_errors_relative` | `k -> k * (1 + value)` for `k0`, `k1`, `k2` |
   | `multipole_errors` | Added to the element `knl[n]` and `ksl[n]` |