# Lattices

Lattices and assumed error tolerances are stored on EOS:

- CERNBox: https://cernbox.cern.ch/s/pQB9J1q8RWWl4kH
- EOS: `/eos/user/t/tprebiba/Xsuite_Imperfections_Model_Lattices/lattices_LEP3_Z_V12`

```
lattices_LEP3_Z_V12/
├── lattices_with_imperfections/                 # Lattices with imperfections only (no corrections)
├── lattices_with_corrected_imperfections/
│   ├── 01_orbit_corrected_only/                 # After orbit correction
│   └── 02_orbit_and_optics_corrected/           # Final lattices: orbit and optics corrected, with radiation
│       └── twiss/                               # Twiss for all seeds, with radiation
```

## Local lattices

The local `lattices/` folder (~130 MB) does not store full lattices per seed. It holds the ideal line
plus, for each seed (1–49), the imperfections and corrections as JSON, from which the corrected machine is rebuilt:

```
lattices/
├── reference_lattice/                  # Ideal line: LEP3_Z_v12_wMarkers_wBPMs_wCorr_wTrims.json
└── imperfections_and_corrections/      # Per seed: imperfections json + corrections json
```

## Rebuilding the corrected machine

1. Load the ideal line `reference_lattice/LEP3_Z_v12_wMarkers_wBPMs_wCorr_wTrims.json`.
   It already starts at `'rf800.0'` and already includes the chroma knobs (`sf.k2n.chroma.knob`, `sd.k2n.chroma.knob`),
   so no cycling or `add_chroma_knobs` is needed.
2. Insert the RF frequency shift element (needed for `rf_dzeta`):

   ```python
   line.insert('rf_frequency_shift', xt.ZetaShift(dzeta=0), at=0.0)
   line.element_refs['rf_frequency_shift'].dzeta = line.vars['rf_dzeta']
   ```

3. Apply the **corrections** json:

   | Key | How to apply |
   |---|---|
   | `other_knobs` | `line.vars[name] = value` (tune knobs `kqf1`/`kqd1`, chroma knobs, sextupole powering switches, `rf_dzeta`) |
   | `optics_correctors` | `line.vars[name] = value` (`knl1_<quad>`: normal trims, `ksl1_<sext>`: skew trims; one knob per magnet) |
   | `orbit_correctors` | Kicks of `hcor_<quad>` (`knl[0]`) and `vcor_<quad>` (`ksl[0]`); in the ideal line only the hcor at the QF and the vcor at the QD are driven by `line.vars['knl_<quad>']` and `line.vars['ksl_<quad>']`, so set the kicks directly on the elements |

4. Apply the **imperfections** json (values generated with the imperfection switches stored under `'switches'`):

   | Key | How to apply |
   |---|---|
   | `misalignments` | Set element attributes to these values (`shift_x`, `shift_y`, `shift_s`, `rot_s_rad_no_frame`) |
   | `field_errors_relative` | `k -> k * (1 + value)` for `k0`, `k1`, `k2` |
   | `multipole_errors` | Added to the element `knl[n]` and `ksl[n]` |
