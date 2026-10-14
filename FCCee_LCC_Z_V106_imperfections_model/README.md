# Lattices

Lattices and assumed error tolerances are stored on EOS:

- CERNBox: https://cernbox.cern.ch/s/pQB9J1q8RWWl4kH
- EOS: `/eos/user/t/tprebiba/Xsuite_Imperfections_Model_Lattices/lattices_FCCee_LCC_Z_V106`

```
lattices_FCCee_LCC_Z_V106/
├── reference_lattice_LCC_V106/                               # Reference (ideal) lattice
├── lattices_with_imperfections/                              # Lattices with imperfections only (no corrections)
│   ├── old_tolerances/
│   └── updated_tolerances/
├── lattices_with_corrected_imperfections/
│   ├── 01_orbit_corrected_only/                              # After orbit correction
│   │   ├── old_tolerances/
│   │   └── updated_tolerances/
│   └── 02_orbit_and_optics_corrected/                        # Final lattices: orbit and optics corrected, with and without radiation
│       ├── old_tolerances/
│       └── updated_tolerances/
├── Imperfections Model - Old Error Tolerances - LCC V106.png      # Old error tolerances
└── Imperfections Model - Updated Error Tolerances - LCC V106.png  # Updated error tolerances
```