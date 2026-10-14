# Lattices

Lattices and assumed error tolerances are stored on EOS:

- CERNBox: https://cernbox.cern.ch/s/pQB9J1q8RWWl4kH
- EOS: `/eos/user/t/tprebiba/Xsuite_Imperfections_Model_Lattices/lattices_FCCee_LCC_Z_V105`

````
lattices_FCCee_LCC_Z_V105/
├── reference_lattice/                                       # Reference (ideal) lattice
├── lattices_with_imperfections/                             # Lattices with imperfections only (no corrections)
├── lattices_with_corrected_imperfections/
│   ├── 01_orbit_corrected_only/                             # After orbit correction
│   ├── 02_orbit_and_optics_corrected_without_radiation/     # After orbit and optics correction, no radiation
│   └── 03_orbit_and_optics_corrected_with_radiation_FINAL/  # Final lattices: orbit and optics corrected, with radiation
├── Imperfections Model - Error Tolerances - LCC V105.png    # Assumed error tolerances
└── regex_V105elements.ipynb                                 # Regex patterns used to select V105 element families
````