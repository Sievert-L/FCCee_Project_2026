# Imperfections Model of the FCC-ee with Corrections and Beam-Beam Effects
## Master's Project 2026
### Lara Sophie Sievert (Supervisors: Dr. Tirsi Prebibaj, Dr. Tatiana Pieloni, Prof. Dr. Mike Seidel)

#### Development of an imperfections model of the FCC-ee for the LCC lattice at the Z pole and the design of a correction scheme to generate lattices with corrected imperfections, which are used to study the impact of beam-beam interactions.

The generic implementation of this model (in Xsuite) allows for it to be extended to all the other FCC-ee energies as well as other accelerators.
The chosen error tolerances for the applied imperfections can easily be modified.

The applied imperfections include:
- Transverse and longitudinal misalignments ($x$, $y$, $s$)
- Rotations around the longitudinal axis ($s$-rotation)
- Field errors of the dipoles, quadrupoles, and sextupoles in the arcs and the straight sections.

The helper functions needed to build the imperfections model are stored inside the `helper_functions` folder. Example usage is provided in the code scripts stored inside the 'create_imperfections_model_with_corrections' folder.

The procedure to follow is the following:
1. **Apply imperfections** to the chosen elements in the lattice using regular-expression based filtering. The error configurations can be controlled through switch variables. The lattices with imperfections switches are found inside the folder:
lattices/choose_line_version/lattices_with_imperfections

2. **Apply a global orbit and optics correction scheme** (using the response matrix approach). The correction routine generates lattices with corrected imperfections, which are found inside the folder: lattices/choose_line_version/lattices_with_corrected_imperfections/03_orbit_and_optics_corrected_with_radiation_FINAL

3. **Investigate the impact of beam-beam effects** in lattices with corrected imperfections using the scripts provided in the install_beambeam_effects folder.

For questions, contact <lara.sophie.sievert@cern.ch>.

Many thanks to: K. André, S. Jagabathuni, W. Herr, G. Katsanevakis, P. Raimondi, K. Skoufaris, L. van Riesen-Haupt, Y. Wu
