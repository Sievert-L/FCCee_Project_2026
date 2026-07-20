# Imperfections Model of the FCC-ee with Corrections and Beam-Beam Effects
## Master's Project 2026
### Lara Sophie Sievert (Supervisors: Dr. Tirsi Prebibaj, Dr. Tatiana Pieloni, Prof. Dr. Mike Seidel)

#### Development of an imperfections model of the FCC-ee for the LCC lattice at the Z pole and the design of a correction scheme to generate lattices with corrected imperfections, which are used to study the impact of beam-beam interactions.

The generic implementation of this model (in Xsuite) allows for it to be extended to all the other FCC-ee energies as well as other accelerators.
The chosen error tolerances for the applied imperfections can easily be modified.

The applied imperfections include:
- Transverse and longitudinal misalignments ($x$, $y$, $s$)
- Rotations around the longitudinal axis ($s$-rotation)
- Field errors
of the dipoles, quadrupoles, and sextupoles in the arcs and the straight sections.

The lattices with imperfections are found inside the folder:
lattices/choose_line_version/lattices_with_imperfections.

Applying a global orbit and optics correction scheme generates lattices with corrected imperfections, which are found inside the folder: lattices/choose_line_version/lattices_with_corrected_imperfections/03_orbit_and_optics_corrected_with_radiation_FINAL.

Many thanks to: K. André, S. Jagabathuni, W. Herr, G. Katsanevakis, P. Raimondi, K. Skoufaris, L. van Riesen-Haupt, Y. Wu
