# Imperfections Model of the FCC-ee with Corrections and Beam-Beam Effects
## Lara Sophie Sievert & Tirsi Prebibaj
### Master's Project 2026 

#### Development of an imperfections model in Xsuite and the design of a correction scheme to generate lattices with corrected imperfections, which are used to study the impact of beam-beam interactions.

Master's Thesis: **Imperfections Model of the FCC-ee with Corrections and Beam-Beam Effects** (Lara Sophie Sievert), July 2026, 
under the supervision of Dr. Tirsi Prebibaj, Dr. Tatiana Pieloni, Prof. Dr. Mike Seidel

The Master's thesis focussed on the FCC-ee for the LCC lattice at the Z pole, but the generic implementation of this model in Xsuite allows for it to be extended to other FCC-ee energies as well as other machines.
The chosen error tolerances for the applied imperfections can easily be modified.

The applied imperfections include:
- Transverse and longitudinal misalignments ($x$, $y$, $s$)
- Rotations around the longitudinal axis ($s$-rotation)
- Field errors

of the dipoles, quadrupoles, and sextupoles in the arcs and the straight sections.

The helper functions needed to build the imperfections model are found inside the `helpers_for_imperfections_model` file. Example usage is provided in the code scripts stored inside the folders for the respective machine.

The procedure to follow is:
1. Requirements for the imperfections model:
- **Design an optimal BPM & corrector layout**, e.g. BPMs attached to quadrupoles, horizontal and vertical orbit correctors (dipole kickers) at  quadrupoles, normal and skew optics correctors (quadrupole trims) at quadrupoles and sextupoles.
- **Choose error tolerances** for the alignment of elements in the arcs and straight sections.

2. **Apply imperfections** to the chosen elements in the lattice using regular-expression or marker-based filtering. The error configurations can be controlled through switch variables. The lattices with imperfections switches are found inside the folder: `lattices/lattices_with_imperfections`.

3. **Apply a global orbit correction routine** using SVD or MICADO and threading. A functionality for a gradual ramping of the sextupole strengths is implemented. The orbit-corrected lattices are found inside the folder: `lattices/lattices_with_corrected_imperfections/01_orbit_corrected_only`. 

4. **Apply a global optics correction routine** using the response matrix approach, targeting the beta-beating, dispersion, phase advance, coupling, etc. The response matrices are stored inside the `helpers_for_RM_generation` folder, which also contains the scripts used to generate them. The lattices with corrected imperfections are stored inside the folder: `lattices/lattices_with_corrected_imperfections/02_orbit_and_optics_corrected`.

5. **Investigate the impact of beam-beam effects** in lattices with corrected imperfections.

For questions, contact <lara.sophie.sievert@cern.ch>.

Many thanks to: K. André, S. Jagabathuni, W. Herr, G. Katsanevakis, T. Pieloni, T. Prebibaj, P. Raimondi, M. Seidel, K. Skoufaris, L. van Riesen-Haupt, Y. Wu
