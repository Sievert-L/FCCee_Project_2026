# Markers (start, end) separating the different sections of the LEP3 lattice (installed in notebook 01; hardcoded for now).
# Used for the marker-based filtering of the elements (helpers_for_imperfections_model.find_elements).
# Import in the notebooks with e.g.: from lattice_markers import arc_markers, TR_markers, IR_markers
# Note: there are no separate DS markers. The DS next to the technical regions is inside the arc markers, while the
# ir3/ir7 extents cut through the DS next to the interaction regions (see notebook 02 for how these DS elements are treated).

# ARC SECTIONS (incl. the DS next to the technical regions)
arc_markers = [
    ('ir8_end', 'ir1_start'),
    ('ir1_end', 'ir2_start'),
    ('ir2_end', 'ir3_start'),
    ('ir3_end', 'ir4_start'),
    ('ir4_end', 'ir5_start'),
    ('ir5_end', 'ir6_start'),
    ('ir6_end', 'ir7_start'),
    ('ir7_end', 'ir8_start')]
# TECHNICAL REGIONS (TR)
TR_markers = [
    ('ir1_start', 'ir1_end'),
    ('ir2_start', 'ir2_end'),
    ('ir4_start', 'ir4_end'),
    ('ir5_start', 'ir5_end'),
    ('ir6_start', 'ir6_end'),
    ('ir8_start', 'ir8_end')]
# INTERACTION REGIONS (IR), incl. part of the DS
IR_markers = [
    ('ir3_start', 'ir3_end'),
    ('ir7_start', 'ir7_end')]
