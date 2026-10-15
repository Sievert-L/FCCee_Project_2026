# Markers (start, end) separating the different sections of the FCC-ee LCC lattice.
# Used for the marker-based filtering of the elements (helpers_for_imperfections_model.find_elements).
# Import in the notebooks with e.g.: from lattice_markers import arc_markers, DS_markers, IR_markers

# ARC SECTIONS
arc_markers = [
    ('end_ds_start_arc_ipa', 'end_arc_start_ds_ipb'),
    ('end_ds_start_arc_ipb', 'end_arc_start_ds_ipd'),
    ('end_ds_start_arc_ipd', 'end_arc_start_ds_ipf'),
    ('end_ds_start_arc_ipf', 'end_arc_start_ds_ipg'),
    ('end_ds_start_arc_ipg', 'end_arc_start_ds_iph'),
    ('end_ds_start_arc_iph', 'end_arc_start_ds_ipj'),
    ('end_ds_start_arc_ipj', 'end_arc_start_ds_ipl'),
    ('end_ds_start_arc_ipl', 'end_arc_start_ds_ipa')]
# DISPERSION SUPPRESSION REGIONS (count as part of the arc)
DS_markers = [
    ('end_straight_start_ds_ipa', 'end_ds_start_arc_ipa'),
    ('end_arc_start_ds_ipb', 'end_ds_start_straight_ipb'),
    ('end_straight_start_ds_ipb', 'end_ds_start_arc_ipb'),
    ('end_arc_start_ds_ipd', 'end_ds_start_straight_ipd'),
    ('end_straight_start_ds_ipd', 'end_ds_start_arc_ipd'),
    ('end_arc_start_ds_ipf', 'end_ds_start_straight_ipf'),
    ('end_straight_start_ds_ipf', 'end_ds_start_arc_ipf'),
    ('end_arc_start_ds_ipg', 'end_ds_start_straight_ipg'),
    ('end_straight_start_ds_ipg', 'end_ds_start_arc_ipg'),
    ('end_arc_start_ds_iph', 'end_ds_start_straight_iph'),
    ('end_straight_start_ds_iph', 'end_ds_start_arc_iph'),
    ('end_arc_start_ds_ipj', 'end_ds_start_straight_ipj'),
    ('end_straight_start_ds_ipj', 'end_ds_start_arc_ipj'),
    ('end_arc_start_ds_ipl', 'end_ds_start_straight_ipl'),
    ('end_straight_start_ds_ipl', 'end_ds_start_arc_ipl'),
    ('end_arc_start_ds_ipa', 'end_ds_start_straight_ipa')]
# STRAIGHT SECTIONS
straight_markers = [
    ('end_ds_start_straight_ipb', 'end_straight_start_ds_ipb'),
    ('end_ds_start_straight_ipd', 'end_straight_start_ds_ipd'),
    ('end_ds_start_straight_ipf', 'end_straight_start_ds_ipf'),
    ('end_ds_start_straight_ipg', 'end_straight_start_ds_ipg'),
    ('end_ds_start_straight_iph', 'end_straight_start_ds_iph'),
    ('end_ds_start_straight_ipj', 'end_straight_start_ds_ipj'),
    ('end_ds_start_straight_ipl', 'end_straight_start_ds_ipl'),
    ('end_ds_start_straight_ipa', 'end_straight_start_ds_ipa')]
# Split the straight markers into INTERACTION REGION (IR) and TECHNICAL REGION (TR) markers
IR_markers = [
    ('end_ds_start_straight_ipa', 'end_straight_start_ds_ipa'),
    ('end_ds_start_straight_ipd', 'end_straight_start_ds_ipd'),
    ('end_ds_start_straight_ipg', 'end_straight_start_ds_ipg'),
    ('end_ds_start_straight_ipj', 'end_straight_start_ds_ipj')]
TR_markers = [
    ('end_ds_start_straight_ipb', 'end_straight_start_ds_ipb'),
    ('end_ds_start_straight_ipf', 'end_straight_start_ds_ipf'),
    ('end_ds_start_straight_iph', 'end_straight_start_ds_iph'),
    ('end_ds_start_straight_ipl', 'end_straight_start_ds_ipl')]
