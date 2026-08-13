import warnings
import numpy as np
import xtrack as xt
import matplotlib.pyplot as plt

##############################################
# Helpers for applying imperfections
##############################################

def find_elements(tt, pattern=None, marker_pairs=None, element_type=None, except_pattern=None, except_names=None, filter_aper=True):
    """
    Function to filter out specific line elements based on regex patterns or markers.
    Provide either a regex pattern or a list of marker pairs (start marker, end marker) to select elements. 
    Optionally, filter by element type and exclude elements with specific regex patterns or names.
    
    Parameters
    ----------
    tt : xtrack.LineTable
        The line table containing all elements.
    pattern : str or list of str, optional
        Regular expression(s) to match element names. If provided, elements matching these patterns will be selected.
    marker_pairs : list of tuples, optional
        List of marker pairs (start_marker, end_marker) to define sections of the line. If provided, elements within these sections will be selected.
    element_type : str, optional
        If specified, only elements of this type are returned.
    except_pattern : str or list of str, optional
        Regular expression(s) to match element names to exclude.
    except_names : list of str, optional
        List of element names to exclude.
    filter_aper : bool, optional
        If True, elements with names ending in '_aper' are excluded.
    
    Returns
    -------
    xtrack.LineTable
        Subset of line elements matching the criteria.
    """

    if pattern is not None and marker_pairs is not None:
        raise ValueError("Provide either 'pattern' (for regex-based filtering) or 'marker_pairs' (for marker-based filtering), not both!")
    
    # --------------------------- Regex-based filtering --------------------------- #
    if pattern is not None: 
        patterns = pattern if isinstance(pattern, (list, tuple)) else [pattern]
        tt_element = sum((tt.rows[f'{p}.*'] for p in patterns), tt.rows[[]])

    # -------------------------- Marker-based filtering -------------------------- #
    elif marker_pairs is not None: 

        # If marker_pairs is a list of several marker-pair lists, flatten it
        if marker_pairs and isinstance(marker_pairs[0][0], (list, tuple)):
            marker_pairs = [pair for group in marker_pairs for pair in group]

        mask = np.zeros(len(tt), dtype=bool)

        for start_marker, end_marker in marker_pairs:
            s_start = tt.rows[tt.name == start_marker].s
            s_end   = tt.rows[tt.name == end_marker].s

            if s_start < s_end:
                # normal section: include start, exclude end
                mask |= (tt.s >= s_start) & (tt.s < s_end)
            else:
                # wraparound section: include start, exclude end
                mask |= (tt.s >= s_start) | (tt.s < s_end)

        tt_element = tt.rows[mask]

    else:
        raise ValueError("Need to specify either 'pattern' (for regex-based filtering) or 'marker_pairs' (for marker-based filtering)!")

    # -------------------- Filter by element type if provided -------------------- #
    if element_type is not None: 
        tt_element = tt_element.rows[tt_element.element_type == element_type]

    # ------- Exclude elements based on except_pattern and/or except_names ------- #
    excluded_elements = []

    if except_pattern is not None:
        patterns = except_pattern if isinstance(except_pattern, (list, tuple)) else [except_pattern]   
        tt_excluded = sum((tt.rows[f'{p}.*'] for p in patterns), tt.rows[[]])
        excluded_elements.extend(tt_excluded.name)

    if except_names is not None:
        names = except_names if isinstance(except_names, (list, tuple, np.ndarray)) else [except_names]
        excluded_elements.extend(names)

    if len(excluded_elements) > 0:
        tt_element = tt_element.rows[~np.isin(tt_element.name, excluded_elements)]

    if filter_aper:
        tt_element = tt_element.rows['.*(?<!_aper)$']

    if len(tt_element) == 0:
            warnings.warn("Warning: No elements matched.")

    return tt_element


def _truncated_normal(rgen, size, sigma, nsigma):
    """
    Generates an array of size `size` containing random numbers 
    drawn from a normal distribution truncated at nsigma*sigma.
    
    Parameters
    ----------
    rgen : np.random.RandomState
        Random number generator.
    size : int
        Number of random numbers to generate.
    sigma : float
        Standard deviation of the normal distribution.
    nsigma : float
        Number of standard deviations to truncate the distribution.
    
    Returns
    -------
    np.ndarray
        Array of random numbers.
    """
    x = rgen.randn(size) * sigma
    # Resample until all values are within bounds
    mask = np.abs(x) > nsigma * sigma
    while np.any(mask):
        x[mask] = rgen.randn(np.sum(mask)) * sigma
        mask = np.abs(x) > nsigma * sigma
    return x


def _apply_random_values(element_names, line, attrs, vals,
                         switch_name=None, wrt_current_expr = False, apply_relative=False):
    """
    A function to apply random values (vals) to element attributes (attrs) 
    for a specific element group (element_names).
    
    Parameters
    ----------
    element_names : list of str
        Names of elements to modify.
    line : xtrack.Line
        Line containing the elements.
    attrs : list of str
        Attributes to modify.
    vals : list of np.ndarray with size (len(element_names), len(attrs))
        Random values to apply to the attributes. Each array in the list corresponds to one attribute.
    switch_name : str, optional
        Name of the switch to control the application of the random values.
    wrt_current_expr : bool, optional
        If True, the random values are added to the existing attribute expressions. 
        If False, the random values are applied directly (absolute change).
    apply_relative : bool, optional
        If True, the random values are applied as a relative change (i.e., multiplied by the existing value). 
        This is only relevant if wrt_current_expr is True.
    
    Returns
    -------
    None
         The function modifies the elements in place.
    """
    # Create the switch if it does not exist, or reset it to 1 if it already exists
    if switch_name is not None:
        line.vars[switch_name] = 1 

    # Apply the random values to the selected elements and attributes
    for name, *v in zip(element_names, *vals):
        elem = line[name]
        elem_ref = line.element_refs[name]
        for a, val in zip(attrs, v):

            if wrt_current_expr==False: # <-- to apply the value/expression directly, without being added to the existing one
                setattr(elem, a, val*line.vars[switch_name] if switch_name is not None else val)
            
            elif wrt_current_expr: # <-- to apply the value/expression as a relative change, i.e. added to the existing one
                # Take current expression
                current = getattr(elem_ref, a)._expr
                if current is None: # <-- to handle the case in which the attribute has no expression, just a value
                    current = getattr(elem_ref, a)._value
                    if current == 'from_h': # <-- if k0 is set to the string value 'from_h', we compute its numerical value via angle/length
                        angle = getattr(elem_ref, 'angle')._value
                        length = getattr(elem_ref, 'length')._value
                        current = angle / length
                # Add it on top of the existing one, multiplied by the switch if it exists
                if apply_relative==False:
                    setattr(elem, a, (current + val * line.vars[switch_name]) if switch_name is not None else (current+val))
                # Apply it as a relative change, i.e. multiplied by the existing one (with or without a switch)
                elif apply_relative:
                    setattr(elem, a, (current * (1 + val * line.vars[switch_name])) if switch_name is not None else (current * (1+val)))
                    

def _group_arc_elements_in_girders(tt, arcquad_names, element_types=None, allowed_names=None):
    '''
    Group elements that share the same girder. The assumed model is that there is a girder per arc
    quadrupole, that includes additionally a sextupole, BPM, dipole correctors, and quad correctors. 
    
    The function works by looping all arc sextupoles and searching for the closest bends on the left 
    and right. Then it collects all elements in between (sextupoles, BPMs, etc.).
    
    This method can be problematic for quads that are located in the arc edges. Thus, if arc_sext_names 
    is provided (list of all arc sextupoles), only these ones are kept.
    
    Parameters
    ----------
    tt : xtrack.LineTable
        Line table containing all elements to search (typically all line elements).
    arcquad_names: list of str
        List of names of the arc quadrupoles (one per girder is assumed).
    element_types: list of str, optional
        If provided, only elements whose type is in this list are included.
    allowed_names: list of str, optional
        If provided, only elements whose name is in this list are included.

    Returns
    -------
    dict
        Dictionary with keys being the arc quadrupole names and values being the list of sextupoles
        (or more generally elements) that are in the same girder (i.e. between the closest bends on the left and right).
    '''
    names = list(tt['name'])
    name_to_index = {n: i for i, n in enumerate(names)}
    types = list(tt['element_type'])
    
    groups = {}
    # For each quadrupole in the arc, search for the closest bends on the left and right, and collect sextupoles in between
    for q in arcquad_names:
        i0 = name_to_index[q]
        # search the left bend
        i_left = i0
        while i_left > 0 and types[i_left] != 'RBend':
            i_left -= 1
        # search the right bend
        i_right = i0
        while i_right < len(names)-1 and types[i_right] != 'RBend':
            i_right += 1
        
        elems = [] 
        for i in range(i_left+1, i_right):

            if element_types is not None and types[i] not in element_types:
                continue

            if allowed_names is not None and names[i] not in allowed_names:
                continue

            elems.append(names[i])
        groups[q] = elems

    return groups


def apply_errors(line, seed, sigmas, attrs, pattern=None, marker_pairs=None, switch_name=None,
                 line_table=None, element_type=None, filter_aper=True,
                 nsigma=2.5, wrt_current_expr=False, apply_relative=False,
                 vals=None, element_names=None):
    """
    Apply random errors (field errors, misalignments, rotations) to 
    elements in the line based on either the provided regex patterns or marker pairs. 
    If none are provided, the element_names parameter must be used instead.
    
    Parameters
    ----------
    line : xtrack.Line
        Line containing the elements to misalign.
    seed : int
        Random seed for reproducibility.
    sigmas : list of float
        Standard deviations for the random values to apply to the attributes.
    attrs : list of str
        Attributes to modify (e.g., ['shift_x', 'shift_y', 'shift_s'] for misalignments, ['rot_x_rad', 'rot_y_rad', 'rot_s_rad_no_frame'] for rotations).
    pattern : str or list of str, optional
        Regular expression(s) to match element names. 
        If None, filtering must be performed based on marker_pairs, or element_names must be provided instead.
    marker_pairs : list of str, optional
        List of marker pairs to use for filtering elements. 
        If None, filtering must be performed based on regex patterns, or element_names must be provided instead.
    switch_name : str, optional
        Name of the switch to control the application of the random values. If None, the values are applied directly without being multiplied by a switch variable.
    line_table : xtrack.LineTable, optional
        Line table to use for searching. If None, the line's table is calculated and used.
    element_type : str, optional
        If specified, only elements of this type are considered for imperfections.
    filter_aper : bool, optional
        If True, elements with names ending in '_aper' are excluded from receiving imperfections.
    nsigma : float, optional
        Number of standard deviations to truncate the normal distribution for the random values.
    wrt_current_expr : bool, optional
        If True, the random values are added to the existing attribute expressions.
        If False, the random values are applied directly (absolute change).
    apply_relative : bool, optional
        If True, the random values are applied as a relative change (i.e., multiplied by the existing value). 
        This is only relevant if wrt_current_expr is True.
    vals: list of np.ndarray with size (len(element_names), len(attrs))
        Random values to apply to the attributes. Each array in the list corresponds to one attribute.
        If provided, the random values are not generated, and the seed, sigmas, and nsigma parameters are ignored.
    element_names : list of str, optional
        List of element names to apply imperfections to. If provided, the pattern parameter and marker pairs are ignored.
    
    Returns
    -------
    element_names : list of str
        List of elements that were misaligned/rotated.
    """
    if line_table is None:
            line_table = line.get_table()

    # First get the names of the elements to receive imperfections, either from 
    # the provided element_names or by searching with the regex patterns or markers in the line
    if element_names is None:
        # Filter out the elements to receive imperfections based on the provided regex pattern or marker pairs
        tt_element = find_elements(tt=line_table, pattern=pattern, marker_pairs=marker_pairs, element_type=element_type, filter_aper=filter_aper)
        element_names = tt_element.name
        if len(element_names) == 0:
            print("No elements matched. Aborting the application of imperfections.")
            return element_names

    # Generate the random values to apply to the attributes, if not provided
    if vals is None:
        rgen = np.random.RandomState(seed)
        # The random values are generated for all elements and attributes at once
        # vals size: (len(element_names), len(attrs))
        vals = [_truncated_normal(rgen, len(element_names), s, nsigma) for s in sigmas]
        
    # Apply the random values to the selected elements
    _apply_random_values(element_names, line, attrs, vals,
                         switch_name=switch_name, 
                         wrt_current_expr=wrt_current_expr, apply_relative=apply_relative)     
    
    return element_names


def apply_girder_misalignments(line, seed, sigmas, attrs, switch_name=None, line_table=None, 
                               arcquads_regex=None, arcsexts_regex=None, marker_pairs_arcs=None):
    """
    Apply the same misalignment to all elements in the same girder, by grouping the elements based on the closest bends.
    For now, only applied to the arc quadrupoles and sextupoles, which are filtered using either the provided regex patterns or the marker pairs.

    Parameters
    ----------
    line : xtrack.Line
        Line containing the elements to misalign.
    seed : int
        Random seed for reproducibility.
    sigmas : list of float
        Standard deviations for the random values to apply to the attributes.
    attrs : list of str
        Attributes to modify.
    switch_name : str, optional
        Name of the switch to control the application of the random values. 
        If None, the values are applied directly without being multiplied by a switch variable.
    line_table : xtrack.LineTable, optional
        Line table to use for searching. If None, the line's table is calculated and used.
    arcquads_regex : list of str, optional
        List of regular expressions to identify the arc quadrupoles that define the girders. 
        If not provided, marker_pairs_arcs must be used to filter out the elements.
        For FCC-ee, the regex pattern for arc quadrupoles is ['qf2a', 'qf3a', 'qd1a'].
    arcsexts_regex : list of str, optional
        List of regular expressions to identify the arc sextupoles. 
        If not provided, marker_pairs_arcs must be used to filter out the elements.
        For FCC-ee, the regex pattern for arc sextupoles is ['sf1a', 'sd1a', 'sf2a', 'sd2a'].
    marker_pairs_arcs : list of tuples, optional
        List of marker pairs in the form (start marker, end marker) to define the arcs of the line to filter out the arc quadrupoles and sextupoles. 
        If not provided, the arc quadrupoles and sextupoles must be identified using the respective regex patterns.

    Returns
    -------
    dict
        Dictionary with keys being the arc quadrupole names and values being the list of sextupoles (or more generally elements) 
        that are in the same girder (i.e. between the closest bends on the left and right).
    """
    if line_table is None:
        line_table = line.get_table()

    # Validating that exactly one filtering method (regex patterns or marker pairs) is provided for both arc quadrupoles and sextupoles
    if (arcquads_regex is None) == (marker_pairs_arcs is None):
        raise ValueError("For arc quadrupoles, specify exactly one of 'arcquads_regex' or 'marker_pairs_arcs'.")
    if (arcsexts_regex is None) == (marker_pairs_arcs is None):
        raise ValueError("For arc sextupoles, specify exactly one of 'arcsexts_regex' or 'marker_pairs_arcs'.")
    
    # Filtering out arc quadrupoles and sextupoles based on the provided regex patterns or marker pairs
    if arcquads_regex is not None:
        ttarcquad = find_elements(tt=line_table, pattern=arcquads_regex, element_type='Quadrupole') # filter by regex pattern
    else:
        ttarcquad = find_elements(tt=line_table, marker_pairs=marker_pairs_arcs, element_type='Quadrupole') # filter by marker pairs

    if arcsexts_regex is not None:
        ttarcsext = find_elements(tt=line_table, pattern=arcsexts_regex, element_type='Sextupole') # filter by regex pattern
    else:
        ttarcsext = find_elements(tt=line_table, marker_pairs=marker_pairs_arcs, element_type='Sextupole') # filter by marker pairs

    
    groups = _group_arc_elements_in_girders(line_table, ttarcquad.name, 
                                            element_types=['Sextupole'], allowed_names=ttarcsext.name)

    # All elements of the group share the same girder, so they get the same misalignment 
    # Generate all misalignements for all girders at once, and then apply them group by group
    rgen = np.random.RandomState(seed)
    girder_vals = np.array([_truncated_normal(rgen, len(groups), s, nsigma=2.5) for s in sigmas])
    
    # Loop over all girders and apply the same misalignment to all elements in the girder (but with the same switch)
    for ii, (q, elems) in enumerate(groups.items()):
        element_names = [q] + elems
        # we get the same values for all element quads but different for the different attributes
        vals = np.repeat(girder_vals[:,ii][:,None], len(element_names), axis=1)
        apply_errors(line, pattern = None, seed=seed, sigmas=sigmas, attrs=attrs, switch_name=switch_name,
                     line_table=line_table, element_type=None, filter_aper=True, nsigma=2.5, wrt_current_expr=True, # <- important
                     element_names=element_names, vals=vals, # <- important
                     )

    return groups


def generate_monitor_misalignments(line, line_table, pattern='bpm', element_type='Marker',
                                   attrs=['shift_x', 'shift_y', 'rot_s_rad']):
    """
    Generate a dictionary with the misalignments and rotations of the monitors.
    The assumption is that the monitors are attached to the quadrupoles, 
    so they inherit the misalignments/rotations of the quadrupoles (which themselves may be placed on top of a girder).

    Parameters
    ----------
    line : xtrack.Line
        Line containing the elements to misalign.
    line_table : xtrack.LineTable
        Line table to use for searching.
    pattern : str, optional
        Regular expression to match monitor element names. Default is 'bpm'.
    attrs : list of str, optional
        Attributes to include in the output dictionary (e.g., ['shift_x', 'shift_y', 'rot_s_rad'] for misalignments and rotations). 
        Default is ['shift_x', 'shift_y', 'rot_s_rad'].
    element_type : str, optional
        If specified, only elements of this type are considered for searching. Default is 'Marker'.
    
    Returns
    -------
    dict
        Dictionary with monitor names as keys and their misalignments/rotations as values.
    """
    monitor_alignment = {}
    tt_bpm = find_elements(tt=line_table, pattern=pattern, element_type=element_type)
    for bpm in tt_bpm.name:
        quad = bpm.replace('bpm_', '')
        quad_elem = line[quad]

        monitor_alignment[bpm] = {}
        for attr in attrs:
            attr_quad = 'rot_s_rad_no_frame' if attr == 'rot_s_rad' else attr
            monitor_alignment[bpm][attr] = float(getattr(quad_elem, attr_quad))
    
    return monitor_alignment


def apply_monitor_misalignments(monitor_alignment, seed, sigmas, nsigma=2.5):
    """
    Generate additional monitor misalignments and rotations that are added on top of the misalignments 
    inherited from the quadrupoles already stored in the monitor alignment dictionary.
    The additional misalignments are generated based on the truncated Gaussian model, 
    with the specified standard deviations (sigmas) and truncation at nsigma*sigma.
    
    Parameters
    ----------
    monitor_alignment : dict
        The original monitor alignment dictionary.
    seed : int
        Seed for the random number generator.
    sigmas : list of float
        Standard deviations for the random values to apply to the shift/rotation attributes..
    nsigma : float, optional
        Number of standard deviations to truncate the normal distribution for the random values. Default is 2.5.
    Returns
    -------
    dict
        The monitor alignment dictionary with the additional misalignments applied.
    """
    rgen = np.random.RandomState(seed)

    for bpm in monitor_alignment:
        for i, attr in enumerate(monitor_alignment[bpm]):
            val = _truncated_normal(rgen, 1, sigmas[i], nsigma)[0]
            monitor_alignment[bpm][attr] += val
            
    return monitor_alignment


def scale_monitor_alignment(monitor_alignment, scaling_factor):
    """
    Scale the monitor alignment values by a given factor
    (to be able to introduce BPM misalignments adiabatically).

    Parameters
    ----------
    monitor_alignment : dict
        The original monitor alignment dictionary.
    scaling_factor : float
        The factor by which to scale the alignment values.
    Returns
    -------
    dict
        A new monitor alignment dictionary with scaled values.
    """
    for _, values in monitor_alignment.items():
        for key, val in values.items():
            values[key] = val * scaling_factor
    return monitor_alignment


def create_elements_switch(line, line_table, switch_name, pattern=None, marker_pairs=None, element_type=None, 
                           except_pattern=None, except_names=None, filter_aper=True):
    """
    Create a switch variable in the line for the specified elements.
    
    Parameters
    ----------
    line : xtrack.Line
        Line containing the elements.
    line_table : xtrack.LineTable, optional
        Line table to use for searching. If None, the line's table is calculated and used.
    switch_name : str
        Name of the switch variable to create.
    pattern : str or list of str
        Regular expression(s) to match element names for which the switch should be created.
    marker_pairs : list of tuples, optional
        List of marker pairs (start_marker, end_marker) to define sections of the line.
    element_type : str, optional
        If specified, only elements of this type are considered for switch creation. 
        If None, the type is inferred from the elements in tt_element. 
        Supported types are 'RBend', 'Quadrupole', 'Sextupole', 'Octupole', and 'Multipole'.
        This is for an extra layer of safety, in case we have a discrepancy in the tt_element types
        and the switch name that we want to apply. 
    except_pattern : str or list of str, optional
        Regular expression(s) to match element names to exclude from the switch creation.
    except_names : list of str, optional
        List of element names to exclude from the switch creation.
    filter_aper : bool, optional
        If True, elements with names ending in '_aper' are excluded from the switch creation.
    
    Returns
    -------
    None
        The function modifies the line in place by adding the switch variable.
    """
    if switch_name not in line.vars:
    
        tt_element = find_elements(tt=line_table, pattern=pattern, marker_pairs=marker_pairs, element_type=element_type, 
                                    except_pattern=except_pattern, except_names=except_names, filter_aper=filter_aper)

        print(len(tt_element), "elements found for switch", switch_name)

        # To make sure all elements in table are of the same type
        element_types = set(tt_element.element_type)
        if len(element_types) > 1:
            raise ValueError("All elements in the table must be of the same type.")
        if element_type is None:
            element_type = list(element_types)[0]

        line.vars[switch_name] = 1
        if element_type == 'RBend':
            for name in tt_element.name:
                # TODO: To check if k0 overwrites angle or vice-versa
                line.element_refs[name].k0 = line.vars[switch_name] * line.element_refs[name].k0._expr
                #line.element_refs[name].angle = line.vars[switch_name] * line.element_refs[name].angle._expr
        elif element_type == 'Quadrupole':
            for name in tt_element.name:
                line.element_refs[name].k1 = line.vars[switch_name] * line.element_refs[name].k1._expr
        elif element_type == 'Sextupole':
            for name in tt_element.name:
                line.element_refs[name].k2 = line.vars[switch_name] * line.element_refs[name].k2._expr
        elif element_type == 'Octupole':
            for name in tt_element.name:
                line.element_refs[name].k3 = line.vars[switch_name] * line.element_refs[name].k3._expr
        elif element_type == 'Multipole':
            for name in tt_element.name:
                line.element_refs[name].knl = line.vars[switch_name] * line.element_refs[name].knl._expr
                line.element_refs[name].ksl = line.vars[switch_name] * line.element_refs[name].ksl._expr
        else:
            raise ValueError(f"Unsupported element type '{element_type}' for switch creation.")
            
    else:
        raise ValueError(f"Switch name '{switch_name}' already exists in the line variables. Doing nothing.")
    


##############################################
# Helpers for performing orbit correction
##############################################

def install_orbit_correctors(line, elements, prefix=('hcor_', 'vcor_')):
    """
    Install horizontal and vertical orbit correctors (dipole kickers) at the specified elements in the line, 
    as long as no orbit correctors are already present at those elements. 
    The correctors are implemented as Multipole elements with zero strength.
    
    Parameters
    ----------
    line : xtrack.Line
        Line containing the elements.
    elements : list of str
        Names of the elements at which the correctors will be installed.
    prefix : tuple of str, optional
        Prefixes for the names of the horizontal and vertical correctors. Default is ('hcor_', 'vcor_').
    
    Returns
    -------
    None
        The function modifies the line in place by adding the correctors.
    """

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


def apply_orbit_correction(line, twiss_table, monitor_alignment=None, num_sing_vals=None, rcond=None, n_micado=None, 
                           corrector_limits_x=None, corrector_limits_y=None,  ds_thread=5000, rcond_short=1e-7, rcond_long=1e-7):
    """
    Perform orbit correction, using the standard correct_trajectory method in Xsuite. However, if the closed orbit search 
    fails due to the presence of strong lattice perturbations, the function falls back to the threading method, 
    which allows to correct the orbit even in such cases. 
    SVD or MICADO can be used for the correction, depending on the input parameters.
    
    Parameters
    ----------
    line : xtrack.Line
        Line containing the elements.
    twiss_table : xtrack.TwissTable
        Twiss table containing the reference optics parameters.
    monitor_alignment : dict, optional
        Dictionary with monitor names as keys and their misalignments/rotations as values. 
        If None, no monitor misalignments are considered in the orbit correction.
    num_sing_vals : int, optional
        Number of singular values to use in the correction. If None, all singular values are used. Default is None.
    rcond : float, optional
        Cutoff for small singular values (relative to the largest singular value).
        Singular values smaller than rcond are considered zero.
    n_micado : int, optional
        The default is None, which means that SVD is used for the correction.
        If n_micado is provided, the MICADO algorithm is used for the correction. 
        In that case, the number of correctors to be used is given by n_micado.
    corrector_limits_x : tuple of float, optional
        Limits for the horizontal corrector strengths. If not None, it should be a tuple of two arrays 
        (lower_limits, upper_limits) with the same length as the number of horizontal correctors. 
        If None, no limits are applied.
    corrector_limits_y : tuple of float, optional
        Limits for the vertical corrector strengths. If not None, it should be a tuple of two arrays 
        (lower_limits, upper_limits) with the same length as the number of vertical correctors. 
        If None, no limits are applied.
    ds_thread : float, optional
        Step size for the threading method. This is the length of the portion added at each iteration. 
        A smaller value can lead to a more stable correction but longer correction time, 
        while a larger value can speed up the correction but may lead to convergence issues.
    rcond_short : float, optional
        Cutoff for small singular values (relative to the largest singular value) 
        in the threading method (used for the correction of the newly added part). 
        Singular values smaller than rcond_short are considered zero.
    rcond_long : float, optional
        Cutoff for small singular values (relative to the largest singular value) in the threading method
        (used for the correction of the whole portion up to the end of the newly added part). 
        Singular values smaller than rcond_long are considered zero.

    Returns
    -------
    The function modifies the line in place by applying the orbit correction.
    x_sv : np.ndarray
        Singular values of the horizontal orbit response matrix.
    y_sv : np.ndarray
        Singular values of the vertical orbit response matrix.
    kicks_x : np.ndarray
        Kick values of the horizontal correctors.
    kicks_y : np.ndarray
        Kick values of the vertical correctors.
    """

    orbit_correction = line.correct_trajectory(twiss_table=twiss_table, monitor_alignment=monitor_alignment, corrector_limits_x=corrector_limits_x, corrector_limits_y=corrector_limits_y, run=False)
    
    x_sv = orbit_correction.x_correction.singular_values
    y_sv = orbit_correction.y_correction.singular_values

    if num_sing_vals is None:
        n_sv = (len(x_sv), len(y_sv))
    else:
        n_sv = (num_sing_vals, num_sing_vals)

    try: 
        orbit_correction.correct(n_singular_values=n_sv, rcond=rcond, n_micado=n_micado, delta0=0)
        
    except:
        print('Starting orbit correction with threading method...')
        orbit_correction.thread(ds_thread=ds_thread, rcond_short=rcond_short, rcond_long=rcond_long)
        orbit_correction.correct(n_singular_values=n_sv, rcond=rcond, n_micado=n_micado, delta0=0)

    kicks_x = orbit_correction.x_correction.get_kick_values()
    kicks_y = orbit_correction.y_correction.get_kick_values()
    
    return x_sv, y_sv, kicks_x, kicks_y



##############################################
# Helpers for performing optics correction 
# with a response matrix
##############################################

def add_correctors(line, element_names, type='normal', order=1,  switch_name=None):
    """
    Add a corrector to the specified elements by creating a variable in the line.
    The corrector can be either normal or skew, and of any order (e.g., dipole, quadrupole, sextupole, etc.).

    Parameters
    ----------
    line : xtrack.Line
        Line containing the elements to which the corrector will be added.
    element_names : list of str
        Names of the elements to which the corrector will be added.
    type : str, optional
        Type of corrector to add. Supported types are 'normal' (for normal correctors) and 'skew' (for skew correctors). Default is 'normal'.
    order : int, optional
        Order of the corrector to add (e.g., 0 for dipole corrector, 1 for quadrupole corrector, etc.). Default is 1.
    switch_name : str, optional
        Name of the switch variable to control the corrector. If provided, the corrector will be multiplied by this switch variable.
    """
    # Create the switch if it does not exist, or reset it to 1 if it already exists
    if switch_name is not None:
        line.vars[switch_name] = 1

    if type == 'normal':
        attr = 'knl'
    elif type == 'skew':
        attr = 'ksl'
    else:
        raise ValueError(f"Unsupported corrector type '{type}'. Supported types are 'normal' and 'skew'.")

    for element in element_names:
        line.vars[attr + str(order) + '_' + element] = 0
        if switch_name is not None:
            getattr(line.element_refs[element], attr)[order] = line.vars[attr + str(order) + '_' + element]*line.vars[switch_name]
        else:
            getattr(line.element_refs[element], attr)[order] = line.vars[attr + str(order) + '_' + element]


def _load_twiss(twiss, OBSERVABLES, OBSERVATION_POINTS=None):
    '''
    Simply loads the twiss from a csv file or xtrack.Table, and selects the columns and rows 
    corresponding to the observables and observation points of interest.

    Parameters
    ----------
    twiss : str or xtrack.Table
        Path to the csv file containing the twiss or an xtrack.Table object.
    OBSERVABLES : list of str
        List of observables to select from the twiss (e.g. ['mux', 'muy', 'dx']).
    OBSERVATION_POINTS : list of str, optional
        List of observation points to select from the twiss. 
        If None, all points are selected. Default is None.

    Returns
    -------
    xtrack.Table
        Table containing the selected twiss data.
    '''
    if isinstance(twiss, str):
        tw = xt.Table.from_csv(twiss)
    elif isinstance(twiss, xt.Table):
        tw = twiss
    else:
        raise ValueError("twiss_file must be a string path to a .csv table or an xtrack.Table")
    tw = tw.cols[OBSERVABLES]
    if OBSERVATION_POINTS is not None:
        tw = tw.rows[OBSERVATION_POINTS]
    return tw


def _construct_vector(twiss, OBSERVABLES):
    '''
    Constructs a vector from the twiss table, by stacking the columns corresponding to the observables of interest.
    The order of the observables in the vector is the same as the order in the OBSERVABLES list, 
    and the order of the observation points is the same as the order in the twiss table.

    Parameters
    ----------
    twiss : xtrack.Table
        Table containing the twiss data for the observation points and observables of interest.
    OBSERVABLES : list of str
        List of observables to include in the vector (e.g. ['mux', 'muy', 'dx']).
    
    Returns
    -------
    np.ndarray
        Vector constructed from the twiss table.
    '''
    return np.vstack([twiss[o] for o in OBSERVABLES]).reshape(-1)


def build_response_matrix(twiss_files, OBSERVATION_POINTS, OBSERVABLES, vector0, print_progress=True):
    '''
    Builds the response matrix by looping over the twiss files, extracting the corrector name and dk from the file name,
    loading the twiss, constructing the vector, and computing the Jacobian element as (vector - vector0)/dk.

    Parameters
    ----------
    twiss_files : list of str
        List of paths to the csv files containing the twiss data for each corrector perturbation
    OBSERVATION_POINTS : list of str
        List of observation points to select from the twiss.
    OBSERVABLES : list of str
        List of observables to select from the twiss.
    vector0 : np.ndarray
        Reference vector corresponding to the unperturbed case, constructed from the reference twiss.
    print_progress : bool, optional
        If True, prints the progress of the response matrix construction. Default is True.

    Returns
    -------
    RM : np.ndarray
        Response matrix with shape (nxm)xl, where n is the number of observation points, m is the number of observables, 
        and l is the number of correctors (or columns of the response matrix).
    corr_knob_names : list of str
        List of corrector knob names corresponding to the columns of the response matrix.
    dk_list : list of float
        List of dk values corresponding to each corrector perturbation.
    '''
    delta_vectors_over_dk = [] # the elements of the response matrix
    corr_knob_names = [] # the correctors (corresponding to the columns of the response matrix)
    dk_list = [] # the dk list (should be the same for all correctors)
    for ii, file in enumerate(twiss_files):
        if print_progress:
            print(f'Processing file {ii+1}/{len(twiss_files)}: {file}')
        
        # The dk and corrector name are extracted from the file name, which is assumed to be in the format 'twiss_dk_correctorname.csv'.
        # Hardcoded for now; to be improved
        parts = file.replace(".csv", "").split("/")[-1].split('_')
        dk = float(parts[1])
        corr_knob_name = parts[2]+'_'+ parts[3]
        tw = _load_twiss(file, OBSERVABLES=OBSERVABLES, OBSERVATION_POINTS=OBSERVATION_POINTS)
        vector = _construct_vector(tw, OBSERVABLES=OBSERVABLES)

        delta_vectors_over_dk.append((vector - vector0)/dk) # <-- Jacobian element
        dk_list.append(dk)
        corr_knob_names.append(corr_knob_name)

    # The response matrix has the form of (nxm)xl with n the number of observation points, 
    # m the number of observables, and l the number of correctors (or columns of the response matrix)
    RM = np.array(delta_vectors_over_dk).T 

    return RM, corr_knob_names, dk_list


def compute_pseudo_inverse(R, epsilon=0.001, full_matrices=False, Tikhonov_lambda=None):
    '''
    Compute the Moore-Penrose pseudo-inverse of a matrix R using Singular Value Decomposition (SVD).
    Small singular values (less than epsilon * max(S)) are set to zero to improve numerical stability.
    
    Parameters:
    ----------
        R : np.ndarray
            The input matrix to be inverted.
        full_matrices : bool, optional
            If True, compute the full-sized U and VT matrices. Default is False.
        epsilon : float, optional
            Threshold for small singular values. Default is 0.001.
        Tikhonov_lambda : float, optional
            If provided, applies Tikhonov regularisation to the inversion of singular values. Default is None (no regularisation).
    
    Returns:
    ----------
        R_inv : np.ndarray
            The pseudo-inverse of the input matrix R.
        U : np.ndarray
            The left singular vectors of R.
        S : np.ndarray
            The singular values of R.
        VT : np.ndarray
            The right singular vectors of R.
    '''
    U, S, VT = np.linalg.svd(R, full_matrices=full_matrices)

    S_inv = []
    S_max = max(S)

    for s in S:
        if s <= epsilon * S_max:
            S_inv.append(0)
        else:
            if Tikhonov_lambda is not None:
                S_inv.append(s / (s**2 + Tikhonov_lambda**2))
            else:
                S_inv.append(1/s)
            
    R_inv = np.matmul(np.matmul(VT.transpose(), np.diag(S_inv)), U.transpose())
    
    return R_inv, U, S, VT


def apply_optics_correction(line, RM_inv, corr_knob_names, tw_ref, OBSERVATION_POINTS, OBSERVABLES, nloops):
    """
    Apply optics correction to the line by iteratively calculating the strength deltas 
    from the response matrix and applying them to the correctors.

    Parameters
    ----------
    line : xtrack.Line
        Line containing the elements to which the correction will be applied.
    RM_inv : np.ndarray
        Pseudo-inverse of the response matrix.
    corr_knob_names : list of str
        List of corrector knob names corresponding to the columns of the response matrix.
    tw_ref : xtrack.Twiss
        Reference twiss corresponding to the unperturbed case, used to construct the reference vector for the correction.
    OBSERVATION_POINTS : list of str
        List of observation points to select from the twiss.
    OBSERVABLES : list of str
        List of observables to select from the twiss.
    nloops : int
        Number of correction iterations to perform.
    
    Returns
        The function modifies the line in place by applying the correction to the correctors.
    """
    tw0 = _load_twiss(tw_ref, OBSERVATION_POINTS=OBSERVATION_POINTS, OBSERVABLES=OBSERVABLES)
    vector0 = _construct_vector(tw0, OBSERVABLES=OBSERVABLES)
    
    for i in range(nloops):
        tw_before = line.twiss(coupling_edw_teng=True, matrix_stability_tol=0.20)
        # Generate the real and imaginary coupling RDTs as separate columns in the twiss table
        for col in ['f1001', 'f1010']:
            tw_before[col + "c"] = np.real(tw_before[col])
            tw_before[col + "s"] = np.imag(tw_before[col])

        tw1 = _load_twiss(tw_before, OBSERVATION_POINTS=OBSERVATION_POINTS, OBSERVABLES=OBSERVABLES)
        vector1 = _construct_vector(tw1, OBSERVABLES=OBSERVABLES)

        # Create target
        target = vector0 - vector1

        # Calculate strength deltas
        STRENGTH_DELTAS = RM_inv @ target

        # Apply deltas to the correctors
        for corr_name, strength in zip(corr_knob_names, STRENGTH_DELTAS):
            line.vars[corr_name] += strength
