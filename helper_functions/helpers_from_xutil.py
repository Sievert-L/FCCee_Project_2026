import xtrack as xt
import xpart as xp
import numpy as np

def match_tune_chroma(line, target_twiss, match_quantities='tune_chroma', method='6d', machine='FCCee'):

    if isinstance(target_twiss, dict):
        if 'tune' in match_quantities:
            target_qx = target_twiss['qx']
            target_qy = target_twiss['qy']
        if 'chroma' in match_quantities:
            target_dqx = target_twiss['dqx']
            target_dqy = target_twiss['dqy']
    else:
        if 'tune' in match_quantities:
            target_qx = target_twiss.qx
            target_qy = target_twiss.qy
        if 'chroma' in match_quantities:
            target_dqx = target_twiss.dqx
            target_dqy = target_twiss.dqy

    if machine == 'FCCee':
        if 'tune' in match_quantities:
            if 'k1qf2' in line.vars.get_table().name:
                opt_tune = line.match(
                    method=method,
                    vary=[xt.VaryList(['k1qf4', 'k1qf2', 'k1qd3', 'k1qd1',], step=1e-8, tag='quad'),
                        ],
                    targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-5, tag='tune'),
                        ])
            
            elif 'kqf6' in line.vars.get_table().name:
                opt_tune = line.match(
                    method=method,
                    vary=[xt.VaryList(['kqf2', 'kqf4', 'kqf6', 'kqd1', 'kqd3', 'kqd5',], step=1e-8, tag='quad'),
                        ],
                    targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-5, tag='tune'),
                        ])
            
            elif 'kqf2' in line.vars.get_table().name:
                opt_tune = line.match(
                    method=method,
                    vary=[xt.VaryList(['kqf2', 'kqd1', ], step=1e-8, tag='quad'),
                        ],
                    targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-5, tag='tune'),
                        ])
            
            opt_tune.target_status()
            opt_tune.vary_status()

        if 'chroma' in match_quantities:
            chroma_knob_refs = line.vars['sf.k2n.chroma.knob']._find_dependant_targets()[1:]
            if len(chroma_knob_refs) == 0:
                print('WARNING: Chroma knobs are not defined, could not match chroma')
            else:
                opt_chroma = line.match(
                    method=method,
                    vary=[xt.VaryList(['sf.k2n.chroma.knob', 'sd.k2n.chroma.knob',], step=1e-3, tag='sext'),
                        ],
                    targets=[xt.TargetSet(dqx=target_dqx, dqy=target_dqy, tol=1e-2, tag='chrom'),
                        ])
                opt_chroma.target_status()
                opt_chroma.vary_status()


    elif machine == 'LHC':
        #if 'dqx.b1' in line.vars.get_table().name:
        if 'tune' in match_quantities:
            opt_tune = line.match(
                method=method,
                vary=[xt.VaryList(['dqx.b1','dqy.b1',], step=1e-8, tag='quad'),
                    ],
                targets=[xt.TargetSet(qx=target_qx, qy=target_qy, tol=1e-5, tag='tune'),
                    ])
            opt_tune.target_status()
            opt_tune.vary_status()
        
        if 'chroma' in match_quantities:
            opt_chroma = line.match(
                method=method,
                vary=[xt.VaryList(['dqpx.b1', 'dqpy.b1',], step=1e-3, tag='sext'),
                    ],
                targets=[xt.TargetSet(dqx=target_dqx, dqy=target_dqy, tol=1e-2, tag='chrom'),
                    ])
            opt_chroma.target_status()
            opt_chroma.vary_status()

    else:
        raise Exception('The matching of asked machine is not suported yet!')

    return


def add_chroma_knobs(line, optics_type=None):

    tt0 = line.get_table(attr=True)

    if optics_type is None:
        if len(tt0.rows['sy.*'].name) == 0:
            optics_type = 'LCC'
        else:
            optics_type = 'GHC'

    line.vars['sf.k2n.chroma.knob'] = 1
    line.vars['sd.k2n.chroma.knob'] = 1
    if optics_type == 'GHC':
        for ii in tt0.rows['sf.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sf.k2n.chroma.knob']*line.element_refs[ii].k2._expr

        for ii in tt0.rows['sd.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sd.k2n.chroma.knob']*line.element_refs[ii].k2._expr

    elif optics_type == 'LCC':
        for ii in tt0.rows['sf[12]a.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sf.k2n.chroma.knob']*line.element_refs[ii].k2._expr

        for ii in tt0.rows['sd[12]a.*'].name:
            if '_aper' not in ii:
                line.element_refs[ii].k2 = line.vars['sd.k2n.chroma.knob']*line.element_refs[ii].k2._expr

    return


def initial_conditions_grid (study, energy_spread=None, cartesian_polar=None, min_r_y=None, max_r_y=None, num_r_y_points=None, min_theta_x=None, max_theta_x=None, num_theta_x_points=None, r_range_x=(5,10), r_range_y=(5,10), theta_range_x=(0,2*np.pi), theta_range_y=(0,2*np.pi), delta_initial_values=None, num_particles=None, rnd_seed=105):

    """
    Generate initial conditions grid for a given study.

    Parameters
    ----------
    study : str
        Study type. Can be 'DA', 'MA' or 'halo'.
    energy_spread : float
        Energy spread of the beam.
    cartesian_polar : str
        Type of initial condition for 'DA' and 'MA' studies. Can be 'cartesian' or 'polar'.
    min_r_y : float
        Minimum y for cartesian initial conditions or minimum radius for polar.
    max_r_y : float
        Maximum y for cartesian initial conditions or maximum radius for polar.
    num_r_y_points : int
        Number of points in y or radial plane.
    min_theta_x : float
        Minimum x for cartesian initial conditions or minimum theta for polar.
    max_theta_x : float
        Maximum x for cartesian initial conditions or minimum theta for polar.
    num_theta_x_points : int
        Number of points in x or theta plane.
    r_range_x : tuple
        Range of radial distances in phase space for horizontal coordinates for halo.
    r_range_y : tuple
        Range of radial distances in phase space for vertical coordinates for halo.
    theta_range_x : tuple
        Range of thata angles in phase space for horizontal coordinates for halo.
    theta_range_y : tuple
        Range of thata angles in phase space for vertical coordinates for halo.
    delta_initial_values : array_like
        Initial values of delta.
    num_particles : int
        Number of particles needed only for halo distribution.
    rnd_seed : int
        Random number seed.

    Returns for 'MA' and 'DA'
    -------
    x_normalized : array_like
        Normalized x coordinates of the particles.
    y_normalized : array_like
        Normalized y coordinates of the particles.
    delta_init : array_like
        Initial values of delta.
    num_theta_x_points : int
        Number of points in x plane.
    num_r_y_points : int
        Number of points in y plane.
    num_delta : int
        Number of initial values of delta.
    num_particles : int
        Number of particles.

    Returns for 'halo'
    -------
    x_normalized : array_like
        Normalized x coordinates of the particles.
    y_normalized : array_like
        Normalized y coordinates of the particles.
    px_normalized : array_like
        Normalized px coordinates of the particles.
    py_normalized : array_like
        Normalized py coordinates of the particles.
    """

    np.random.seed(rnd_seed)

    if min_r_y is None:
        if study in ['DA','MA']:
            min_r_y = 0
    
    if max_r_y is None:
        if study=='DA':
            max_r_y = 50
        elif study=='MA':
            max_r_y = 30

    if num_r_y_points is None:
        if study=='DA':
            num_r_y_points = 71 #51
        elif study=='MA':
            num_r_y_points = 31

    if min_theta_x is None:
        if study=='DA':
            min_theta_x = -30 #-20
        elif study=='MA':
            min_theta_x = np.pi/4

    if max_theta_x is None:
        if study=='DA':
            max_theta_x= 30 #20
        elif study=='MA':
            max_theta_x= np.pi/4

    if num_theta_x_points is None:
        if study=='DA':
            num_theta_x_points = 61 #41
        elif study=='MA':
            num_theta_x_points = 1

    if delta_initial_values is None:
        if study=='DA':
            delta_initial_values = 0
        elif study=='MA':
            delta_initial_values = np.linspace(-25*energy_spread, 25*energy_spread, 51) 

    
    if study=='DA':
        if cartesian_polar is None or cartesian_polar=='cartesian':
            x_norm_points = np.linspace(min_theta_x, max_theta_x, num_theta_x_points)
            y_norm_points = np.linspace(min_r_y, max_r_y, num_r_y_points)
            x_norm_grid, y_norm_grid = np.meshgrid(x_norm_points, y_norm_points)
            x_normalized = x_norm_grid.flatten()
            y_normalized = y_norm_grid.flatten()

        elif cartesian_polar=='polar':
            x_normalized, y_normalized, r_xy, theta_xy = xp.generate_2D_polar_grid(
                r_range=(min_r_y, max_r_y), # beam sigmas
                theta_range=(min_theta_x, max_theta_x),
                nr=num_r_y_points, ntheta=num_theta_x_points)
            
    if study=='MA':
        if cartesian_polar is None or cartesian_polar=='polar':
            x_normalized, y_normalized, r_xy, theta_xy = xp.generate_2D_polar_grid(
                r_range=(min_r_y, max_r_y), # beam sigmas
                theta_range=(min_theta_x, max_theta_x),
                nr=num_r_y_points, ntheta=num_theta_x_points)
            
        elif cartesian_polar=='cartesian':
            x_norm_points = np.linspace(min_theta_x, max_theta_x, num_theta_x_points)
            y_norm_points = np.linspace(min_r_y, max_r_y, num_r_y_points)
            x_norm_grid, y_norm_grid = np.meshgrid(x_norm_points, y_norm_points)
            x_normalized = x_norm_grid.flatten()
            y_normalized = y_norm_grid.flatten()

    if study in ['DA','MA']:
        num_delta = np.size(delta_initial_values)
        num_particles = num_delta*num_theta_x_points*num_r_y_points
        if num_delta != 1:
            x_normalized = np.tile(x_normalized, num_delta)
            y_normalized = np.tile(y_normalized, num_delta)
            delta_init = np.repeat(delta_initial_values, np.size(x_normalized)/num_delta)
        else:
            delta_init = delta_initial_values
    
    return (x_normalized, y_normalized, delta_init, num_theta_x_points, num_r_y_points, num_delta, num_particles)


def generate_particle_grid(line, study_param, particle_capacity=None, cartesian_polar=None, 
                            min_r_y=None, max_r_y=None, num_r_y_points=None, min_theta_x=None, max_theta_x=None, 
                            num_theta_x_points=None, delta_initial_values=None, 
                            beambeam_strength_used=None, radiation_off=False, rnd_seed=101):
    #r_range_x=(5,10), r_range_y=(5,10), theta_range_x=(0,2*np.pi), theta_range_y=(0,2*np.pi),
    if radiation_off:
        wig_v = line.vv['on_wiggler_v']
        SR_model = line._radiation_model
        BS_model = line._beamstrahlung_model
        line.vars['on_wiggler_v']=0
        line.configure_radiation(model=None, model_beamstrahlung=None)
    if beambeam_strength_used is not None:
        tt = line.get_table(attr=True)
        bb_elem_name = tt.rows[tt.element_type=='BeamBeamBiGaussian3D'].name
        dd_bb = {}
        for jj in bb_elem_name:
            dd_bb[jj] = line[jj].scale_strength
            line[jj].scale_strength = beambeam_strength_used

    tw = line.twiss(eneloss_and_damping=True)
    ref_part = line.particle_ref
    if 'grid' in study_param['ini_cond_type']:
        # The longitudinal closed orbit needs to be manually supplied for now
        zeta_co = tw.zeta[0] 
        delta_co = tw.delta[0]

        if any(ii in study_param['ini_cond_type'] for ii in ['DA', 'MA']):
            found = next(ii for ii in ['DA', 'MA'] if ii in study_param['ini_cond_type'])
            (x_normalized, y_normalized, delta_init, num_theta_x_points, num_r_y_points, num_delta, num_particles) = initial_conditions_grid (found, 
                                                                                                                                             study_param['ini_cond_energy_spread'],
                                                                                                                                             cartesian_polar=cartesian_polar, 
                                                                                                                                             min_r_y=min_r_y, max_r_y=max_r_y, num_r_y_points=num_r_y_points, 
                                                                                                                                             min_theta_x=min_theta_x, max_theta_x=max_theta_x, num_theta_x_points=num_theta_x_points, 
                                                                                                                                             delta_initial_values=delta_initial_values, rnd_seed=rnd_seed)
            px_normalized = 0 
            py_normalized = 0
            zeta = zeta_co
            delta = delta_init + delta_co

            grid_details = {
            'num_theta_x_points':num_theta_x_points,
            'num_r_y_points':num_r_y_points,
            'num_delta':num_delta,
            'num_particles':num_particles,
            'x_normalized':x_normalized,
            'y_normalized':y_normalized,
            'delta_init':delta_init
            }
        
        else:
            raise ValueError(f"Unknown initial condition!")
        
        if particle_capacity is None:
            capacity = len(x_normalized)
        else:
            capacity = particle_capacity
        
        particles = line.build_particles(_capacity=capacity,
                x_norm=x_normalized, 
                y_norm=y_normalized,  
                px_norm=px_normalized,
                py_norm=py_normalized,
                nemitt_x=study_param['ini_cond_nemittance_x'], 
                nemitt_y=study_param['ini_cond_nemittance_y'], 
                zeta= zeta, 
                delta= delta) 
        
    if radiation_off:
        line.vars['on_wiggler_v']=wig_v
        line.configure_radiation(model=SR_model, model_beamstrahlung=BS_model)
    if beambeam_strength_used is not None:
        for jj in bb_elem_name:
            line[jj].scale_strength = dd_bb[jj]

    return (particles, grid_details)
