import os
import math
import numpy as np
import matplotlib.pyplot as plt
import mshr
from dolfin import *
from .wind_speed import wind_speeds_from_file
from .wind_speed import wind_speeds_from_normal

def advection_diffusion_with_wind_model(diffusion_coeff, base_param):
    """

    Solves the advection diffusion on a closed disk with absorbing boundary
    condition  given the diffusion coefficient and a dictionary of simulation
    parameters. Solution is found for range velocities specified by a set of 
    wind parameters. 

    Aguments:
      diffusion_coeff :  diffusion coefficient (m**2/s) 
      base_param      :  dictionary of simulation parameters where

      base_param = { 
        'boundary_radius'   :  radius of closed disk (m)
        'time_step'         :  time step for simulation (s) 
        'num_boundary_edge' :  number of boundary edges (used for flux calcs) 
        'mesh_resolution'   :  resultion of mesh,
        'pvd_file_name'     :  pvd filename for full density output
        'pvd_save_modulo'   :  step modulo for saving full density output 
        'plot_domain_mesh'  :  flag (True/False) for plot mesh prior to start
        'plot_results'      :  flag (True/False) for ploting results and end
        'print_info'        :  flag (True/False) for printing info during run
        'g_traj_min'        :  minimum g_traj = R/t to achieve, sets t_end
        'wind_param'        :  dictionary of parameters for wind velocities
        'initial_cond_ref'  :  dictionary of parameters specifying initial condition 
        'output_directory'  :  name of directory for data files
        }
      
      
      Currently two types of wind parameters are allowed 'normal' and 'file'

      When the wind parameter type is 'normal' the wind values are taken from a 
      normal distribtion. 

      wind_param = { 
        'type'    : 'normal' 
        'mu'      :  mean value (m/s) of normal distribution
        'sigma'   :  standard deviation (m) of normal distribution 
        'num_pts' :  number of evenly spaced sample points to use from distribtion
        'cutoff'  :  cutoff probability for setting sample point range
        }
 
      When the wind parameters type is 'file' the wind values are taken from the 
      pdf specified in the given file. 

      wind_param = { 
        'type'       : 'file'
        'filename'   : name of wind parameter file
        'num_speed'  : number of evenly spaced sample points to use from ditribution 
        'frac_start' : start fraction (specifies lower bound of range for sample points) 
        'frac_stop'  : stop fraction (specifies upper bound of range for sample points)
        }

      The initial_cond_ref is used to specify the time for evolving the delta function 
      specifying  the initial condition. It is given in terms of a reference time value 
      and a reference diffusion coefficient.  
      
      initial_cond_ref =  {
        't_ref'               :  reference time delta function evolution  
        'diffusion_coeff_ref' :  reference diffusion coeff for delta function evolution
        },

    """

    # Check to see if output directory exists if not create it
    if not os.path.exists(base_param['output_directory']):
        os.mkdir(base_param['output_directory'])

    # Create output filename
    output_file = 'adv_sim_data_D{:0.0f}_R{:0.0f}.pkl'.format(
            diffusion_coeff, 
            base_param['boundary_radius']
            )
    output_file = os.path.join(base_param['output_directory'], output_file)
    print('output_file: {}'.format(output_file))
    if os.path.exists(output_file):
        raise RuntimeError(f'output file: {output_file} exits!')

    # Get list of wind speeds and weight values based on wind parameters
    wind_param = base_param['wind_param']
    print('wind_param:')
    for k,v in wind_param.items():
        print('{}: {}'.format(k,v))

    if wind_param['type'] == 'normal':
        speed_list, weight_list = wind_speeds_from_normal(
                wind_param['mu'], 
                wind_param['sigma'],
                wind_param['num_pts'],
                wind_param['cutoff'], 
                normalize = True,
                plot=False,
                )
    elif wind_param['type'] == 'file':
        speed_list, weight_list = wind_speeds_from_file(
                wind_param['filename'], 
                wind_param['num_speed'], 
                wind_param['frac_start'], 
                wind_param['frac_stop'], 
                plot=False
                )
    else:
        raise ValueError('unknown wind_param type: {}'.format(wind_param['type']))

    num_speed = speed_list.shape[0]
    print('num_speed: {}'.format(len(speed_list)))
    print()

    t_ref = base_param['initial_cond_ref']['t_ref']
    diffusion_coeff_ref = base_param['initial_cond_ref']['diffusion_coeff_ref']

    with open(output_file,'wb') as f: 

        # Loop over velocities and solve advection diffuction equation
        for j, speed, weight in zip(range(num_speed), speed_list, weight_list): 
            info_tuple = (j+1, num_speed, diffusion_coeff, speed, weight)
            print('  {}/{}, D: {:0.1f}, speed: {:0.3f}, weight: {:0.3f}'.format(*info_tuple))
            param = dict(base_param)
            param['velocity'] = float(speed)
            param['weight']  = float(weight)
            param['diffusion_coeff'] = float(diffusion_coeff)
            param['initial_cond_dt'] = t_ref*diffusion_coeff_ref/diffusion_coeff
            param['termination'] = {
                    'type'  :  'time', 
                    'value' :   param['boundary_radius']/param['g_traj_min'], 
                    }
            result = advection_diffusion(param)
            pickle.dump(result, f)



def advection_diffusion(param):
    """ 
    Solves the advection-diffusion equation on a closed disk with absorbing
    boundary condition given the diffusion coefficient and a dictionary of
    simulation parameters. 

    Arguments:

      param = dictionary of run parameters with the folliwng keys
      param = {
              'diffusion_coeff'   : 1.0,   # (m**2/s)
              'boundary_radius'   : 1000,  # (m)
              'velocity'          : 0.14,  # (m/s)
              'time_step'         : 10.0,  # (s) 
              'initial_cond_dt'   : 20.0,  # (s)
              'num_boundary_edge' : 100,
              'mesh_resolution'   : 100,
              'pvd_file_name'     : 'output.pvd',
              'pvd_save_modulo'   : False, 
              'plot_domain_mesh'  : False,
              'plot_results'      : True,
              'print_info'        : True,
              'termination'       : { 
                  'type'          : 'exit_prob',
                  'value'         : 0.90, 
                  }
              }

    Return:

      result = dictionary of simulation results with the following keys
      result = { 
              # Flux section data
              'flux_section' : {
                  't'      : t_grid,             # time grid
                  'angle'  : angle_grid,         # angle grid
                  'density': flux_section_grid,  # probability density
                  },
              # Exit probablity data
              'exit': {
                  't'    : t_array,              # time points
                  'prob' : prob_exit_array,      # exit probability
                  },
              'param': param,                    # simulation parameters
              }

    """
    # Extract parameters
    D = float(param['diffusion_coeff'])
    R = float(param['boundary_radius'])
    V = float(param['velocity'])
    dt = float(param['time_step'])
    t0 = float(param['initial_cond_dt'])
    termination = param['termination']
    num_boundary_edge = param['num_boundary_edge']
    mesh_resolution = param['mesh_resolution']
    pvd_file_name = param['pvd_file_name']
    pvd_save_modulo = param['pvd_save_modulo']
    plot_domain_mesh = param['plot_domain_mesh']
    plot_results = param['plot_results']
    print_info = param['print_info']

    # Create mesh
    domain = mshr.Circle(Point(0.0,0.0), R, num_boundary_edge) 
    mesh = mshr.generate_mesh(domain, mesh_resolution)
    if plot_domain_mesh:
        plot(mesh)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.show()
        assert 1==0

    # Create FunctionSpaces
    Q = FunctionSpace(mesh, "CG", 3)  # using 2 or 3 improves boundary integrals 
    #V = VectorFunctionSpace(mesh, "CG", 1)
    
    # Set initial condition
    expr_str = '(1.0/(4.0*pi*D*t))*exp(-(pow(x[0]-a,2)+pow(x[1]-b,2))/(4.0*D*t))'
    u_0 = Expression(expr_str, a=V*t0, b=0.0, pi=math.pi, D=D, t=t0, degree=3)

    # Initialize solution from previous time step
    u_n = interpolate(u_0,Q)

    # Create velocity field 
    velocity = Constant((V,0.0))
    
    # Test and trial functions
    u = TrialFunction(Q) 
    v = TestFunction(Q)
    
    # Mid-point solution (Crank-Nicolson)
    u_last = Function(Q)
    u_mid = 0.5*(u_n + u)
    
    # Galerkin variational problem
    F = v*(u-u_n)*dx + dt*(v*dot(velocity, grad(u_mid))*dx + D*dot(grad(v), grad(u_mid))*dx)
    
    # Create bilinear and linear forms
    a = lhs(F)
    L = rhs(F)
    
    # Assemble matrix
    A = assemble(a)

    # Boundary condition
    boundary = Boundary()
    boundary_meshfunc = MeshFunction('size_t', mesh, 1)
    boundary_meshfunc.set_all(0)
    boundary.mark(boundary_meshfunc,1)
    bc = DirichletBC(Q, Constant(0.0), boundary)
    bc.apply(A)

    # Create sections for flux calculations 
    flux_section_angles = np.linspace(-np.pi, np.pi, num_boundary_edge+1)
    flux_section_angles = flux_section_angles[1:-1:2]
    flux_section_width = flux_section_angles[1] - flux_section_angles[0]

    angle_to_flux_section_ind = {}
    flux_section_meshfunc = MeshFunction('size_t', mesh, 1)
    flux_section_meshfunc.set_all(0)
    flux_section_list = []
    for i, angle in enumerate(flux_section_angles):
        ind = len(flux_section_angles) - i 
        flux_section = FluxSection(angle, flux_section_width)
        flux_section_list.append(flux_section)
        flux_section.mark(flux_section_meshfunc,ind)
        angle_to_flux_section_ind[angle] = ind

    normal = FacetNormal(mesh) # used for flux calculations

    # Create linear solver and factorize matrix
    solver = LUSolver(A)

    # Set initial condition 
    u = u_n 
    t = t0 + dt 
    i = 1

    # Save initial state
    if pvd_save_modulo:
        print('saving to pvd: {}'.format(0))
        outfile = File(pvd_file_name)
        outfile << u

    # Initialize last flux values (for cumulative flux integration)
    flux_value_last = 0
    flux_value_check_last = 0

    # Initialize cumulative flux values 
    cum_flux_value = 0
    cum_flux_value_check = 0

    # Initialize data containers
    t_list = []
    survival_list = []
    cum_flux_list = []
    cum_flux_check_list = []
    angle_to_flux_list = {angle:[] for angle in flux_section_angles}

    done = False
    while not done:
    
        t_list.append(t)

        # Survival probability
        survival_prob = assemble(u*dx)
        survival_list.append(survival_prob)
    
        # Flux across whole boundary
        ds_boundary = Measure('ds')(subdomain_data=boundary_meshfunc)
        flux = dot(velocity*u - D*grad(u) ,normal)*ds_boundary(1)
        flux_value = assemble(flux)
        cum_flux_value += 0.5*(flux_value + flux_value_last)*dt
        flux_value_last = flux_value
        cum_flux_list.append(cum_flux_value)
        err = survival_prob - (1.0 - cum_flux_value)
    
        # Flux across boundary sections
        ds_flux = Measure('ds')(subdomain_data=flux_section_meshfunc)
        flux_value_check = 0
        for angle in flux_section_angles:
            ind = angle_to_flux_section_ind[angle]
            flux_at_ind = dot(velocity*u - D*grad(u), normal)*ds_flux(ind)
            flux_value_at_ind = assemble(flux_at_ind)
            flux_value_check += flux_value_at_ind 
            angle_to_flux_list[angle].append(flux_value_at_ind/(flux_section_width))
    
        cum_flux_value_check += 0.5*(flux_value_check + flux_value_check_last)*dt
        flux_value_check_last = flux_value_check
        cum_flux_check_list.append(cum_flux_value_check)
    
        if print_info:
            survival_prob_flux = 1.0 - cum_flux_value
            survival_prob_flux_check = 1.0 - cum_flux_value_check
            info_tuple = (t, survival_prob, survival_prob_flux,survival_prob_flux_check, err)
            print('{:0.3f}, {:0.6f}, {:0.6f}, {:0.6f}, {:0.6f}'.format(*info_tuple))
    
        # Assemble vector and apply boundary conditions
        b = assemble(L)
        bc.apply(b)
    
        # Solve the linear system (re-use the already factorized matrix A)
        solver.solve(u.vector(), b)
    
        # Copy solution from previous interval
        u_n.assign(u)
    
        # Save solution for paraview plotting 
        if pvd_save_modulo and i%pvd_save_modulo == 0:
            if print_info:
                print('saving to pvd: {}'.format(i))
            outfile << u
    
        # Check termination condition
        if termination['type'] == 'exit_prob':
            if cum_flux_value > termination['value']:
                done = True
        elif termination['type'] == 'time':
            if t >= termination['value']:
                done = True
        else:
            err_msg = 'unknown termination condition: {}'.format(termination['type'])
            raise(RuntimeError,err_msg) 

        # Move to next interval and adjust boundary condition
        t += dt
        i += 1

    # Convert data to numpy arrays
    t_array = np.array(t_list)
    survival_array = np.array(survival_list)
    cum_flux_array = np.array(cum_flux_list)
    cum_flux_check_array = np.array(cum_flux_check_list)
    prob_exit_array = 1.0 - survival_array
    arrive_t_prob = (prob_exit_array[1:] - prob_exit_array[:-1])/dt

    num_t = len(t_list)
    flux_section_angles = np.array(flux_section_angles)
    t_grid, angle_grid = np.meshgrid(t_array, flux_section_angles)
    flux_section_grid = np.zeros(t_grid.shape)
    for i, angle in enumerate(flux_section_angles):
        flux_section_grid[i,:] = np.array(angle_to_flux_list[angle])
    
    if plot_results:
        plt.figure()
        plt.plot(t_array, survival_array,'b')
        plt.plot(t_array, 1.0 - cum_flux_array,'r')
        plt.plot(t_array, 1.0 - cum_flux_check_array,'g')
        plt.grid(True)
        plt.xlabel('t (sec)')
        plt.ylabel('prob survive to t ')
        
        plt.figure()
        plt.plot(t_array, prob_exit_array)
        plt.grid(True)
        plt.xlabel('t (sec)')
        plt.ylabel('prob exit by t')
        
        plt.figure()
        plt.plot(t_array[1:], arrive_t_prob)
        plt.grid(True)
        plt.xlabel('t (sec)')
        plt.ylabel('exit prob density')
        
        plt.figure()
        plt.pcolor(t_grid, np.rad2deg(angle_grid), flux_section_grid, cmap='binary')
        plt.xlabel('t (secs)')
        plt.ylabel('angle (deg)')
        
        plt.show()


    result = { 
            # Flux section data
            'flux_section' : {
                't'      : t_grid,             # time grid
                'angle'  : angle_grid,         # angle grid
                'density': flux_section_grid,  # probability density
                },
            # Exit probablity data
            'exit': {
                't'    : t_array,          # time points
                'prob' : prob_exit_array,  # exit probability
                },
            # Simulation parameters
            'param': param,  
            }

    return result



class Boundary(SubDomain):

    """ Domain boundary """

    def inside(self, x, on_boundary):
        return on_boundary


class FluxSection(SubDomain):

    """ Sections for finding around boundary """

    def __init__(self, angle, width):
        super().__init__()
        self.angle = angle
        self.width = width
        self.tol = 1.0e-6

    def inside(self, x, on_boundary):
        x_angle = np.arctan2(x[1],x[0])
        if self.angle > np.pi/2:
            if x_angle < 0:
                x_angle = x_angle + 2*pi
        elif self.angle < -np.pi/2:
            if x_angle > 0:
                x_angle = x_angle - 2*pi
        lower = self.angle - (0.5*self.width + self.tol)
        upper = self.angle + (0.5*self.width + self.tol)
        in_wedge =  x_angle > lower and x_angle < upper
        return on_boundary and in_wedge




# -----------------------------------------------------------------------------
if __name__ == '__main__':

    import pickle

    param = {
            'diffusion_coeff'   : 1.0,   # (m**2/s)
            'boundary_radius'   : 1000,  # (m)
            'velocity'          : 0.14,  # (m/s)
            'time_step'         : 10.0,  # (s) 
            'initial_cond_dt'   : 20.0,  # (s)
            'num_boundary_edge' : 100,
            'mesh_resolution'   : 100,
            'pvd_file_name'     : 'output.pvd',
            'pvd_save_modulo'   : False, 
            'plot_domain_mesh'  : False,
            'plot_results'      : True,
            'print_info'        : True,
            'termination'       : { 
                'type'          : 'exit_prob',
                'value'         : 0.90, 
                }
            }

    result = advection_diffusion(param)

    with open('test_save.pkl','wb') as f: 
        pickle.dump(result, f)




