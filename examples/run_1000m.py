import os
import numpy 
from multiprocessing import Process
from ad_coyote_lake import merge_datasets
from ad_coyote_lake import advection_diffusion_with_wind_model 

use_multiprocessing = True

diffusion_coeff_array = numpy.append(
        numpy.arange(10.0,  100.0,  10.0 ), 
        numpy.arange(100.0 ,1100.0 ,100.0)
        )
num_coeff = diffusion_coeff_array.shape[0]

wind_param = {
        'type'       : 'file',
        'filename'   : 'wind_speed_pdf.txt',
        'num_speed'  : 50, 
        'frac_start' : 0.05, 
        'frac_stop'  : 0.85,
        }

base_param = {
        'boundary_radius'   : 1000.0,  # (m)
        'time_step'         : 10.0,    # (s) 
        'num_boundary_edge' : 100,
        'mesh_resolution'   : 100,
        'pvd_file_name'     : 'output.pvd',
        'pvd_save_modulo'   : False, 
        'plot_domain_mesh'  : False,
        'plot_results'      : False,
        'print_info'        : False,
        'g_traj_min'        : 0.2,
        'wind_param'        : wind_param, 
        'initial_cond_ref'  : {'t_ref': 60.0, 'diffusion_coeff_ref': 10.0},
        'output_directory'  : 'data_1000m',
        }

# Run simulations
for i, diffusion_coeff in enumerate(diffusion_coeff_array):
    print('{}/{}, D: {:0.1f}'.format(i+1, num_coeff, diffusion_coeff))
    if use_multiprocessing:
        # Running separate processes seems to prevent 'out of memory errors' when running long batches 
        proc = Process(target=advection_diffusion_with_wind_model, args=(diffusion_coeff, base_param))
        proc.start()
        proc.join()
    else:
        advection_diffusion_with_wind_model(diffusion_coeff, base_param)
    print()

# Merge datasets
dataset_list = [f for f in os.listdir(base_param['output_directory']) if os.path.splitext(f)[1]=='.pkl']
dataset_list = [os.path.join(base_param['output_directory'],f) for f in dataset_list]
dataset_list.sort()

max_diffusion_coeff = diffusion_coeff_array.max()
min_diffusion_coeff = diffusion_coeff_array.min()

merged_dataset_file = 'merged_adv_sim_data_D{:0.0f}_D{:0.0f}_R{:0.0f}.pkl'.format(
            min_diffusion_coeff, 
            max_diffusion_coeff, 
            base_param['boundary_radius']
            )
merged_dataset_file = os.path.join(base_param['output_directory'], merged_dataset_file)
merge_datasets(dataset_list, merged_dataset_file)








