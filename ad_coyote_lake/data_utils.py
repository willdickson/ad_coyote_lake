import os
import pickle


def merge_datasets(file_list, output_file):
    """
    Merge individual datasets into one large dataset. 
    """
    data_list = []
    print()
    print('loading data')
    for i, file_name in enumerate(file_list):
        print('  {}/{}: {}'.format(i+1,len(file_list), file_name))
        data_list.extend(load_sim_data(file_name))
    print('done')
    print()
    
    data_dict = {}
    for data in data_list:
        coeff = data['param']['diffusion_coeff']
        data['param']['diffusion_coeff'] = float(coeff)
        try:
            data_dict[coeff].append(data)
        except KeyError:
            data_dict[coeff] = [data]
    
    coeff_list = sorted([k for k in data_dict])
    
    print('creating merged file: {}'.format(output_file))
    with open(output_file,'wb') as f:
        for coeff in coeff_list:
            print('coeff: {:0.2f}'.format(coeff))
            for data in data_dict[coeff]:
                pickle.dump(data,f,protocol=2)
        print('done')
    print()


def load_sim_data(filename):
    """
    Load pickled dataset and return as list. 
    """
    with open(filename, 'rb') as f:
        done = False
        data_list = []
        while not done:
            try:
                data = pickle.load(f)
                data_list.append(data)
            except EOFError as err:
                done = True
    return data_list

