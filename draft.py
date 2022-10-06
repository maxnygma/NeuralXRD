import os
import random
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def create_df_from_xrd_files(path_to_xrd_files='xrd_patterns'):
    data = pd.DataFrame(columns=['id', 'material', '2theta', 'intensity'])

    for filename in tqdm(os.listdir(path_to_xrd_files)):
        material_data = pd.read_csv(f'{path_to_xrd_files}/{filename}', sep='\t')

        material_data.columns = ['2theta', 'intensity']
        material_data['material'] = filename.split('_')[0]
        material_data['id'] = filename.split('_')[1]


        data = pd.concat([data, material_data]).reset_index(drop=True)

        # plt.plot(data['2_theta'], data['intensity'])
        # plt.show()

    return data


def generate_synthetic_phases(data):
    ''' Generate synthetic phases by combining phases of two distinct materials '''

    # Get sample ids for all of materials
    material_ids_dict = {}
    for material in data['material'].unique():
        material_ids = np.unique(data['id'][data['material'] == material].values)
        material_ids_dict[material] = material_ids

    # Iterate over materials
    for material_1 in tqdm(material_ids_dict):
        for material_2 in material_ids_dict:
            if material_1 == material_2:
                continue

            ids_1 = material_ids_dict[material_1]
            ids_2 = material_ids_dict[material_2]

            # Iterate over ids for two selected materials
            for id_1 in ids_1:
                for id_2 in ids_2:
                    # Check if combination is already processed
                    if (f'{id_1}_{id_2}_1' in data['id'].values) or (f'{id_2}_{id_1}_1' in data['id'].values):
                        continue
                    
                    # Get data of two materials
                    material_data_1 = data[data['id'] == id_1]
                    material_data_2 = data[data['id'] == id_2]

                    # Shift intensity by a random 2theta
                    for i in range(5):
                        material_data_1['intensity'] = material_data_1['intensity'].shift(random.randint(0, 50), fill_value=0.0) # what degree shift is ok? (step 0.2)
                        material_data_2['intensity'] = material_data_2['intensity'].shift(random.randint(0, 50), fill_value=0.0)

                        # Scale intensity by a random value
                        material_data_1['intensity'] *= random.uniform(0.5, 1.5)
                        material_data_2['intensity'] *= random.uniform(0.5, 1.5)

                        # Combine phases
                        stacked_material_intensity = material_data_1['intensity'].values + material_data_2['intensity'].values

                        synthetic_phase_data = pd.DataFrame({'id': id_1 + '_' + id_2 + '_' + str(i), 
                                                            'material': material_1 + '_' + material_2,
                                                            '2theta': material_data_1['2theta'].values,
                                                            'intensity': stacked_material_intensity})

                        data = pd.concat([data, synthetic_phase_data]).reset_index(drop=True)

    return data


data = create_df_from_xrd_files(path_to_xrd_files='xrd_patterns')
initial_shape = data.shape[0]

data = generate_synthetic_phases(data)
print(data)
print((data.shape[0] - initial_shape) / 2250)

random_idx = random.choice(data['id'].unique())
plt.plot(data['2theta'][data['id'] == random_idx], data['intensity'][data['id'] == random_idx], label=data['material'][data['id'] == random_idx].values[0])
plt.legend()
plt.show()


# plt.plot(data['2theta'][data['id'] == '1100108'], data['intensity'][data['id'] == '1100108'], label='1100108')
# plt.plot(data['2theta'][data['id'] == '5000220'], data['intensity'][data['id'] == '5000220'], label='5000220')
# plt.legend()
# plt.show()


