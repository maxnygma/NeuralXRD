import os
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


def create_df_from_xrd_files(path_to_xrd_files='xrd_patterns'):
    data = pd.DataFrame(columns=['material', '2theta', 'intensity'])

    for filename in tqdm(os.listdir(path_to_xrd_files)):
        material_data = pd.read_csv(f'{path_to_xrd_files}/{filename}', sep='\t')

        material_data.columns = ['2theta', 'intensity']
        material_data['material'] = filename[:-4]

        data = pd.concat([data, material_data]).reset_index(drop=True)

        # plt.plot(data['2_theta'], data['intensity'])
        # plt.show()

    return data

data = create_df_from_xrd_files(path_to_xrd_files='xrd_patterns')

print(data)

# for material in data['material'].unique():
#     random_coeff = np.random.uniform(1, 1)
#     plt.plot(data['2theta'][data['material'] == material], data['intensity'][data['material'] == material] * random_coeff, label=material)
plt.plot(data['2theta'][data['material'] == 'cu_zn'], data['intensity'][data['material'] == 'cu_zn'], label='cu_zn')
plt.plot(data['2theta'][data['material'] == 'zn'], data['intensity'][data['material'] == 'zn'], label='zn')
plt.legend()
plt.show()


