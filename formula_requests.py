import os
import requests
from tqdm import tqdm

import pandas as pd


# data = pd.DataFrame(columns=['cif_id', 'formula'])
cif_files = []; formulas = []

for dir in tqdm(os.listdir('/home/xzcodes/Downloads/cod/cif')):
    for dir2 in tqdm(os.listdir(f'/home/xzcodes/Downloads/cod/cif/{dir}')):
        for dir3 in tqdm(os.listdir(f'/home/xzcodes/Downloads/cod/cif/{dir}/{dir2}')):
            for filename in os.listdir(f'/home/xzcodes/Downloads/cod/cif/{dir}/{dir2}/{dir3}'):
                response = requests.get(f'http://www.crystallography.net/cod/{filename}').text
                
                idx = response.index('_chemical_formula_sum')

                elements = response[idx:idx + len('_chemical_formula_sum') + 50].split()[1:-1]
                elements[0] = elements[0][1:]
                elements[-1] = elements[-1][:-1]

                element = ''.join(elements)

                cif_files.append(filename)
                formulas.append(element)
            
            data_part = pd.DataFrame({'cif_id': cif_files, 'formula': formulas})
            print(data_part.shape)
            data_part.to_csv('data.csv', index=False)



# filename = '1000099.cif'
# response = requests.get(f'http://www.crystallography.net/cod/{filename}').text

# idx = response.index('_chemical_formula_sum')
# elements = response[idx:idx + len('_chemical_formula_sum') + 50].split()[1:-1]
# elements[0] = elements[0][1:]
# elements[-1] = elements[-1][:-1]

# element = ''.join(elements)

# print(element)