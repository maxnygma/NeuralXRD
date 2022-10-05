import os
import time
import pyautogui
from tqdm import tqdm
from urllib.request import urlopen

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


def get_cif_files_for_element(material):
    ''' Request and download cif files based on URL with a specific material mentioned '''

    for page in range(100):
        # Ti
        data_url = pd.read_html(f'http://www.crystallography.net/cod/result.php?CODSESSION=m3gu2ds6gnlp9pnbdvv3ptf87a&count=1000&page={page}&order_by=file&order=asc')[0]

        if len(data_url) == 0:
            break

        for cod_id in tqdm(data_url[data_url.columns[0]].values):
            url = f'http://www.crystallography.net/cod/{cod_id}.cif'
            save_as = f'/home/xzcodes/projects/xray_diff/{material}_cif_files/{cod_id}.cif'

            with urlopen(url) as file:
                content = file.read().decode()

            with open(save_as, 'w') as d:
                d.write(content)


def get_dataframe_from_cif_files(material):
    ''' Create a DataFrame from cif files and compound names written in them '''

    files = sorted(os.listdir(f'{material}_cif_files'))
    files = [file[:-4] for file in files if '.cif' in file]

    list_of_names = []

    for filename in files:
        with open(f'{material}_cif_files/{filename}.cif', 'r') as f:
            cif_file = f.read()

        idx = cif_file.index('_chemical_formula_sum')

        elements = cif_file[idx:idx + len('_chemical_formula_sum') + 50].split()[1:-1]
        elements[0] = elements[0][1:]
        elements[-1] = elements[-1][:-1]

        element = ''.join(elements)
        
        list_of_names.append(element)

    data = pd.DataFrame({'cod_id': files, 'compound_name': list_of_names})
    data['material'] = material

    return data


def generate_powder_pattern(material):
    pyautogui.PAUSE = 0.5

    files = sorted(os.listdir('cif_files'))
    # files = sorted(os.listdir(f'{material}_cif_files'))

    tsv_files = [file for file in files if '.tsv' in file]
    files_to_skip = [file.replace('tsv', 'cif')[3:] for file in tsv_files]
    files = [file for file in files if '.cif' in file]

    print(files) 
    print(len(files), len(files_to_skip))

    with pyautogui.hold('alt'):
        pyautogui.press('tab')

    pyautogui.moveTo(350, 80) # click "Calculate" in Mercury
    pyautogui.mouseDown(); pyautogui.mouseUp()
    # pyautogui.mo1523987.cifuseDown(); pyautogui.mouseUp()

    pyautogui.moveTo(350, 270) # click "Powder Pattern" in Mercury
    pyautogui.mouseDown(); pyautogui.mouseUp()

    for i, material_cif in enumerate(files):
        if material_cif in files_to_skip:
            print(material_cif, 'skipped')
            continue
    
        pyautogui.moveTo(35, 80) # click "File  " in Mercury
        pyautogui.mouseDown(); pyautogui.mouseUp() 

        pyautogui.moveTo(35, 100) # click "Open" in Mercury
        pyautogui.mouseDown(); pyautogui.mouseUp() 
        # pyautogui.mouseDown(); pyautogui.mouseUp() 
        
        pyautogui.write(material_cif) # write cif file name
        pyautogui.press('enter')

        pyautogui.moveTo(1280, 570) # click "Save..." in Mercury
        pyautogui.mouseDown(); pyautogui.mouseUp()

        pyautogui.write(material_cif[:-4]) # write cif file name

        pyautogui.moveTo(1550, 860) # click "Save" to save powder pattern
        pyautogui.mouseDown(); pyautogui.mouseUp()
        # pyautogui.mouseDown(); pyautogui.mouseUp()

        pyautogui.moveTo(100, 100)


# get_cif_files_for_element(material='ti)
generate_powder_pattern(material='ti')

# data = get_dataframe_from_cif_files(material='ti')
# print(data)
# print(data['compound_name'][data['cod_id'] == '1528044'])

# xrd = pd.read_csv('ti_cif_files/ti_1528044.tsv', sep='\t')
# xrd.columns = ['2theta', 'intensity']
# print(xrd)

# plt.plot(xrd)
# plt.show()
