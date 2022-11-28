import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def intensities_to_list(data):
    data['intensity_list'] = 0
    data['intensity_list'] = data['intensity_list'].astype('object')

    for i in range(0, len(data) - 2250, 2250):
        intensity = data['intensity'].iloc[i:i + 2250].values.tolist()
        data['intensity_list'].iloc[i] = intensity

    data = data[data['intensity_list'] != 0].reset_index(drop=True)
    data = data.drop(['intensity', '2theta'], axis=1)
    data.rename(columns={'intensity_list': 'intensity'}, inplace=True)
    
    return data


def normalize_intensity(data):
    for x in range(0, len(data) - 2250, 2250):
        intensity = data['intensity'].iloc[x:x + 2250].values
        intensity = (intensity - min(intensity)) / (max(intensity) - min(intensity))

        data['intensity'].iloc[x:x + 2250] = intensity

    return data