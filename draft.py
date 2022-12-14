import os
import random
import joblib
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy import signal as sp_sig
from scipy.spatial import distance
from scipy.spatial.distance import cosine

#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from data_generation import *
from utils import *
from integration_algorithm import *




def windowed_similarity(data, intensity, window_size):
    assert intensity.shape[-1] % window_size == 0

    single_element_data = data[~data['id'].str.contains('_')].reset_index(drop=True)
    similarities = {x: 0 for x in single_element_data['material'].unique()}
    #similarities.keys = single_element_data['material'].unique()

    # Iterate over a set of samples for a particular element
    for i in range(0, len(single_element_data) - 2250, 2250):
        intensity_single_element = single_element_data['intensity'].iloc[i:i + 2250].values

        similarities_element = []
        for window_step in range(2250 // window_size - 1):
            intensity_windowed = intensity[window_size * window_step:window_size * (window_step + 1)]
            intensity_single_element_windowed = intensity_single_element[window_size * window_step:window_size * (window_step + 1)]
            if np.max(intensity_single_element_windowed) < 0.1 or np.max(intensity_windowed) < 0.1:
                continue

            # Normalize slices
            intensity_windowed = (intensity_windowed - np.min(intensity_windowed)) / \
                                 (np.max(intensity_windowed) - np.min(intensity_windowed))
            intensity_single_element_windowed = (intensity_single_element_windowed - np.min(intensity_single_element_windowed)) / \
                                                (np.max(intensity_single_element_windowed) - np.min(intensity_single_element_windowed))

            # print(np.mean(intensity_single_element_windowed))
            # print(np.mean(intensity_windowed))

            similarity_step = 1 - cosine(intensity_single_element_windowed, intensity_windowed)
            #similarity_step = distance.euclidean(intensity_single_element_windowed, intensity_windowed)
            similarities_element.append(similarity_step)

            if single_element_data['material'].iloc[i] == 'cr' or single_element_data['material'].iloc[i] == 'ti':
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(range(0, window_size), intensity_single_element_windowed)
                plt.xlabel(single_element_data['material'].iloc[i])
                plt.subplot(1, 2, 2)
                plt.plot(range(0, window_size), intensity_windowed)
                plt.xlabel(similarity_step)
                plt.show()

            print(single_element_data['material'].iloc[i], similarity_step)

        if len(similarities_element) == 0:
            similarity = 0
        else:
            similarity = max(similarities_element)
        print(single_element_data['material'].iloc[i], similarity)
    
        if similarities[single_element_data['material'].iloc[i]] < similarity:
            similarities[single_element_data['material'].iloc[i]] = similarity

    # Print top-3 similarities
    print('Top-3 materials')
    top_3_sim = sorted(similarities.values())[::-1][:3]
    for x in top_3_sim:
        print(list(similarities.keys())[list(similarities.values()).index(x)], x)


def als_baseline_correction(y, lam, p, n_iters=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)

    for i in range(n_iters):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    
    return z
    
# def als_baseline_correction(y, lam, p, n_iters=10):
#     l = len(y)
#     d = sparse.csc_matrix(np.diff(np.eye(l), 2))
#     w = np.ones(l)

#     for i in range(n_iters):
#         W = sparse.spdiags(w, 0, l, l)
#         Z = W + lam * d.dot(d.transpose())
#         z = sparse.linalg.spsolve(Z, w * y)
#         w = p * (y > z) + (1 - p) * (y < z)
    
#     return z


if __name__ == '__main__':
    # data = create_df_from_xrd_files(path_to_xrd_files='xrd_patterns')
    # initial_shape = data.shape[0]
    # data = generate_synthetic_phases(data)

    # with open('real_samples/ti21v40cr35.txt', 'r') as f:
    #     real_data = f.read()
    #     real_data = ' '.join(real_data.split())
    # real_data = [float(x) for x in real_data.split(' ')]  

    # degrees = np.array([x for i, x in enumerate(real_data) if i % 2 == 0])

    # intensity = np.array([x for i, x in enumerate(real_data) if i % 2 != 0])
    # # intensity = sp_sig.savgol_filter(intensity, 50, 3)
    # # baseline = als_baseline_correction(intensity, lam=10**2, p=0.001, n_iters=10)
    # # intensity = intensity - baseline
    # # intensity = (intensity - min(intensity)) / (max(intensity) - min(intensity))

    # # plt.plot(degrees, intensity)
    # # plt.show()

    # data = pd.read_csv('data/data.csv')
    # data = normalize_intensity(data)
    # data = intensities_to_list(data)

    # original_sample = np.array(random.choice(data['intensity'][data['material'] == 'ti2o'].values))
    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(0, len(intensity)), intensity)
    # plt.xlabel('tivcr')
    # plt.subplot(1, 2, 2)
    # plt.plot(range(0, 2250), original_sample)
    # plt.xlabel('v')
    # plt.show()

    # model = joblib.load('calibration_model.pkl')

    # outputs = compute_peak_area_similarity(intensity, data, clip_threshold=0.1, 
    #                                     peak_distance=None, peak_height=0.03, rounding_factor=3, verbose=True,
    #                                     material_name=None, save_experiments=False, calibration_model=model)

    # exit()

    data = pd.read_csv('data/data.csv')
    data = normalize_intensity(data)
    data = intensities_to_list(data)

    # data = data[data['id'].str.contains('_')].reset_index(drop=True)

    # Get XRD patterns of a two materials and a combination of them
    first_material = 'fe'
    second_material = 'al2o3'

    original_sample = np.array(random.choice(data['intensity'][data['material'] == first_material].values))
    original_sample_2 = np.array(random.choice(data['intensity'][data['material'] == second_material].values))
    try:
        combined_intensity = np.array(random.choice(data['intensity'][data['material'] == first_material + '_' + second_material].values))
    except:
        combined_intensity = np.array(random.choice(data['intensity'][data['material'] == second_material + '_' + first_material].values))

    model = joblib.load('calibration_model.pkl')
        
    outputs = compute_peak_area_similarity(combined_intensity, data, clip_threshold=0.1, 
                                        peak_distance=None, peak_height=0.005, rounding_factor=4, verbose=True,
                                        material_name=None, save_experiments=False, calibration_model=model)
    #score_method(data, save_experiments=False, calibration_model=model)
    # F1 0.823 - without support model, F1 - 0.971 - with support model

    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.plot(range(0, 2250), original_sample)
    # # plt.xlabel(first_material)
    # plt.subplot(1, 3, 2)
    # plt.plot(range(0, 2250), original_sample_2)
    # # plt.xlabel(second_material)
    # plt.subplot(1, 3, 3)
    # plt.plot(range(0, 2250), combined_intensity)
    # # plt.xlabel(first_material + '_' + second_material)
    # plt.show()



