import random
import joblib
import itertools

import numpy as np
import pandas as pd

import scipy
from scipy import signal as sp_sig

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils import *


def detect_peaks(intensity, peak_distance=None, peak_height=None):
    peaks = sp_sig.find_peaks(intensity, distance=peak_distance, height=peak_height)[0]

    print(peaks)


def match_materials(intensity, data, material, peak_distance=None, peak_height=None, model=None):
    single_element_data = data[~data['id'].str.contains('_')].reset_index(drop=True)

    experiments_dict = {'source_material': [], 'reference_material': [], 'peak_1': [], 'peak_2': [], 'peak_3': [],
                        'peak_4': [], 'peak_5': [], 'peak_6': [], 'peak_7': [], 'peak_8': [], 'peak_9': [], 'peak_10': [],
                        'peak_1_reference': [], 'peak_2_reference': [], 'peak_3_reference': [],
                        'peak_4_reference': [], 'peak_5_reference': [], 'peak_6_reference': [], 'peak_7_reference': [], 
                        'peak_8_reference': [], 'peak_9_reference': [], 'peak_10_reference': [],}

    peaks_source = sp_sig.find_peaks(intensity, distance=peak_distance, height=peak_height)[0]
    if len(peaks_source) < 10:
        peaks_source = [*peaks_source, *([0] * (10 - len(peaks_source)))]

    for i in range(len(single_element_data)):
        single_element = single_element_data['material'].iloc[i]
        intensity_single_element = np.array(single_element_data['intensity'].iloc[i])

        peaks_reference = sp_sig.find_peaks(intensity_single_element, distance=peak_distance, height=peak_height)[0]
        if len(peaks_reference) < 10:
            peaks_reference = [*peaks_reference, *([0] * (10 - len(peaks_reference)))]

        pred = model.predict([[*peaks_source, *peaks_reference]])
        if pred[0] == 1:
            print(single_element, pred)

        # print(single_element)
        # print(peaks_source, peaks_reference)

        # closest_pair = sorted(itertools.product(peaks_source, peaks_reference), key=lambda t: abs(t[0]-t[1]))[0]
        # print(abs(closest_pair[0] - closest_pair[1]))

        experiments_dict['source_material'].append(material)
        experiments_dict['reference_material'].append(single_element)
        for x in range(10):
            experiments_dict[f'peak_{x + 1}'].append(peaks_source[x])
            experiments_dict[f'peak_{x + 1}_reference'].append(peaks_reference[x])

        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 2, 1)
        # plt.plot(range(0, 2250), intensity_single_element)
        # plt.xlabel(single_element)
        # plt.subplot(1, 2, 2)
        # plt.plot(range(0, 2250), intensity)
        # plt.xlabel('fe + v')
        # plt.show()

    experiments_data = pd.DataFrame.from_dict(experiments_dict)
    
    return experiments_data


def train_model(data):
    # np.random.seed(99)
    # random.seed(99)

    data['match'] = 0
    for i in range(len(data)):
        if data['reference_material'].iloc[i] in data['source_material'].iloc[i]:
            data['match'].iloc[i] = 1

    train_data, val_data = train_test_split(data, test_size=0.25, random_state=99, shuffle=True, stratify=data['match'])
    train_x, train_y = train_data.drop(['match', 'source_material', 'reference_material'], axis=1), train_data['match']
    val_x, val_y = val_data.drop(['match', 'source_material', 'reference_material'], axis=1), val_data['match']

    # scaler = StandardScaler()
    # train_x[['start_point_reference', 'last_point_reference', 'start_point_detected', 'last_point_detected']] = scaler.fit_transform(train_x[['start_point_reference', 
    #                                                                                                                                           'last_point_reference', 
    #                                                                                                                                           'start_point_detected', 
    #                                                                                                                                           'last_point_detected']])

    model = RandomForestClassifier(n_estimators=50, max_depth=14, criterion='log_loss', random_state=99)
    #model = LogisticRegression()
    model.fit(train_x, train_y)

    joblib.dump(model, 'indexing_model.pkl')
    model = joblib.load('indexing_model.pkl')

    preds = model.predict(val_x)
    print(metrics.precision_score(val_y, preds), metrics.recall_score(val_y, preds))


if __name__ == '__main__':
    data = pd.read_csv('data/data.csv')
    data = normalize_intensity(data)
    data = intensities_to_list(data)
    single_element_data = data[~data['id'].str.contains('_')].reset_index(drop=True)
    # print(data)

    # Get XRD patterns of a two materials and a combination of them
    first_material = 'nao3'
    second_material = 'zn'

    original_sample = np.array(random.choice(data['intensity'][data['material'] == first_material].values))
    original_sample_2 = np.array(random.choice(data['intensity'][data['material'] == second_material].values))
    try:
        combined_intensity = np.array(random.choice(data['intensity'][data['material'] == first_material + '_' + second_material].values))
    except:
        combined_intensity = np.array(random.choice(data['intensity'][data['material'] == second_material + '_' + first_material].values))

    model = joblib.load('indexing_model.pkl')
    _ = match_materials(combined_intensity, data, first_material + '_' + second_material, peak_distance=None, peak_height=0.05, model=model)

    # Collect data
    # complete_training_data = pd.DataFrame(columns=['source_material', 'reference_material', 'peak_1', 'peak_2', 'peak_3',
    #                     'peak_4', 'peak_5', 'peak_6', 'peak_7', 'peak_8', 'peak_9', 'peak_10',
    #                     'peak_1_reference', 'peak_2_reference', 'peak_3_reference',
    #                     'peak_4_reference', 'peak_5_reference', 'peak_6_reference', 'peak_7_reference', 
    #                     'peak_8_reference', 'peak_9_reference', 'peak_10_reference'])

    # for first_material in single_element_data['material'].unique():
    #     for second_material in single_element_data['material'].unique():
    #         if first_material == second_material:
    #             continue

    #         try:
    #             combined_intensity = np.array(random.choice(data['intensity'][data['material'] == first_material + '_' + second_material].values))
    #         except:
    #             combined_intensity = np.array(random.choice(data['intensity'][data['material'] == second_material + '_' + first_material].values))

    #         experiment_data = match_materials(combined_intensity, data, first_material + '_' + second_material, peak_distance=None, peak_height=0.05, model=model)
    #         complete_training_data = pd.concat([complete_training_data, experiment_data])

    # print(complete_training_data)

    # train_model(complete_training_data)

    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.plot(range(0, 2250), original_sample)
    # plt.xlabel(first_material)
    # plt.subplot(1, 3, 2)
    # plt.plot(range(0, 2250), original_sample_2)
    # plt.xlabel(second_material)
    # plt.subplot(1, 3, 3)
    # plt.plot(range(0, 2250), combined_intensity)
    # plt.xlabel(first_material + '_' + second_material)
    # plt.show()
