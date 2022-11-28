import os
import random
import joblib
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy
from scipy import signal as sp_sig
from scipy.spatial import distance
from scipy.spatial.distance import cosine

import matplotlib
matplotlib.use('TkAgg')
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

                    # Shift intensity by a random 2theta (range of [-1; 1])
                    for i in range(15):
                        material_data_1['intensity'] = material_data_1['intensity'].shift(random.randint(-50, 50), fill_value=0.0)
                        material_data_2['intensity'] = material_data_2['intensity'].shift(random.randint(-50, 50), fill_value=0.0)

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

    data.to_csv('data/data.csv', index=False)

    return data


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


def compute_peak_areas(intensity, clip_threshold=0, peak_distance=None, peak_height=None,
                       rounding_factor=5):
    ''' Compute areas of peaks of a selected pattern '''
    # Get peak indexes
    x_values = np.arange(0, 2250)
    #peaks = sp_sig.find_peaks_cwt(intensity, widths=np.arange(5, 90, 30))
    peaks = sp_sig.find_peaks(intensity, distance=peak_distance, height=peak_height)[0]

    # Get minimums between peaks
    ix_peak = np.searchsorted(x_values, peaks)
    ix_min = np.array([np.argmin(i) for i in np.split(intensity, ix_peak)])
    ix_min[1:] += ix_peak
    minimums = x_values[ix_min]

    # Split sequence based on minimums
    x_split = np.split(x_values, ix_min[1:-1])
    y_split = np.split(intensity, ix_min[1:-1])
    y_split = [(x - min(x)) / (max(x) - min(x)) for x in y_split]

    # Clip values on the bottom of peaks to get more accurate results on peaks with little overlap
    for i in range(len(y_split)):
        peak = y_split[i]
        peak[peak < clip_threshold] = 0
        y_split[i] = peak

    # Calculate areas
    areas = [round(np.trapz(ys, xs), rounding_factor) for xs, ys in zip(x_split, y_split)]

    return areas, minimums


def score_method(data):
    tp = 0; fp = 0;

    dataset_size = 0
    data_combined = data[data['id'].str.contains('_')].reset_index(drop=True)

    experiment_data_complete = pd.DataFrame(columns=['reference_material', 'detected_material', 'areas_reference',
                        'areas_detected', 'num_matched_peaks', 'start_point_reference',
                        'last_point_reference', 'start_point_detected', 'last_point_detected'])

    for material in tqdm(data_combined['material'].unique()):
        # Get a random sample of combined material
        dataset_size += len(data_combined['intensity'][data_combined['material'] == material].values)

        for intensity in data_combined['intensity'][data_combined['material'] == material].values:
        #   intensity = random.choice(data_combined['intensity'][data_combined['material'] == material].values.reshape((-1, 2250)))
            intensity = np.array(intensity)
            outputs, experiment_data = compute_peak_area_similarity(intensity=intensity, data=data, clip_threshold=0.25, 
                                               peak_distance=None, peak_height=0.05, rounding_factor=4,
                                               verbose=False, material_name=material, save_experiments=False, calibration_model=None)
            
            experiment_data_complete = pd.concat([experiment_data_complete, experiment_data])

            if len(outputs) > 0:
                for output in outputs:
                    if output['material'] in material:
                        tp += 1
                    elif output['material'] not in material:
                        fp += 1    
            else:
                continue

    precision = tp / (tp + fp)
    recall = tp / (dataset_size * 2)
    f_score = 2 * ((precision * recall) / (precision + recall))
    print(precision, recall)
    print(f_score) 

    print(experiment_data_complete)
    experiment_data_complete.to_csv('experiment_data.csv', index=False)


def compute_peak_area_similarity(intensity, data, clip_threshold, peak_distance=None, peak_height=None, rounding_factor=5, 
                                 verbose=False, material_name=None, save_experiments=False, calibration_model=None):
    ''' Find matching elements for a selected XRD samples '''

    ### Deteck peaks and their minima -> calculate area under peak for each normalized peak -> 
    ### -> do the same for selected sample (also normalize each peak) -> compare number of matching areas
    single_element_data = data[~data['id'].str.contains('_')].reset_index(drop=True)

    areas, minimums = compute_peak_areas(intensity, clip_threshold=clip_threshold, peak_distance=peak_distance, 
                                        peak_height=peak_height, rounding_factor=rounding_factor)

    outputs = []
    if save_experiments:
        experiments_dict = {'reference_material': [], 'detected_material': [], 'areas_reference': [],
                            'areas_detected': [], 'num_matched_peaks': [], 'start_point_reference': [],
                            'last_point_reference': [], 'start_point_detected': [], 'last_point_detected': []}

    # Iterate over a set of samples for a particular element
    for i in range(len(single_element_data)):
        single_element = single_element_data['material'].iloc[i]
        intensity_single_element = np.array(single_element_data['intensity'].iloc[i])

        # Compute peak areas for the searched element
        areas_single_element, _ = compute_peak_areas(intensity_single_element, clip_threshold=clip_threshold, 
                                                  peak_distance=peak_distance, peak_height=peak_height, 
                                                  rounding_factor=rounding_factor)
        
        s_point, l_point = np.where(intensity_single_element > 0)[0][0], np.where(intensity_single_element > 0)[0][-1]
        try:
            s_point_reference, l_point_reference = np.where(intensity[s_point:l_point] > 0)[0][0], np.where(intensity[s_point:l_point] > 0)[0][-1]
        except:
            s_point_reference, l_point_reference = None, None

        # Get number of detected peaks
        num_detected_peaks = len([x for x in range(len(areas_single_element)) if areas_single_element[x] in areas])

        if save_experiments:
            experiments_dict['reference_material'].append(material_name)
            experiments_dict['detected_material'].append(single_element)
            experiments_dict['areas_reference'].append(areas)
            experiments_dict['areas_detected'].append(areas_single_element)
            experiments_dict['num_matched_peaks'].append(num_detected_peaks)
            experiments_dict['start_point_reference'].append(s_point_reference)
            experiments_dict['last_point_reference'].append(l_point_reference)
            experiments_dict['start_point_detected'].append(s_point)
            experiments_dict['last_point_detected'].append(l_point)

        if calibration_model is not None:
            if s_point_reference is None or l_point_reference is None:
                s_point_reference = 0
                l_point_reference = 100

            pred = calibration_model.predict([[num_detected_peaks, s_point_reference, l_point_reference, s_point, l_point, sum(areas), sum(areas_single_element)]])
            prob = calibration_model.predict_proba([[num_detected_peaks, s_point_reference, l_point_reference, s_point, l_point, sum(areas), sum(areas_single_element)]])
        
        print(single_element, pred, prob[0][1])

        # if num_detected_peaks == 1 and len(areas_single_element) > 2:
        #     continue # Check that some random peak of a complex material won't count as prediction
        # elif (l_point - s_point) * 0.85 < len(np.where(intensity[s_point:l_point] > 0)[0]) and num_detected_peaks == 1:
        #     continue # Check that some one-peak material which is not located in a range of detected peak won't count
        if num_detected_peaks > 0:
            if verbose:
                print(single_element, num_detected_peaks)

                print(areas_single_element)
                print(areas)

                print(pred, prob[0][1])
                #print(l_point - s_point, len(np.where(intensity[s_point:l_point] > 0)[0]))
        elif pred[0] == 1:
            print(f'Calibration model detected {single_element.upper()}')

            outputs.append({
                'material': single_element,
                'areas_material': areas_single_element,
                'areas_combined_material': areas,
                'num_matched_peaks': num_detected_peaks,
                'start_point_material': s_point,
                'last_point_material': l_point,
                'start_point_combined_material': s_point_reference,
                'last_point_combined_material': l_point_reference
            })

    if save_experiments:
        experiments_data = pd.DataFrame.from_dict(experiments_dict)

        return outputs, experiments_data
    else:
        return outputs

        
        # print(single_element, num_detected_peaks)



if __name__ == '__main__':
    # data = create_df_from_xrd_files(path_to_xrd_files='xrd_patterns')
    # initial_shape = data.shape[0]
    # data = generate_synthetic_phases(data)

    data = pd.read_csv('data/data.csv')
    data = normalize_intensity(data)
    data = intensities_to_list(data)

    print(data['material'].value_counts())

    # data = data[data['id'].str.contains('_')].reset_index(drop=True)

    # Get XRD patterns of a two materials and a combination of them
    first_material = 'fe'
    second_material = 'v'

    original_sample = np.array(random.choice(data['intensity'][data['material'] == first_material].values))
    original_sample_2 = np.array(random.choice(data['intensity'][data['material'] == second_material].values))
    try:
        combined_intensity = np.array(random.choice(data['intensity'][data['material'] == first_material + '_' + second_material].values))
    except:
        combined_intensity = np.array(random.choice(data['intensity'][data['material'] == second_material + '_' + first_material].values))

    ### Hard samples: mgo + al2o3, overlapping samples like fe + ti

    model = joblib.load('calibration_model.pkl')
        
    outputs = compute_peak_area_similarity(combined_intensity, data, clip_threshold=0.1, 
                                        peak_distance=None, peak_height=0.005, rounding_factor=4, verbose=True,
                                        material_name=None, save_experiments=False, calibration_model=model)
    # score_method(data)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(range(0, 2250), original_sample)
    plt.xlabel(first_material)
    plt.subplot(1, 3, 2)
    plt.plot(range(0, 2250), original_sample_2)
    plt.xlabel(second_material)
    plt.subplot(1, 3, 3)
    plt.plot(range(0, 2250), combined_intensity)
    plt.xlabel(first_material + '_' + second_material)
    plt.show()



