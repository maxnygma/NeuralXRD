from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy import signal as sp_sig

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def compute_peak_areas(intensity, clip_threshold=0, peak_distance=None, peak_height=None,
                       rounding_factor=5):
    ''' Compute areas of peaks of a selected pattern '''

    # Get peak indexes
    # x_values = np.arange(0, 2250)
    x_values = np.arange(0, len(intensity))
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


def compute_peak_area_similarity(intensity, data, clip_threshold, peak_distance=None, peak_height=None, rounding_factor=5, 
                                 verbose=False, material_name=None, save_experiments=False, calibration_model=None):
    ''' 
    Find matching elements for a selected XRD samples 

    Args:
        intensity (np.ndarray): intensity data to compute peaks for
        data (pd.DataFrame): data with complete list of materials to find matches
        clip_threshold (float): threshold to clip peaks at the bottom
        peak_distance (float): distance for scipy.find_peaks
        peak_height (float): height for scipy.find_peaks
        rounding_factor (int): floating point to round peak areas to
        verbose (bool): print output or not
        material_name (str): name of the main material
        save_experiments (bool): output cvs data of experiments or not
        calibration_model (sklearn model or None): weights for calibration model 
    '''

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
        else:
            pred = [[0]]

        # Add output if detected peaks > 0 or support model detected material
        if num_detected_peaks > 0 or pred[0] == 1:
            outputs.append({
                    'material': single_element,
                    'areas_material': areas_single_element,
                    'areas_combined_material': areas,
                    'num_matched_peaks': num_detected_peaks,
                    'start_point_material': s_point,
                    'last_point_material': l_point,
                    'start_point_combined_material': s_point_reference,
                    'last_point_combined_material': l_point_reference,
                    'model_pred': pred[0]
                })

        if verbose:
            if num_detected_peaks > 0:
                print(f'Integration algorithm detected {single_element.upper()}')
            if pred[0] == 1:
                print(f'Calibration model detected {single_element.upper()}')

    if save_experiments:
        experiments_data = pd.DataFrame.from_dict(experiments_dict)

        return outputs, experiments_data
    else:
        return outputs


def score_method(data, save_experiments=False, calibration_model=None):
    ''' Evaluate the method on the provided data '''

    tp = 0
    fp = 0

    dataset_size = 0
    data_combined = data[data['id'].str.contains('_')].reset_index(drop=True)

    experiment_data_complete = pd.DataFrame(columns=['reference_material', 'detected_material', 'areas_reference',
                        'areas_detected', 'num_matched_peaks', 'start_point_reference',
                        'last_point_reference', 'start_point_detected', 'last_point_detected'])

    for material in tqdm(data_combined['material'].unique()):
        # Get a random sample of combined material
        dataset_size += len(data_combined['intensity'][data_combined['material'] == material].values)

        for intensity in data_combined['intensity'][data_combined['material'] == material].values:
            intensity = np.array(intensity)

            if save_experiments:
                outputs, experiment_data = compute_peak_area_similarity(intensity=intensity, data=data, clip_threshold=0.1, 
                                                peak_distance=None, peak_height=0.005, rounding_factor=4,
                                                verbose=False, material_name=material, save_experiments=save_experiments, calibration_model=calibration_model)
                
                experiment_data_complete = pd.concat([experiment_data_complete, experiment_data])
            else:
                outputs = compute_peak_area_similarity(intensity=intensity, data=data, clip_threshold=0.1, 
                                                peak_distance=None, peak_height=0.005, rounding_factor=4,
                                                verbose=False, material_name=material, save_experiments=save_experiments, calibration_model=calibration_model)

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
    print('P:', precision, 'R:', recall)
    print('F1:', f_score) 

    if save_experiments:
        print(experiment_data_complete)
        experiment_data_complete.to_csv('experiment_data.csv', index=False)