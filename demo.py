from data_generation import create_df_from_xrd_files, generate_synthetic_phases
from integration_algorithm import score_method
from utils import normalize_intensity, intensities_to_list


if __name__ == '__main__':
    # Acquire the data
    data = create_df_from_xrd_files('xrd_patterns')
    data = generate_synthetic_phases(data)

    # Preprocess 
    data = normalize_intensity(data)
    data = intensities_to_list(data)

    # Evaluate
    score_method(data, save_experiments=False, calibration_model=None)