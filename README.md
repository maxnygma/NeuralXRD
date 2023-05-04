# NeuralXRD
This is the official code for [ICLR'23 ML4Materials Workshop](https://www.ml4materials.com/) paper "Machine Learning-Assisted Close-Set X-ray 
Diffraction Phase Identification of Transition Metals".

![method_overview](https://user-images.githubusercontent.com/51490783/230770271-715af7bc-8bf4-4d9c-ad7b-a536940eb236.png)

## Overview
* **xrd_patterns**: a folder containing a batch of XRD diffraction patterns in tsv format. All of the phases come from [Crystallography Open Database](http://www.crystallography.net/cod/)
and were extracted using open Mercury software.
* **data_generation.py**: functions to create a dataset of synthetic phases.
* **integration_algorithm**: main functionality of the proposed method. File includes functions to compute phase stats and match elements.
* **calibration.py**: training of calibration models. They help in making corrections to the integration method results.
* **utils.py**: utility functions.
* **demo.py**: an example of the running code.

## How to run the code
In order to run an experiment, you need to create your DataFrame using functions from data_generation.py and then apply integration algorithm. 
Simple example (demo.py):
```python
from data_generation import create_df_from_xrd_files, generate_synthetic_phases
from integration_algorithm import score_method
from utils import normalize_intensity, intensities_to_list


if __name__ == '__main__':
    # Acquire the data
    data = create_df_from_xrd_files(path_to_xrd_files='xrd_patterns')
    data = generate_synthetic_phases(data)

    # Preprocess 
    data = normalize_intensity(data)
    data = intensities_to_list(data)

    # Evaluate
    score_method(data, save_experiments=False, calibration_model=None)
```

# Reference
To cite this paper use the following reference:
```
@inproceedings{zhdanov2023machine,
  title={Machine learning-assisted close-set X-ray diffraction phase identification of transition metals},
  author={Zhdanov, Maksim and Zhdanov, Andrey},
  booktitle={Workshop on''Machine Learning for Materials''ICLR 2023}
}
```
