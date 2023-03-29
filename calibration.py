import joblib

import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')


def prepare_training_data(data):
    data['match'] = 0
    data['area_reference_total'] = 0
    data['area_detected_total'] = 0
    for i, row in data.iterrows():
        if row['detected_material'] in row['reference_material']:
            data['match'].iloc[i] = 1

        data['area_reference_total'].iloc[i] = sum(eval(row['areas_reference']))
        data['area_detected_total'].iloc[i] = sum(eval(row['areas_detected']))

    data = data.dropna(axis=0)

    return data


def split_data(data):
    train_data, val_data = train_test_split(data, test_size=0.25, random_state=99, shuffle=True, stratify=data['match'])
    train_x, train_y = train_data.drop(['match', 'reference_material', 'detected_material', 'areas_reference', 'areas_detected'], axis=1), train_data['match']
    val_x, val_y = val_data.drop(['match', 'reference_material', 'detected_material', 'areas_reference', 'areas_detected'], axis=1), val_data['match']

    return train_x, train_y, val_x, val_y


if __name__ == '__main__':
    data = pd.read_csv('experiment_data.csv')
    
    data = prepare_training_data(data)
    train_x, train_y, val_x, val_y = split_data(data)

    model = RandomForestClassifier(n_estimators=50, max_depth=14, criterion='log_loss', random_state=99)
    model.fit(train_x, train_y)

    joblib.dump(model, 'calibration_model.pkl')
    model = joblib.load('calibration_model.pkl')

    preds = model.predict(val_x)
    score = f1_score(val_y, preds)
    
    print(score)


