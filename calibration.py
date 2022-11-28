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
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('experiment_data.csv')
data['match'] = 0
data['area_reference_total'] = 0
data['area_detected_total'] = 0
# data['len_reference'] = 0
# data['is_simple_element'] = 1
for i, row in data.iterrows():
    if row['detected_material'] in row['reference_material']:
        data['match'].iloc[i] = 1

    data['area_reference_total'].iloc[i] = sum(eval(row['areas_reference']))
    data['area_detected_total'].iloc[i] = sum(eval(row['areas_detected']))

    # if len(row['detected_material']) > 2:
    #     data['is_simple_element'].iloc[i] = 0
        
    # data['len_reference'].iloc[0] = (row['last_point_reference'] - row['start_point_reference']) - (row['last_point_detected'] - row['start_point_detected'])

data = data.dropna(axis=0)
# data['area_reference_total'] = data['areas_reference'].apply(lambda x: sum(eval(x)))
# data['area_detected_total'] = data['areas_detected'].apply(lambda x: sum(eval(x)))

print(data)
print('False samples', len(data[data['match'] == 0][data['num_matched_peaks'] > 0]))
print(len(data[data['match'] == 1][data['num_matched_peaks'] == 0]))

train_data, val_data = train_test_split(data, test_size=0.25, random_state=99, shuffle=True, stratify=data['match'])
train_x, train_y = train_data.drop(['match', 'reference_material', 'detected_material', 'areas_reference', 'areas_detected'], axis=1), train_data['match']
val_x, val_y = val_data.drop(['match', 'reference_material', 'detected_material', 'areas_reference', 'areas_detected'], axis=1), val_data['match']

# scaler = StandardScaler()
# train_x[['start_point_reference', 'last_point_reference', 'start_point_detected', 'last_point_detected']] = scaler.fit_transform(train_x[['start_point_reference', 
#                                                                                                                                           'last_point_reference', 
#                                                                                                                                           'start_point_detected', 
#                                                                                                                                           'last_point_detected']])

model = RandomForestClassifier(n_estimators=50, max_depth=14, criterion='log_loss', random_state=99)
model.fit(train_x, train_y)

joblib.dump(model, 'calibration_model.pkl')
model = joblib.load('calibration_model.pkl')

preds = model.predict(val_x)
score = accuracy_score(val_y, preds)
print(score)


