#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
#from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

df = pd.read_csv('./data.txt', sep='\t',header=0,index_col="sample",low_memory=False)
target = df.pop("group")

train_data, test_data, train_target, test_target = train_test_split(df, target, test_size=0.2, random_state=2025)

rf = RandomForestClassifier(n_estimators=500, random_state=2025, max_depth=10, max_features="sqrt", min_samples_split=2, class_weight=None, bootstrap=False, ccp_alpha=0.0, max_leaf_nodes=50, min_impurity_decrease=0.0, min_samples_leaf=1)
rf.fit(train_data,train_target)

# get the feature importance scores
importances = rf.feature_importances_

# sort the important features
feature_importance = pd.DataFrame({'Feature': df.columns,'Importance': importances})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
feature_importance.to_csv('./feature_importance.csv',header=True,index=True)

print("Feature Importance:")
print(feature_importance)

top_n = 40
top_features = feature_importance['Feature'][:top_n]
