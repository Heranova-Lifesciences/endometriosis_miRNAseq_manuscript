#!/usr/bin/env python3

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import joblib

# get the data path
data_file_path = "./data.txt"

# read in data
df = pd.read_csv(data_file_path, sep='\t', index_col="sample")
target = df.pop("group")
#data = df.values

combined_results = []
num_split = 30
num_training = 100
split_summary = []

# perform repeated subsampling cross validation
for i in range(num_split):
    print(f"Split Iteration {i+1}")

    train_data, test_data, train_target, test_target = train_test_split(df, target, test_size=0.2, random_state=((i+42)*2))

    train_filename = f"train_{i+1}.txt"
    test_filename = f"test_{i+1}.txt"
    train_df = pd.concat([train_data, train_target], axis=1)
    test_df = pd.concat([test_data, test_target], axis=1)
    train_df.to_csv(train_filename, sep="\t")
    test_df.to_csv(test_filename, sep="\t")

    all_pred_probas = np.zeros((num_training, len(test_target)))
    iteration_summary = []

    # loop through 30 iterations
    for k in range(num_training):
        rf = RandomForestClassifier(n_estimators=500, random_state=(k+1), max_depth=10, max_features="sqrt", min_samples_split=2, class_weight=None, bootstrap=False, ccp_alpha=0.0, max_leaf_nodes=50, min_impurity_decrease=0.0, min_samples_leaf=1)
        rf.fit(train_data, train_target)

        pred_probas = rf.predict_proba(test_data)[:, 1]
        all_pred_probas[k, :] = pred_probas
    
        binary_predictions = (pred_probas > 0.5).astype(int)
    
        # save the performance metrics in confusion matrix and process the matrix for auc, sensitivity and specificity
        cm = confusion_matrix(test_target, binary_predictions, labels=[0, 1])
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        auc_score = roc_auc_score(test_target, pred_probas)
    
        # store all iteration results
        combined_results.append({'Split':i+1, 'model':k+1, 'auc': auc_score, 'sensitivity': sensitivity, 'specificity': specificity})

        # store this iteration results
        iteration_summary.append({'AUC': auc_score, 'Sensitivity': sensitivity, 'Specificity': specificity})

    # get the summary for 30 iterations
    summary_df = pd.DataFrame(iteration_summary)
    mean_auc = summary_df['AUC'].mean()
    std_auc = summary_df['AUC'].std()
    mean_sensitivity = summary_df['Sensitivity'].mean()
    std_sensitivity = summary_df['Sensitivity'].std()
    mean_specificity = summary_df['Specificity'].mean()
    std_specificity = summary_df['Specificity'].std()

    test_pred_filename = f"test_pred{i+1}.txt"
    pred_df = pd.DataFrame(binary_predictions, columns=["Prediction"])
    test_pred_df = test_df
    test_pred_df["Prediction"] = pred_df["Prediction"].values
    test_pred_df.to_csv(test_pred_filename, sep="\t")

    rf_name = f"rf_model_{i+1}.pkl"
    joblib.dump(rf, rf_name)

    print(f"Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")
    print(f"Mean Sensitivity: {mean_sensitivity:.4f} (+/- {std_sensitivity:.4f})")
    print(f"Mean Specificity: {mean_specificity:.4f} (+/- {std_specificity:.4f})")

    # compute roc auc curve
    mean_pred_probas = all_pred_probas.mean(axis=0)
    fpr, tpr, thresholds = roc_curve(test_target, mean_pred_probas)
    roc_auc = auc(fpr, tpr)

    roc_filename = f'roc_curve_{i+1}.png'

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(roc_filename)
    plt.clf()

    mean_sensitivity_list = []
    mean_specificity_list = []
    sensitivity_specificity_at_cutoffs = []

    cutoffs = np.arange(0.1, 1.0, 0.1)
    for cutoff in cutoffs:
        sensitivity_list = []
        specificity_list = []
        for x in range(num_training):
            pred_probas = all_pred_probas[x, :]
            binary_predictions = (pred_probas > cutoff).astype(int)
            cm = confusion_matrix(test_target, binary_predictions, labels=[0, 1])
            TN, FP, FN, TP = cm.ravel()
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            sensitivity_list.append(sensitivity)
            specificity_list.append(specificity)
        mean_sensitivity = np.mean(sensitivity_list)
        mean_specificity = np.mean(specificity_list)
        mean_sensitivity_list.append(mean_sensitivity)
        mean_specificity_list.append(mean_specificity)
        sensitivity_specificity_at_cutoffs.append({
            'cutoff': cutoff,
            'mean_sensitivity': mean_sensitivity,
            'mean_specificity': mean_specificity,
        })

    # plot performance metrics
    performance_filename = f'performance_{i+1}.png'

    plt.figure()
    plt.plot(cutoffs, mean_sensitivity_list, '-o', label='Sensitivity')
    plt.plot(cutoffs, mean_specificity_list, '-o', label='Specificity')
    plt.xlabel('Cutoff')
    plt.ylabel('Performance')
    plt.title('Sensitivity and Specificity')
    plt.legend()
    plt.savefig(performance_filename)
    plt.clf()

    # use 0.5 as threshold for binary classification
    mean_sensitivity_at_cutoff_0_5 = next(item['mean_sensitivity'] for item in sensitivity_specificity_at_cutoffs if item['cutoff'] == 0.5)
    mean_specificity_at_cutoff_0_5 = next(item['mean_specificity'] for item in sensitivity_specificity_at_cutoffs if item['cutoff'] == 0.5)

    split_summary.append({
        'Split': i+1,
        'Mean AUC': mean_auc,
        'Std AUC': std_auc,
        'Mean Sensitivity': mean_sensitivity,
        'Std Sensitivity': std_sensitivity,
        'Mean Specificity': mean_specificity,
        'Std Specificity': std_specificity,
        'Mean Sensitivity at 0.5': mean_sensitivity_at_cutoff_0_5,
        'Mean Specificity at 0.5': mean_specificity_at_cutoff_0_5
    })

# save results in csv format
combined_results_df = pd.DataFrame(combined_results)
combined_results_df.to_csv('split_performance_metrics.csv', index=False)

split_summary_df = pd.DataFrame(split_summary)
split_summary_df.to_csv('split_average_performance.csv', index=False)
