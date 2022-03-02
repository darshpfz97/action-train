from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def eval_avg_score(best_predictions_list, n_folds, project_path, label, dataset):
    acc_list = []
    mic_f1_list = []
    mac_f1_list = []
    weig_f1_list = []
    cm_list = []
    cm_norm_list = []
    for predicted_labels,actual_labels in best_predictions_list:
        acc = accuracy_score(actual_labels, predicted_labels)
        acc_list.append(acc)
        mic_f1 = f1_score(actual_labels, predicted_labels, average='micro')
        mic_f1_list.append(mic_f1)
        mac_f1 = f1_score(actual_labels, predicted_labels, average='macro')
        mac_f1_list.append(mac_f1)
        weig_f1 = f1_score(actual_labels, predicted_labels, average='weighted')
        weig_f1_list.append(weig_f1)

        cm = confusion_matrix(actual_labels, predicted_labels)
        cm_list.append(cm)
        cm_norm = confusion_matrix(actual_labels, predicted_labels, normalize='true')
        cm_norm_list.append(cm_norm)

    cm_sum = sum(cm_list)
    cm_sum_norm = sum(cm_norm_list)

    output_cf_file_name = "average_confusion_matrix.png"
    # output_cf_file_name = "average_confusion_matrix_insights.png"

    output_cf_path = '{}/results/{}/{}/'.format(project_path, label, dataset)
    print(output_cf_path)
    output_cf_norm_file_name = "average_confusion_matrix_normalized.png"
    # output_cf_norm_file_name = "average_confusion_matrix_normalized_insights.png"

    if not os.path.exists(output_cf_path):
        os.makedirs(output_cf_path)

    plt.figure(figsize=(8, 8))
    sns.heatmap(np.around(cm_sum / n_folds, decimals=1), annot=True, xticklabels="auto", yticklabels="auto")
    # plt.figure(figsize=(8,8))
    # sns.heatmap(cm_sum, annot=True, xticklabels=labels_set, yticklabels=labels_set)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.savefig(output_cf_path+output_cf_file_name)

    plt.figure(figsize=(8, 8))
    sns.heatmap(np.around(cm_sum_norm / n_folds, decimals=1), annot=True, xticklabels="auto", yticklabels="auto")
    # plt.figure(figsize=(5,5))
    # sns.heatmap(cm_sum_norm, annot=True, xticklabels=labels_set, yticklabels=labels_set)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.savefig(output_cf_path+output_cf_norm_file_name)

    metrics = {
            'acc': sum(acc_list)/len(acc_list),
            'micro_f1': sum(mic_f1_list)/len(mic_f1_list),
            'macro_f1': sum(mac_f1_list)/len(mac_f1_list),
            'weig_f1': sum(weig_f1_list)/len(weig_f1_list)
    }

    print("Accuracies across folds: ", acc_list)
    print("\n\nAverage scores across all folds:-\n\n",
            metrics, "\n\n", cm_sum/n_folds, "\n\n", cm_sum_norm/n_folds)
    return metrics