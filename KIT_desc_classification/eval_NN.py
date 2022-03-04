from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def eval_avg_score(best_predictions_list,eval_path):
    acc_list = []
    mic_f1_list = []
    mac_f1_list = []
    weig_f1_list = []
    
    for predicted_labels,actual_labels in best_predictions_list:
        acc = accuracy_score(actual_labels, predicted_labels)
        acc_list.append(acc)
        mic_f1 = f1_score(actual_labels, predicted_labels, average='micro')
        mic_f1_list.append(mic_f1)
        mac_f1 = f1_score(actual_labels, predicted_labels, average='macro')
        mac_f1_list.append(mac_f1)
        weig_f1 = f1_score(actual_labels, predicted_labels, average='weighted')
        weig_f1_list.append(weig_f1)

    metrics = {
            'acc': sum(acc_list)/len(acc_list),
            'micro_f1': sum(mic_f1_list)/len(mic_f1_list),
            'macro_f1': sum(mac_f1_list)/len(mac_f1_list),
            'weig_f1': sum(weig_f1_list)/len(weig_f1_list)
    }
    data_items=metrics.items()
    data_list=list(data_items)
    df=pd.DataFrame(data_list)
    gq="/model-eval.csv"
    df.to_csv(eval_path+gq)

    
    return metrics