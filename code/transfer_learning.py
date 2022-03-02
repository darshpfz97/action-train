"""
PLEASE NOTE FOR TESTING MODELS:
1. DEFINE PARAMETERS FOR FINETUNING IN THE VARIABLES SUGGESTED
2. FOR THE LAST STEP COMMENT OUT THE REST OF THE CODE AND RUN A DIFFERENT EPOCH BASED ON THE GRAPHS OUTPUT; THE CODE WRITTEN OUTPUTS THE LAST EPOCH BUT SOMETIMES DEPENDING 
ON THE NUMBER OF EPOCHS, A MIDDLE EPOCH PERFORMS BETTER
"""

import torch
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

def evaluate_model(dataloader_val, model):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to('cuda') for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

def accuracy_score_calc(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

def get_saved_model_path(epoch,fold_no,biobert_models_path,model_alias):
    model_path = biobert_models_path+f"/Fold_{fold_no}_{model_alias}_epoch_{epoch}.pt"
    return model_path

def format_data(data, labels, model_name, batch_size):
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    encoded_data_train = tokenizer.batch_encode_plus(
        data.values,
        add_special_tokens=True,
        return_attention_mask=True,
        padding='max_length',
        max_length=256,
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoded_data_train['input_ids']
    attention_masks = encoded_data_train['attention_mask']
    labels_tensor = torch.tensor(labels)

    dataset_tensor = TensorDataset(input_ids, attention_masks, labels_tensor)
    return dataset_tensor

def train(dataset_train, dataset_val, model_name, num_labels, batch_size, seed_val, fold_no, epochs,
           verbose, biobert_models_path, model_alias):
    model = BertForSequenceClassification.from_pretrained(model_name,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False)

    dataloader_train = DataLoader(dataset_train,
                                    sampler=RandomSampler(dataset_train),
                                    batch_size=batch_size)

    dataloader_validation = DataLoader(dataset_val,
                                        sampler=SequentialSampler(dataset_val),
                                        batch_size=batch_size)

    model.cuda()
    optimizer = AdamW(model.parameters(),
                        lr=1e-5,
                        eps=1e-8)


    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(dataloader_train) * epochs)

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    

    best_acc = 0.
    best_epoch = 1
    best_epoch_file = ''
    
    for epoch in tqdm(range(1, epochs + 1), desc=f"FOLD {fold_no}:"):

        model.train()

        loss_train_total = 0

        progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in progress_bar:
            model.zero_grad()
            device = torch.device('cuda')
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[2],
                        }

            outputs = model(**inputs)

            loss = outputs[0]
            loss_train_total += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

#         torch.save(model.state_dict(), f'finetuned_{model_alias}_epoch_{epoch}.model')

        tqdm.write(f'\nEpoch {epoch}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')

        val_loss, predictions, true_vals = evaluate_model(dataloader_validation, model)
        val_f1 = f1_score_func(predictions, true_vals)
        val_acc = accuracy_score_calc(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'Validation accuracy: {val_acc}')
        tqdm.write(f'F1 Score (Weighted): {val_f1}')
        
        ##########
        best_acc = 0. if epoch == 1 else best_acc
        best_epoch = 1 if epoch == 1 else best_epoch
        best_epoch_file = '' if epoch == 1 else best_epoch_file
        ##########
        if val_acc > best_acc:
            prev_best_epoch_file = get_saved_model_path(best_epoch,fold_no,biobert_models_path,model_alias)
            if os.path.exists(prev_best_epoch_file):
                os.remove(prev_best_epoch_file)
                
            best_acc = val_acc
            best_epoch = epoch
            best_epoch_file = get_saved_model_path(best_epoch,fold_no,biobert_models_path,model_alias)
            _, best_predictions, best_true_vals = evaluate_model(dataloader_validation, model)
            best_predictions = np.argmax(best_predictions, axis=1)
            print(f'\nEpoch: {best_epoch} - New best accuracy! Accuracy: {best_acc}\n\n\n')
        
            # torch.save(model.state_dict(), biobert_models_path+f"/Fold_{fold_no}_{model_alias}_epoch_{epoch}.model")
            torch.save(model, biobert_models_path+f"/Fold_{fold_no}_{model_alias}_epoch_{epoch}.pt")

        
    
    return model, best_acc, best_predictions, best_true_vals

def cross_val_train(folds, model_name, epochs, batch_size, seed_val,  
                    verbose, biobert_models_path, model_alias):
    fold_no = 0
    best_acc_list = []
    best_predictions_list = []
    for fold in folds:
        print("FOLD: ",fold_no)

        train_dfo = fold[0][0]
        train_labels = fold[0][1].to_numpy()

        test_dfo = fold[1][0]
        test_labels = fold[1][1].to_numpy()

        train_set = format_data(train_dfo,train_labels, model_name, batch_size)
        test_set = format_data(test_dfo, test_labels, model_name, batch_size)

        num_labels = len(fold[0][1].unique())
        _, best_acc, best_predictions, best_true_vals = train(train_set, test_set, 
                                                              model_name, num_labels, 
                                                              batch_size, seed_val, 
                                                              fold_no, epochs, 
                                                              verbose, biobert_models_path,
                                                              model_alias)
        best_acc_list.append(best_acc)
        best_predictions_list.append((best_predictions, best_true_vals))
        fold_no = fold_no + 1

    # avg_best_acc = sum(best_acc_list)/len(best_acc_list)
    # print(avg_best_acc)
    # label = "KIT description"
    # dataset = "cresemba"
    # metrics = evaluate(best_predictions_list, label=label, dataset=dataset)

    #     dataloader_test = DataLoader(test_set, sampler=SequentialSampler(test_set), batch_size=batch_size)
    #     loss_val_avg, predictions, true_vals = evaluate(dataloader_test,trained_model)
    #     val_acc = accuracy_score_calc(predictions, true_vals)

    return best_predictions_list
