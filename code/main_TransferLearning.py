from preprocess import *
import os
import ast
import warnings
from utils import *
from collections import Counter
warnings.filterwarnings("ignore", category=DeprecationWarning)
from transfer_learning import *
from eval_NN import eval_avg_score
import glob
import argparse

current_path = os.path.dirname(__file__)
project_path = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))

def main_pipeline(dataset, n_folds, eval_test, label,input_path, 
                  model_name, model_alias, epochs,
                  batch_size, seed_val,model_path,cresemba_path,verbose):
    kit_label_encoding_cresemba = read_object(cresemba_path, file_type="txt")
    kit_label_decoding_cresemba = dict([(value, key) for key, value in kit_label_encoding_cresemba.items()])
    if label == 'KIT description':
        if dataset == 'cresemba':
            label_encoding = kit_label_encoding_cresemba
            label_decoding = kit_label_decoding_cresemba
            dataset_df = preprocess_cresemba_raw_for_kit(input_path)
        elif dataset == 'zavicefta':
            label_encoding = kit_label_encoding_zavicefta
            label_decoding = kit_label_decoding_zavicefta
            dataset_df = preprocess_zavicefta_raw_for_kit()
    elif label == 'Insights Category':
        if dataset == 'cresemba':
            label_encoding = insights_label_encoding_cresemba
            label_decoding = insights_label_decoding_cresemba
            dataset_df = preprocess_cresemba_raw_for_insights(label_encoding)
        elif dataset == 'zavicefta':
            label_encoding = insights_label_encoding_zavicefta
            label_decoding = insights_label_decoding_zavicefta
            dataset_df = preprocess_zavicefta_raw_for_insights(label_encoding)

    labels_set = list(label_encoding.keys())
    
    
    
    biobert_models_path = model_path

    cv_data, _ = cross_val(df=dataset_df, n_folds=int(n_folds), label=label)

    print('Number of data points in each class:')
    labels = cv_data[0][0][label]
    print(Counter(labels))

    cv_data, label_set = pick_data_for_exp_scenario(folds=cv_data, eval_test=eval_test, dataset=dataset, label=label,
                                                    label_mapping=label_encoding, labels_set=labels_set)

    

    best_predictions_list = cross_val_train(cv_data, model_name,
                                            epochs, batch_size, seed_val, 
                                            verbose, biobert_models_path, model_alias)

    #metrics = eval_avg_score(best_predictions_list, n_folds, project_path, label, dataset)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--n_folds', type=int, default=2)
    parser.add_argument('--dataset', type=str, default="cresemba")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--label', type=str, default="KIT description")
    parser.add_argument('--model_alias', type=str, default="ak_save")
    parser.add_argument('--model_name', type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--seed_val', type=int, default=17)
    parser.add_argument('--eval_test', type=bool, default=True)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--cresemba_path', type=str)
    
    # SageMaker environment variables.
    parser.add_argument('--hosts', type=list, default=os.environ['SM_HOSTS'])
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR']) # output_path arg from train_model.py.
    parser.add_argument('--num_cpus', type=int, default=os.environ['SM_NUM_CPUS'])
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    # Parse command-line args and run main.
    args, _ = parser.parse_known_args()

    main_pipeline(dataset = args.dataset,
                    n_folds = args.n_folds,
                    input_path=args.input_path,
                    eval_test = args.eval_test,
                    label = args.label,
                    
                    # DEFINE PARAMETERS FOR FINETUNING:
                    model_name = args.model_name,
                    model_alias = args.model_alias, #for model saving file name
                    epochs = args.epochs,
                    batch_size = args.batch_size,
                    seed_val = args.seed_val,
                    model_path=args.model_dir,
                    cresemba_path=args.cresemba_path,verbose = args.verbose)
    torch.cuda.empty_cache()