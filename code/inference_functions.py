import os
import json
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd

def load_JSON(path):
    with open(path) as json_file:
        dict = json.load(json_file)
    return dict

def model_fn(model_dir):
    """
    Load the model for inference
    """

#     model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
#     model_alias="biobert"
#     model_path = 'bert_model/Fold_0_ms_pubmed_bert_epoch_13.pt'
#     encoding_path = 'bert_model/utility_files/cresemba_KITS.txt'

    #encoding_path = os.environ["encoding_path"]
    model_name = os.environ["model_name"]
    model_path = os.environ["model_path"]
    
    #label_encoding = load_JSON(os.path.join(model_dir,encoding_path))
    #label_decoding = dict([(value, key) for key, value in label_encoding.items()])

    
    # Load BERT tokenizer from disk.
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    # Load BERT model from disk.
    model = torch.load(os.path.join(model_dir,model_path))

    model_dict = {'model': model, 'tokenizer':tokenizer}
    
    return model_dict

def preprocessing_for_bert(data, model):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # Load the BERT tokenizer
    tokenizer = model['tokenizer']

    # `encode_plus` will:
    #    (1) Tokenize the sentence
    #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
    #    (3) Truncate/Pad sentence to max length
    #    (4) Map tokens to their IDs
    #    (5) Create attention mask
    #    (6) Return a dictionary of outputs
    encoded_data_train = tokenizer.encode_plus(
        data,
        add_special_tokens=True,    # Add `[CLS]` and `[SEP]`
        return_attention_mask=True, # Return attention mask
        padding="max_length",       # Pad sentence to max length
        max_length=256,             # Max length to truncate/pad
        truncation=True,
        return_tensors="pt",        # Return PyTorch tensor
    )

    input_ids = encoded_data_train["input_ids"]
    attention_masks = encoded_data_train["attention_mask"]

    return input_ids, attention_masks

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    bert_model = model['model']
    
    bert_model.to(device)
    bert_model.eval()
    
    test_inputs, test_masks = preprocessing_for_bert(input_data, model)
    
    inputs = {
        "input_ids": test_inputs,
        "attention_mask": test_masks,
    }
    
    with torch.no_grad():
        outputs = bert_model(**inputs)

    logits = outputs[0]

    logits = logits.detach().cpu().numpy()
    
    prediction_labels = np.argmax(logits, axis=1).flatten()
    
    #label_decoding = model['decoding']
    
    # pred_values = pd.Series(prediction_labels).map(label_decoding)
   
    pred_values = {}
    
    pred_values["pred_top_first"] = pd.Series(logits.argsort()[:,-1])[0]
    pred_values["pred_top_second"] = pd.Series(logits.argsort()[:,-2])[0]
    pred_values["pred_top_third"] = pd.Series(logits.argsort()[:,-3])[0]
    
    return pred_values

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    
    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    
    if response_content_type == "application/json":
        response = str(prediction)
    else:
        response = str(prediction)

    return response