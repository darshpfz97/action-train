import pandas as pd
from sklearn import model_selection
import os
import json
from utils import *



def preprocess_cresemba_raw_for_kit(input_path):
    

    df = read_object(input_path, file_type="xlsx")
    #print(df)

    df.columns = df.columns.str.replace(r"\n", r"", regex=True)
    processed_df = df.replace(r"\n", r"", regex=True)
#    processed_df = processed_df.replace("’", "'", regex=True)
    processed_df["KIT description"] = processed_df["KIT description"].replace(
        "Safety and PK of antifungals", "Safety & PK of antifungals"
    )

    cols = [
        "CFM Form ID",
        "DFO Description",
        "KIT description",
        "Insights Category",
    ]
    processed_df = processed_df[cols]

    processed_df["DFO Description"] = (
        processed_df["DFO Description"]
            .str.encode("ascii", "ignore")
            .str.decode("unicode-escape")
    )
    processed_df = processed_df[processed_df['KIT description'] != ""]
    processed_df['KIT description'] = processed_df['KIT description'].replace('NA', 'Not Applicable')

    print("Null values in \'DFO Description\' columns {} being dropped "
          .format(processed_df[processed_df["DFO Description"].isnull()].index.tolist()))

    processed_df = processed_df[processed_df["Insights Category"] != "Repeat DFO"]
    processed_df = processed_df[processed_df["Insights Category"] != "Null DFO"]
    processed_df = processed_df[processed_df["Insights Category"] != "Inaccurate pdt category"]
    processed_df.dropna(subset=["DFO Description"], inplace=True)
    processed_df.reset_index(inplace=True, drop=True)
    
    

    return processed_df


"""def preprocess_cresemba_raw_for_insights(insights_dict, f_input_path=None, f_output_path=None):
    
    This function takes input as cresemba excel file and converts into csv.
    csv output file contains 4 columns: DFO Description, KIT (if available)
    KIT description and Insight Categroy
    Argument:
        f_input_path: str
            Filepath denonating the directory containing Cresemba excel file.
    Returns:
        Dataframe: pd.Dataframe
    
    current_path = os.path.dirname(__file__)
    project_path = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
    if f_input_path is None:
        cresemba_input_name = "Medical Insights_May2021_DFOs_SV_Cresemba_(Isavuconazole).xlsx"
        f_input_path = os.path.join(*[project_path, "data/DFOs/raw_DFOs/", cresemba_input_name])

        insights_labels_name = "cresemba_insights.txt"
        insights_labels_path = os.path.join(*[project_path, "data/DFOs/raw_DFOs/", insights_labels_name])

    if f_output_path is None:
        cresemba_output_name = "cresemba_dataset.csv"
        f_output_path = os.path.join(*[project_path, "data/DFOs/processed_DFOs/", cresemba_output_name])

    df = pd.read_excel(f_input_path, engine="openpyxl", sheet_name="MEDIC_OneMed_DFO_Isavuconazole", skiprows=1,
                       keep_default_na=False)
    df.columns = df.columns.str.replace("\n", "", regex=True)
    processed_df = df.replace("\n", "", regex=True)
#    processed_df = processed_df.replace("’", "'", regex=True)
    processed_df["KIT description"] = processed_df["KIT description"].replace(
        "Safety and PK of antifungals", "Safety & PK of antifungals"
    )

    cols = [
        "CFM Form ID",
        "DFO Description",
        "KIT description",
        "Insights Category",
    ]
    processed_df = processed_df[cols]

    processed_df["DFO Description"] = (
        processed_df["DFO Description"]
            .str.encode("ascii", "ignore")
            .str.decode("unicode-escape")
    )
    processed_df = processed_df[processed_df['KIT description'] != ""]
    processed_df['KIT description'] = processed_df['KIT description'].replace('NA', 'Not Applicable')
    processed_df['Insights Category'] = processed_df['Insights Category'].replace('Preferred therapy',
                                                                                  'Preferred Therapy')

    insights_labels = list(insights_dict.keys())
    processed_df = processed_df[processed_df["Insights Category"].isin(insights_labels)]

    print("Null values in \'DFO Description\' columns {} being dropped "
          .format(processed_df[processed_df["DFO Description"].isnull()].index.tolist()))

    processed_df.dropna(subset=["DFO Description"], inplace=True)
    processed_df.reset_index(inplace=True, drop=True)
    processed_df.to_csv(f_output_path, index=False)

    return processed_df


def preprocess_zavicefta_raw_for_kit(f_input_path=None, f_output_path=None):
    '''
    This function takes input as dataframe and give output with columns without NA. Final output contains 3 columns DFO Description,
    KIT description, Insight Categroy

    '''

    current_path = os.path.dirname(__file__)
    project_path = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
    if f_input_path is None:
        cresemba_input_name = "Medical Insights_May2021_DFOs_SV_Zavicefta(Ceftazidime-Avibactam).xlsx"
        f_input_path = os.path.join(*[project_path, "data/DFOs/raw_DFOs/", cresemba_input_name])
    if f_output_path is None:
        cresemba_output_name = "zavicefta_dataset.csv"
        f_output_path = os.path.join(*[project_path, "data/DFOs/processed_DFOs/", cresemba_output_name])

    data = pd.read_excel(f_input_path, engine="openpyxl", sheet_name="MEDIC_OneMed_DFO_CeftazAvi", skiprows=1,
                         keep_default_na=False)
    processed_df = data[data['KIT description'] != ""]

    cols = [
        "CFM Form ID",
        "DFO Description",
        "KIT description",
        "Insights Category",
    ]
    processed_df = processed_df[cols]

    processed_df["DFO Description"] = (
        processed_df["DFO Description"].str.encode("ascii", "ignore").str.decode("unicode-escape"))

    processed_df['KIT description'] = processed_df['KIT description'].replace('NA', 'Not Applicable')

    print("Null values in \'DFO Description\' columns {} being dropped "
          .format(processed_df[processed_df["DFO Description"].isnull()].index.tolist()))

    processed_df = processed_df[processed_df["Insights Category"] != "Repeat DFO"]
    processed_df = processed_df[processed_df["Insights Category"] != "Null DFO"]
    processed_df = processed_df[processed_df["Insights Category"] != "Inaccurate pdt category"]
    processed_df.dropna(subset=["DFO Description"], inplace=True)
    processed_df.reset_index(inplace=True, drop=True)
    processed_df.to_csv(f_output_path, index=False)

    return processed_df


def preprocess_zavicefta_raw_for_insights(insights_dict, f_input_path=None, f_output_path=None):
    '''
    This function takes input as dataframe and give output with columns without NA. Final output contains 3 columns DFO Description,
    KIT description, Insight Categroy
    '''

    current_path = os.path.dirname(__file__)
    project_path = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
    if f_input_path is None:
        zavicefta_input_name = "Medical Insights_May2021_DFOs_SV_Zavicefta(Ceftazidime-Avibactam).xlsx"
        f_input_path = os.path.join(*[project_path, "data/DFOs/raw_DFOs/", zavicefta_input_name])

        insights_labels_name = "zavicefta_insights.txt"
        insights_labels_path = os.path.join(*[project_path, "data/DFOs/raw_DFOs/", insights_labels_name])

    if f_output_path is None:
        zavicefta_output_name = "zavicefta_dataset.csv"
        f_output_path = os.path.join(*[project_path, "data/DFOs/processed_DFOs/", zavicefta_output_name])

    data = pd.read_excel(f_input_path, engine="openpyxl", sheet_name="MEDIC_OneMed_DFO_CeftazAvi", skiprows=1,
                         keep_default_na=False)
    processed_df = data[data['KIT description'] != ""]

    cols = [
        "CFM Form ID",
        "DFO Description",
        "KIT description",
        "Insights Category",
    ]
    processed_df = processed_df[cols]

    processed_df["DFO Description"] = (
        processed_df["DFO Description"].str.encode("ascii", "ignore").str.decode("unicode-escape"))

    processed_df['KIT description'] = processed_df['KIT description'].replace('NA', 'Not Applicable')

    insights_labels = list(insights_dict.keys())
    processed_df = processed_df[processed_df["Insights Category"].isin(insights_labels)]

    print("Null values in \'DFO Description\' columns {} being dropped "
          .format(processed_df[processed_df["DFO Description"].isnull()].index.tolist()))

    processed_df.dropna(subset=["DFO Description"], inplace=True)
    processed_df.reset_index(inplace=True, drop=True)
    processed_df.to_csv(f_output_path, index=False)

    return processed_df"""

def cross_val(df, n_folds, label, seed=42):
    """
    This function split the dataset into stratified K folds
    and subsequently split it further to 80-20 train tes. Returns
    a list of tuple, for instance for 3-fold split it will return:-
    [(train_0,val_0,test_0),
     (train_1,val_1,test_1),
     (train_2,val_2,test_2)]
     :param file_path: directory/file path of the dataset
     :param folds: number of K-folds
     :param seed: random_state/seed value defaults to 42
     """

    # Perform stratified K-fold
    kf = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    data = []
    data_length = []

    # Perform subsequent 80-20 train test split
    for train_idx, test_idx in kf.split(df, df[label]):
        train1, test = df.iloc[train_idx], df.iloc[test_idx]

        train, val, _, _ = model_selection.train_test_split(
            train1, train1[label],
            test_size=0.20, random_state=seed,
            stratify=train1[label])
        data.append((train, val, test))
        data_length.append((len(train), len(val), len(test)))

    print("Length of full dataset:", len(df))
    print("Length of datasets after train, validation and test set split respectively:\n", data_length)

    return data, data_length


def pick_data_for_exp_scenario(folds, eval_test, dataset, label, label_mapping, labels_set):
    data = []
    for fold in folds:
        train_dfo = fold[0].loc[:, "DFO Description"]
        val_dfo = fold[1]["DFO Description"]
        test_dfo = fold[2]["DFO Description"]
        if eval_test == False:  # If evaluate on test set is false, then evaluate on validation set, thus train only with train set.
            train_data = train_dfo
            train_labels = fold[0].loc[:, label]
            test_data = val_dfo
            test_labels = fold[1].loc[:, label]
        else:
            train_data = pd.concat([train_dfo, val_dfo])
            train_labels = pd.concat([fold[0].loc[:, label], fold[1].loc[:, label]])
            test_data = test_dfo
            test_labels = fold[2].loc[:, label]

        train_labels = train_labels.map(label_mapping)
        test_labels = test_labels.map(label_mapping)

        data.append(((train_data, train_labels), (test_data, test_labels)))

    return data, labels_set
