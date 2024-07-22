import zipfile
import os
import pandas as pd


def sample(unzip = False):

    if unzip:
        zip_file = "data/sample.zip"
        output_dir = "data/"

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory
    parent_dir = os.path.dirname(current_dir)

    metadata = pd.read_csv(os.path.join(parent_dir,"data" , 'sample_labels.csv'))

    conditions = [
        "No Finding",
        "Hernia",
        "Pneumonia",
        "Fibrosis",
        "Edema",
        "Emphysema",
        "Cardiomegaly",
        "Pleural_Thickening",
        "Consolidation",
        "Pneumothorax",
        "Mass",
        "Nodule",
        "Atelectasis",
        "Effusion",
        "Infiltration"
    ]

    for condition in conditions:
        for condition in conditions:
            metadata.loc[metadata['Finding Labels'].str.contains(condition), condition] = 1
            metadata.loc[~metadata['Finding Labels'].str.contains(condition), condition] = 0


    labels = metadata[["Image Index"]+conditions]
    labels.to_csv(os.path.join(parent_dir, 'data', 'sample', 'labels.csv'), index=False)

def data(unzip = False):
    
    if unzip:
        zip_file = "data/data.zip"
        output_dir = "data/"

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory
    parent_dir = os.path.dirname(current_dir)

    metadata = pd.read_csv(os.path.join(parent_dir,"data" , 'Data_Entry_2017.csv'))

    conditions = [
        "No Finding",
        "Hernia",
        "Pneumonia",
        "Fibrosis",
        "Edema",
        "Emphysema",
        "Cardiomegaly",
        "Pleural_Thickening",
        "Consolidation",
        "Pneumothorax",
        "Mass",
        "Nodule",
        "Atelectasis",
        "Effusion",
        "Infiltration"
    ]

    for condition in conditions:
        for condition in conditions:
            metadata.loc[metadata['Finding Labels'].str.contains(condition), condition] = 1
            metadata.loc[~metadata['Finding Labels'].str.contains(condition), condition] = 0


    labels = metadata[["Image Index"]+conditions]
    labels.to_csv(os.path.join(parent_dir, 'data', 'labels.csv'), index=False)

if __name__ == "__main__":
    sample(False)
    # data()