import zipfile
import os
import pandas as pd



def main():
    zip_file = "data/sample.zip"
    output_dir = "data/"

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    metadata = pd.read_csv(os.path.join(parent_dir,"data" , 'sample_labels.csv'))

    conditions = [
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


    labels = metadata[["Image Index","Emphysema"]]
    labels.to_csv(os.path.join(parent_dir, 'data', 'sample', 'labels.csv'), index=False)


if __name__ == "__main__":
    main()