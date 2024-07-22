from kaggle.api.kaggle_api_extended import KaggleApi
import os

def main():
    api = KaggleApi()
    api.authenticate()  # Make sure you have your kaggle.json file set up correctly
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')

    # Download the dataset
    api.dataset_download_files('nih-chest-xrays/data', path=data_dir, unzip=False)
    # api.dataset_download_files('nih-chest-xrays/sample', path=data_dir, unzip=False)
        
if __name__ == "__main__":
    main()