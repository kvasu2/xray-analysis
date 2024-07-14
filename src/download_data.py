from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()  # Make sure you have your kaggle.json file set up correctly

    # Download the dataset
    # api.dataset_download_files('nih-chest-xrays/data', path='data', unzip=False)
    api.dataset_download_files('nih-chest-xrays/sample', path='data', unzip=False)
        
if __name__ == "__main__":
    main()