import zipfile

zip_file = "data/sample.zip"
output_dir = "data/"

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(output_dir)