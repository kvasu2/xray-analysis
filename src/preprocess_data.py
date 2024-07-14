import zipfile

def main():
    zip_file = "data/sample.zip"
    output_dir = "data/"

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

if __name__ == "__main__":
    main()