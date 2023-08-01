import zipfile
def unzip_file(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract all files in the zip to the current working directory
        zip_ref.extractall()
    print("File has been successfully unzipped.")
    
zip_path = "example.zip"