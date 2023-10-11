# RAW DATASET LINK: https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip

import requests, zipfile, os

DATASET_URL = 'https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip'
DATASET_RAW_PATH_RELATIVE = '/../../data/raw/filtered_paranmt.zip'
DATASET_RAW_PATH_ABSOLUTE = os.path.dirname(__file__) + DATASET_RAW_PATH_RELATIVE
DATASET_RAW_DIR_RELATIVE = '/../../data/raw/filtered_paranmt/'
DATASET_RAW_DIR_ABSOLUTE = os.path.dirname(__file__) + DATASET_RAW_DIR_RELATIVE

# src: https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
def download_dataset(url, save_path, chunk_size=128):
    print('fetching url...')
    r = requests.get(url, stream=True)
    print('Fetched Successfully.')
    with open(save_path, 'wb') as fd:
        print('Writing... (this might take a while)')
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    print("Finished downloading successfully.")

def extract_zip(zip_path, destination_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print('Extracting...')
        zip_ref.extractall(destination_dir)
    print('Finished extracting.')

def get_dataset():
    try:
        # Directory exists, and not empty
        if os.path.isdir(DATASET_RAW_DIR_ABSOLUTE) and len(os.listdir(DATASET_RAW_DIR_ABSOLUTE)) != 0:
            print('Dataset already exists at ' + DATASET_RAW_PATH_ABSOLUTE)
        elif os.path.exists(DATASET_RAW_PATH_ABSOLUTE):
            print(f"Dataset doesn't exist at {DATASET_RAW_DIR_ABSOLUTE}, but .zip file is found")
            extract_zip(zip_path=DATASET_RAW_PATH_ABSOLUTE, destination_dir=DATASET_RAW_DIR_ABSOLUTE)
        else:
            print('Dataset does not exist, downloading...')
            download_dataset(DATASET_URL, DATASET_RAW_PATH_ABSOLUTE)
            extract_zip(zip_path=DATASET_RAW_PATH_ABSOLUTE, destination_dir=DATASET_RAW_DIR_ABSOLUTE)
        
    except Exception as e:
        print('An error occured: ' + e)
        raise e
    finally:
        print('All set. The dataset can be found in project_root_dir/data/raw.')
