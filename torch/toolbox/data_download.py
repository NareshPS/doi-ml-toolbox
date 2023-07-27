
import os
import requests
import zipfile

from pathlib import Path

def download_data(
    source: str,
    destination: str,
    remove_source: bool = True
) -> Path:
    """Downloads zipped dataset from source and unzips to the destination directory.
    
    Usage Example:
        download_data(
            source='https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
            destination='pizza_steak_sushi',
        )
    """
    # 1. Setup path to data folder
    data_path = Path(os.path.expanduser('~/.datasets/'))
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists. Skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # 2. Download the source file
        remote_file_name = Path(source).name
        with open(data_path / remote_file_name, "wb") as f:
            request = requests.get(source)
            print(f'[INFO] Downloading {remote_file_name} from {source}')
            f.write(request.content)

        # 3. Unzip the source file into the destination directory
        with zipfile.ZipFile(data_path / remote_file_name, "r") as zip_ref:
            print(f"[INFO] Unzipping {remote_file_name} data...") 
            zip_ref.extractall(image_path)
        
        # 4. Cleanup the downloaded file.
        if remove_source: os.remove(data_path / remote_file_name)
    
    return image_path
