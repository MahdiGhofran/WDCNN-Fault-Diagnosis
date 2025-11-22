import os
import urllib.request

# Base URL for CWRU Data
BASE_URL = "https://engineering.case.edu/sites/default/files/"

# List of files to download (Example subset)
# 97.mat: Normal Baseline
# 105.mat: Inner Race Fault, 0.007 inches, Load 0
# 118.mat: Ball Fault, 0.007 inches, Load 0
# 130.mat: Outer Race Fault, 0.007 inches, Load 0
FILES = {
    "97.mat": "97.mat",
    "105.mat": "105.mat",
    "118.mat": "118.mat",
    "130.mat": "130.mat"
}

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to find 'data'
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    for filename, url_suffix in FILES.items():
        url = BASE_URL + url_suffix
        output_path = os.path.join(DATA_DIR, filename)
        
        if os.path.exists(output_path):
            print(f"{filename} already exists.")
            continue
            
        print(f"Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_data()

