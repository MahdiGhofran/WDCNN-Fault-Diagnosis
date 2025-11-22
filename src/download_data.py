import os
import urllib.request

# Base URL for CWRU Data
BASE_URL = "https://engineering.case.edu/sites/default/files/"

# Standard 10 Classes (12k Fan End / Drive End data usually used)
# Using Load 0 (approx 1797 RPM) for consistency
FILES = {
    "97.mat": "97.mat",    # Normal
    "105.mat": "105.mat",  # IR 0.007"
    "118.mat": "118.mat",  # Ball 0.007"
    "130.mat": "130.mat",  # OR 0.007" (@6:00)
    "169.mat": "169.mat",  # IR 0.014"
    "185.mat": "185.mat",  # Ball 0.014"
    "197.mat": "197.mat",  # OR 0.014" (@6:00)
    "209.mat": "209.mat",  # IR 0.021"
    "222.mat": "222.mat",  # Ball 0.021"
    "234.mat": "234.mat"   # OR 0.021" (@6:00)
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
