import scipy.io as sio
import os

data_dir = r"E:\AsusIran\Documents\Resume\Electrical Engineering-Control-Master\KNTU\Third semester\industrial automation\WDCNN_Fault_Diagnosis\data"
files = ["97.mat", "105.mat"]

for f in files:
    path = os.path.join(data_dir, f)
    try:
        mat = sio.loadmat(path)
        print(f"--- Keys in {f} ---")
        # Filter out __header__, __version__, etc.
        keys = [k for k in mat.keys() if not k.startswith("__")]
        print(keys)
    except Exception as e:
        print(f"Error reading {f}: {e}")


