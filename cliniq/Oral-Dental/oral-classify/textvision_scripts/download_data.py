# download_data.py
# Download train/test zips from Google Drive and unzip.
#
# Example:
#   python download_data.py --train_id <ID> --test_id <ID>

import argparse
import os
import zipfile
import gdown
from config import Paths

def gdown_file(file_id: str, output_name: str) -> None:
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_name, quiet=False)

def unzip(zip_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_id", type=str, required=True, help="Google Drive file id for train.zip")
    p.add_argument("--test_id", type=str, required=True, help="Google Drive file id for test.zip")
    p.add_argument("--out_root", type=str, default=Paths().data_root, help="Output root directory")
    args = p.parse_args()

    paths = Paths(data_root=args.out_root)

    gdown_file(args.train_id, paths.train_zip)
    gdown_file(args.test_id, paths.test_zip)

    unzip(paths.train_zip, paths.data_root)
    unzip(paths.test_zip, paths.data_root)

    print("Done.")
    print("Train dir:", paths.train_dir)
    print("Test dir:", paths.test_dir)

if __name__ == "__main__":
    main()
