import os
import shutil

RAW_DATA = "/home/moabouag/.cache/kagglehub/datasets/salmansajid05/oral-diseases/versions/3"
CURRENT_DIR = os.getcwd()

target_data_path = None

print("Searching inside:", RAW_DATA)


for root, dirs, files in os.walk(RAW_DATA):
    for d in dirs:
        folder_lower = d.lower()

        if "caries" in folder_lower and "yolo" in folder_lower:
            candidate = os.path.join(root, d)
            print("Found YOLO dataset folder:", candidate)

            inner_data = os.path.join(candidate, "Data")
            if os.path.exists(inner_data):
                target_data_path = candidate
                break

    if target_data_path:
        break

if not target_data_path:
    print(" Could not find YOLO Data folder inside RAW_DATA")
else:
    print("Located YOLO Detect Dataset at:", target_data_path)


    dst = os.path.join(CURRENT_DIR, "oral_yolo_dataset")


    if os.path.exists(dst):
        print("Removing existing 'oral_yolo_dataset'...")
        shutil.rmtree(dst)

    print("Copying dataset to:", dst)
    shutil.copytree(target_data_path, dst)

    print("\n YOLO dataset extracted to:", dst)
