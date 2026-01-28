# %%
from glob import glob
from sklearn.model_selection import train_test_split
import os
import shutil
# Some code for preprocessing 

def get_file_list(data_dir, file_extension="*.npz"):
    file_list = [f for f in glob(os.path.join(data_dir, file_extension))]
    return file_list

def split_dataset(file_list, train_path, test_path, test_size, random_state=42):
    # Ensure output directories exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Split the dataset
    train_files, test_files = train_test_split(file_list, test_size=test_size, random_state=random_state)

    # Copy files to respective directories
    for file in train_files:
        shutil.copy2(file, train_path)
    for file in test_files:
        shutil.copy2(file, test_path)
    return train_files, test_files

if __name__ == "__main__":
    data_dir = "./data/all" # Directory containing all data files
    train_path = "./data/train" # Directory to save training files
    test_path = "./data/test"   # Directory to save testing files
    test_size = 0.2 # splits data into 80% train and 20% test
    file_list = get_file_list(data_dir)
    print(f"found {len(file_list)} files.")
    train_files, test_files = split_dataset(file_list, train_path, test_path, test_size=test_size)
    print(f"Training files: {len(train_files)}, Testing files: {len(test_files)}")
# %%