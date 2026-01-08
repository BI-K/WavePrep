import os
import pandas as pd
import numpy as np
import torch
import argparse
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_dataset_pt_process_folder(folder_path: str, folder_name: str) -> Tuple[List, List]:
    """Process a single folder and return samples."""
    samples = []
    labels = []
    
    observation_files = [f for f in os.listdir(os.path.join(folder_path, folder_name, "observation")) if f.endswith('.csv')]
    for file in observation_files:
        file_path = os.path.join(folder_path, folder_name, "observation", file)
        samples_df = pd.read_csv(file_path)
        labels_df = pd.read_csv(os.path.join(folder_path, folder_name, "prediction", file))
        samples.append(samples_df.values.tolist())
        labels.append(labels_df.values.tolist())
    
    return samples, labels


def create_dataset_pt(path: str, is_train: bool):
    """Create dataset dictionary by reading CSV files in parallel."""
    samples = []
    labels = []

    dataset_name = path.split(os.path.sep)[:-1]

    if is_train:
        path = os.path.join(path, 'train')
    else:
        path = os.path.join(path, 'test')
    # read all folders in the path
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    # Parallelize folder processing using ThreadPoolExecutor (I/O-bound task)
    max_workers = min(os.cpu_count() or 1, len(folders))  # Don't spawn more workers than folders
    # max_workers = 1
    print(f"Processing {len(folders)} folders with {max_workers} workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_dataset_pt_process_folder, path, folder) for folder in folders]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing folders"):
            try:
                folder_samples, folder_labels = future.result()
                samples.extend(folder_samples)
                labels.extend(folder_labels)
            except Exception as e:
                print(f"Error processing folder: {e}")

    dataset_dict = {
        "samples": samples,
        "labels": labels
    }

    print(f"Created dataset with {len(samples)} samples")
    torch.save(dataset_dict, os.path.join(path, f'{"train" if is_train else "test"}_{dataset_name}.pt'))


def main():
    parser = argparse.ArgumentParser(description='Convert observation/prediction CSV files to PyTorch dataset format')

    parser.add_argument('--path', 
                       type=str, 
                       help='Path to the split_data directory containing train/test folders')
    
    args = parser.parse_args()
    
    # Validate path
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist!")
        sys.exit(1)
    
    # Validate that it's a directory
    if not os.path.isdir(args.path):
        print(f"Error: '{args.path}' is not a directory!")
        sys.exit(1)

    # create_dataset_pt(args.path, True)
    # create only test file
    create_dataset_pt(args.path, False)

if __name__ == "__main__":
    main()
