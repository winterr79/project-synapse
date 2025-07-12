import os
from datasets import load_dataset

# This is the main execution block
if __name__ == "__main__":
    # Define the name of the dataset and version
    dataset_name = "cnn_dailymail"
    dataset_version = "3.0.0"
    
    # Define the local path to save the data
    # This creates a path like 'data/cnn_dailymail_3.0.0'
    local_save_path = os.path.join("data", f"{dataset_name}_{dataset_version}")

    # Use the load_dataset function to download and cache the dataset
    print(f"Downloading dataset: {dataset_name} (version {dataset_version})")
    dataset = load_dataset(dataset_name, dataset_version)
    
    # *** NEW, CRITICAL STEP ***
    # Save the downloaded dataset to our local project directory
    print(f"Saving dataset to local path: {local_save_path}")
    dataset.save_to_disk(local_save_path)

    print("\nDataset structure:")
    print(dataset)
    
    print("\nData ingestion and local save complete.")