# Import the necessary library from the 'datasets' package
from datasets import load_dataset

# This is the main execution block, it runs when you execute the script directly
if __name__ == "__main__":
    # Define the name of the dataset we want to download from Hugging Face
    # We are using 'cnn_dailymail' as a robust alternative to 'samsum'.
    # We will use a specific version '3.0.0' for consistency.
    dataset_name = "cnn_dailymail"
    dataset_version = "3.0.0"
    
    # Use the load_dataset function to download and cache the dataset
    print(f"Downloading dataset: {dataset_name} (version {dataset_version})")
    # Note: This dataset is large and may take a significant time to download.
    dataset = load_dataset(dataset_name, dataset_version)
    
    # The dataset is a dictionary-like object, let's see what's inside
    print("\nDataset structure:")
    print(dataset)
    
    # Let's inspect one example from the 'train' split to understand its columns
    print("\nExample from the 'train' split:")
    example = dataset['train'][0]
    
    # Print the article and its summary (highlights)
    print("\nARTICLE:")
    print(example['article'])
    print("\nSUMMARY (HIGHLIGHTS):")
    print(example['highlights'])