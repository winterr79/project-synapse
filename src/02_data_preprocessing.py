import os
from datasets import load_from_disk
from transformers import AutoTokenizer

# This is the main execution block
if __name__ == "__main__":
    # 1. Define constants and paths
    MODEL_CHECKPOINT = "google-t5/t5-small"
    # The path where the raw dataset was saved by the previous script
    RAW_DATASET_PATH = os.path.join("data", "cnn_dailymail_3.0.0") 
    # The path where we will save our new processed dataset
    PROCESSED_DATASET_PATH = os.path.join("data", "cnn_dailymail_tokenized")

    # 2. Load the tokenizer for our chosen model
    print(f"Loading tokenizer for model: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # 3. Load the raw dataset from disk
    print(f"Loading raw dataset from: {RAW_DATASET_PATH}")
    raw_dataset = load_from_disk(RAW_DATASET_PATH)

    # 4. Define the preprocessing function
    # This function will be applied to every example in our dataset
    def preprocess_function(examples):
        # T5 models require a prefix for summarization tasks
        prefix = "summarize: "
        
        # Prepare the inputs by adding the prefix to each article
        inputs = [prefix + doc for doc in examples["article"]]
        
        # Tokenize the inputs (articles)
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        # Tokenize the outputs (summaries)
        # In T5, the 'labels' are the tokenized target text
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["highlights"], max_length=128, truncation=True)

        # Add the tokenized labels to our model inputs
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 5. Apply the preprocessing function to the entire dataset
    print("Applying preprocessing function to the dataset...")
    # The .map() function applies the function to all examples.
    # batched=True processes multiple examples at once for speed.
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

    # 6. Save the processed dataset to disk
    print(f"Saving tokenized dataset to: {PROCESSED_DATASET_PATH}")
    tokenized_dataset.save_to_disk(PROCESSED_DATASET_PATH)
    print("Preprocessing complete.")