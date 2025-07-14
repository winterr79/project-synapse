# src/05_continual_learning.py

import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# This is the main execution block
if __name__ == "__main__":
    # 1. Define Paths
    # Path to the model we want to continue training (our fine-tuned model)
    MODEL_TO_IMPROVE_PATH = os.path.join("models", "t5-small-finetuned-cnn")
    # Path where the newly improved model will be saved
    IMPROVED_MODEL_PATH = os.path.join("models", "t5-small-finetuned-cnn-v2")
    # Path to our collected feedback data
    FEEDBACK_DATA_PATH = os.path.join("data", "feedback_data.csv")

    # 2. Load the feedback data using Pandas
    print(f"Loading feedback data from: {FEEDBACK_DATA_PATH}")
    feedback_df = pd.read_csv(FEEDBACK_DATA_PATH)

    # 3. Convert the Pandas DataFrame into a Hugging Face Dataset object
    # The 'from_pandas' method makes this easy
    feedback_dataset = Dataset.from_pandas(feedback_df)
    print("Feedback data loaded into Hugging Face Dataset format.")

    # 4. Load the tokenizer and the model we want to improve
    print(f"Loading model to improve from: {MODEL_TO_IMPROVE_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TO_IMPROVE_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_TO_IMPROVE_PATH)

    # 5. Preprocess the feedback data (similar to our original preprocessing)
    def preprocess_feedback_function(examples):
        prefix = "summarize: "
        # The 'original_text' is our input
        inputs = [prefix + doc for doc in examples["original_text"]]
        # The 'corrected_summary' is our new target label
        targets = [doc for doc in examples["corrected_summary"]]
        
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(text_target=targets, max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Preprocessing feedback data...")
    tokenized_feedback_dataset = feedback_dataset.map(preprocess_feedback_function, batched=True)

    # 6. Define Training Arguments for this learning phase
    # We use a very small learning rate because we are fine-tuning an already-trained model
    # We also train for more epochs because the dataset is tiny
    training_args = Seq2SeqTrainingArguments(
        output_dir=IMPROVED_MODEL_PATH,
        num_train_epochs=10,  # Train for more epochs on this small, high-quality data
        per_device_train_batch_size=1, # Use a batch size of 1 for very small datasets
        weight_decay=0.01,
        logging_steps=1,
        save_strategy="no", # We will save manually at the end
        report_to="none",
    )

    # 7. Create the Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_feedback_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 8. Start the Continual Learning (Fine-tuning)
    print("Starting continual learning...")
    trainer.train()
    print("Learning complete.")

    # 9. Save the newly improved model
    print(f"Saving the improved model to {IMPROVED_MODEL_PATH}")
    trainer.save_model(IMPROVED_MODEL_PATH)
    print("Model V2 saved successfully.")