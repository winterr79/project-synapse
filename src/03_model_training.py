import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# This is the main execution block
if __name__ == "__main__":
    # 1. Define constants and paths
    MODEL_CHECKPOINT = "google-t5/t5-small"
    PROCESSED_DATASET_PATH = os.path.join("data", "cnn_dailymail_tokenized")
    # This is where our fine-tuned model will be saved
    FINETUNED_MODEL_PATH = os.path.join("models", "t5-small-finetuned-cnn")

    # 2. Load the tokenized dataset and tokenizer
    print("Loading tokenized dataset and tokenizer...")
    tokenized_datasets = load_from_disk(PROCESSED_DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    

    # 3. Load the pre-trained model
    # AutoModelForSeq2SeqLM is used for sequence-to-sequence tasks like summarization
    print(f"Loading pre-trained model: {MODEL_CHECKPOINT}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # 4. Define the Data Collator
    # This will dynamically pad the inputs and labels in each batch
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 5. Define Training Arguments
    # Seq2SeqTrainingArguments contains all the hyperparameters for the training process
    print("Defining training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=FINETUNED_MODEL_PATH,
        num_train_epochs=1,
        warmup_steps=500,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        save_total_limit=3,  # Only keep the best 3 checkpoints to save disk space
        report_to="none",
        dataloader_num_workers=1,
    )

    # 6. Create the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )

    # 7. Start Fine-Tuning
    print("Starting model fine-tuning...")
    trainer.train()
    print("Training complete.")

    # 8. Save the final model and tokenizer
    print(f"Saving the best model to {FINETUNED_MODEL_PATH}")
    trainer.save_model(FINETUNED_MODEL_PATH)
    print("Model saved successfully.")