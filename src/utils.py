# src/utils.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Global cache for the model and tokenizer to avoid reloading them every time
# This is a simple optimization for our application
model_cache = {}

def summarize_text(text_to_summarize: str) -> str:
    """
    Loads the fine-tuned T5 model and generates a summary for the given text.
    Uses a simple cache to avoid reloading the model on every call.
    """
    model_name = "models/t5-small-finetuned-cnn"

    # Check if model and tokenizer are already in cache
    if model_name not in model_cache:
        print(f"Loading model and tokenizer for the first time: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_cache[model_name] = (model, tokenizer)
    else:
        print("Loading model and tokenizer from cache.")
        model, tokenizer = model_cache[model_name]

    # Prepare the input for the T5 model
    prefix = "summarize: "
    input_text = prefix + text_to_summarize

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode and return the summary
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return generated_summary