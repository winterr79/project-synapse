import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# This is the main execution block
if __name__ == "__main__":
    # 1. Define the path to our fine-tuned model
    # This is the model we trained and downloaded from Kaggle
    FINETUNED_MODEL_PATH = os.path.join("models", "t5-small-finetuned-cnn")

    # 2. Load the tokenizer and the fine-tuned model from the local path
    print(f"Loading model from: {FINETUNED_MODEL_PATH}")
    # The tokenizer is saved with the model, so we load it from the same directory
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    # We load the model itself
    model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL_PATH)
    print("Model loaded successfully.")

    # 3. Prepare a sample text to summarize
    # This is a long article about the discovery of a new dinosaur
    text_to_summarize = """
    Scientists have announced the discovery of a new species of dinosaur in Argentina.
    The colossal creature, named 'Patagotitan mayorum', is believed to be the largest
    animal to have ever walked the Earth. The fossils were first discovered in 2012 by a
    local farm worker in the Patagonian desert. A team of paleontologists from the
    Museum of Paleontology Egidio Feruglio spent years excavating the site, unearthing
    over 150 fossilized bones from at least six different individuals.
    Based on the size of its femur (thigh bone), which is over 2.4 meters (8 feet) long,
    researchers estimate that Patagotitan weighed around 70 metric tons – as much as 10
    African elephants – and measured nearly 40 meters (130 feet) from head to tail.
    The dinosaur belongs to a group known as titanosaurs, which were long-necked,
    plant-eating sauropods that lived during the Late Cretaceous period, about 100
    million years ago. The discovery provides new insights into the upper limits of
    how large land animals can grow.
    """

    # 4. Run the inference pipeline
    print("\nSummarizing the following text:\n" + "="*50 + text_to_summarize + "="*50)
    
    # T5 requires a prefix for summarization
    prefix = "summarize: "
    input_text = prefix + text_to_summarize

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary by passing the tokenized input to the model
    # We can control the length and quality of the summary with these parameters
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode the generated tokens back into a human-readable string
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 5. Print the result
    print("\nGenerated Summary:\n" + "="*50)
    print(generated_summary)
    print("="*50)