# src/app.py

import gradio as gr
import os
from utils import summarize_text

# --- Constants and Setup ---
FEEDBACK_FILE_PATH = os.path.join("data", "feedback_data.csv")
MODEL_V1_PATH = os.path.join("models", "t5-small-finetuned-cnn")
MODEL_V2_PATH = os.path.join("models", "t5-small-finetuned-cnn-v2")
MODEL_V3_PATH = os.path.join("models", "t5-small-finetuned-cnn-v3")

# --- Backend Functions ---
# (The save_feedback function remains the same as before)
def save_feedback(original_text, generated_summary, corrected_summary):
    try:
        import csv
        header = ['original_text', 'generated_summary', 'corrected_summary']
        new_feedback = {'original_text': original_text, 'generated_summary': generated_summary, 'corrected_summary': corrected_summary}
        file_exists = os.path.exists(FEEDBACK_FILE_PATH) and os.path.getsize(FEEDBACK_FILE_PATH) > 0
        with open(FEEDBACK_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_feedback)
        return "✅ Feedback saved successfully! Thank you."
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return f"❌ Error saving feedback: {e}"

# Update the interface function
def summarization_interface(model_selection, article_text):
    if model_selection == "V2 (Improved)":
        model_path = MODEL_V2_PATH
    elif model_selection == "V3 (Latest)":
        model_path = MODEL_V3_PATH
    else:
        model_path = MODEL_V1_PATH
    
    summary = summarize_text(article_text, model_path)
    return summary, summary

# --- Gradio UI Definition ---
if __name__ == "__main__":
    with gr.Blocks(title="Project Synapse") as app_interface:
        gr.Markdown("# Project Synapse: An Adaptive Summarizer")
        gr.Markdown("Select a model version, summarize text, then correct the summary to help the model learn.")

        model_choice = gr.Dropdown(
            ["V1 (Original)", "V2 (Improved)", "V3 (Latest)"], label="Model Version", value="V3 (Latest)"
        )

        with gr.Row():
            article_input = gr.Textbox(lines=15, label="Original Text", placeholder="Paste your article or text here...")
            with gr.Column():
                generated_summary_output = gr.Textbox(lines=5, label="Generated Summary")
                corrected_summary_input = gr.Textbox(lines=5, label="Corrected Summary (Edit here)", interactive=True)
        
        summarize_button = gr.Button("Summarize")
        save_feedback_button = gr.Button("Save Feedback")
        status_message = gr.Markdown()

        # --- Component Interactions ---
        summarize_button.click(
            fn=summarization_interface,
            inputs=[model_choice, article_input], # Pass the dropdown value as an input
            outputs=[generated_summary_output, corrected_summary_input]
        )
        save_feedback_button.click(
            fn=save_feedback,
            inputs=[article_input, generated_summary_output, corrected_summary_input],
            outputs=[status_message]
        )

    print("Launching Gradio app...")
    app_interface.launch()