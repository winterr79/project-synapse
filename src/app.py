# src/app.py

import gradio as gr
import pandas as pd
import os
from utils import summarize_text # Import our summarization function

# Define the path for our feedback data
FEEDBACK_FILE_PATH = os.path.join("data", "feedback_data.csv")

# --- Backend Functions ---

def save_feedback(original_text, generated_summary, corrected_summary):
    """Saves the user's feedback to a CSV file."""
    try:
        # Create a new DataFrame with the feedback
        new_feedback = pd.DataFrame({
            'original_text': [original_text],
            'generated_summary': [generated_summary],
            'corrected_summary': [corrected_summary]
        })

        # Check if the file already exists
        if os.path.exists(FEEDBACK_FILE_PATH):
            # Append without writing the header
            new_feedback.to_csv(FEEDBACK_FILE_PATH, mode='a', header=False, index=False)
        else:
            # Write a new file with the header
            new_feedback.to_csv(FEEDBACK_FILE_PATH, mode='w', header=True, index=False)
        
        # Return a confirmation message to the user
        return "✅ Feedback saved successfully! Thank you."
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return f"❌ Error saving feedback: {e}"

def summarization_interface(article_text):
    """
    The main interface function. Takes article text, returns the generated
    summary to two places: the main summary output and the 'corrected' box.
    """
    summary = summarize_text(article_text)
    # The function now returns the summary twice.
    # The first goes to the 'Generated Summary' output box.
    # The second goes to the 'Corrected Summary' box to serve as a starting point for edits.
    return summary, summary

# --- Gradio UI Definition ---

if __name__ == "__main__":
    # Use gr.Blocks for more complex layouts with multiple components and buttons
    with gr.Blocks(title="Project Synapse") as app_interface:
        gr.Markdown("# Project Synapse: An Adaptive Summarizer")
        gr.Markdown("Enter text to summarize, then correct the summary to help the model learn.")

        with gr.Row():
            # Define the input components
            article_input = gr.Textbox(lines=15, label="Original Text", placeholder="Paste your article or text here...")
            
            with gr.Column():
                # Define the output components
                generated_summary_output = gr.Textbox(lines=5, label="Generated Summary")
                corrected_summary_input = gr.Textbox(lines=5, label="Corrected Summary (Edit here)", interactive=True)
        
        # Define the buttons
        summarize_button = gr.Button("Summarize")
        save_feedback_button = gr.Button("Save Feedback")

        # Define a component to show confirmation messages
        status_message = gr.Markdown()

        # --- Define Component Interactions (How the UI works) ---

        # When the 'Summarize' button is clicked:
        summarize_button.click(
            fn=summarization_interface,      # Call our main function
            inputs=[article_input],           # Pass the article text as input
            outputs=[generated_summary_output, corrected_summary_input] # Update both summary boxes
        )

        # When the 'Save Feedback' button is clicked:
        save_feedback_button.click(
            fn=save_feedback,               # Call the save function
            inputs=[article_input, generated_summary_output, corrected_summary_input], # Pass all 3 text boxes
            outputs=[status_message]        # Show the confirmation message
        )

    # Launch the web server
    print("Launching Gradio app...")
    app_interface.launch()