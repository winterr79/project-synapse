# Project Synapse: An Adaptive Summarizer

**Project Synapse** is a proof-of-concept generative AI application designed to summarize text. Its core feature is a built-in feedback loop that allows the model to perform **continual learning**, improving its summarization quality and style based on user corrections over time.

This project was built to explore and demonstrate the entire lifecycle of a modern AI application: from data ingestion and model fine-tuning to building an interactive UI and implementing a mechanism for learning from experience.

---

## Project Status: Complete

The project has successfully achieved its primary goals:

- **Fine-Tuning:** A base `google-t5/t5-small` model was successfully fine-tuned on the `cnn_dailymail` dataset to create a capable baseline summarizer (V1).
- **Inference:** A functional inference pipeline was built to use the trained model.
- **Interactive UI:** A web application was created using Gradio, allowing users to interact with the model.
- **Feedback Loop:** A mechanism for users to correct the model's output and save it as new training data was implemented.
- **Continual Learning:** A script was developed to re-train the model on the collected feedback, creating improved versions (V2, V3).
- **Demonstration:** The end-to-end loop was tested, proving that the model's behavior can be influenced by user feedback, although significant change requires a larger feedback dataset.

---

## Key Learnings & Observations

- **The "Inertia" of Large Models:** The final experiments showed that changing the fundamental behavior of a model trained on a large dataset requires more than a handful of feedback examples. The model's pre-existing knowledge has significant "inertia."
- **Environment Management is Critical:** A significant portion of the development journey involved solving complex dependency conflicts between libraries (`torch`, `transformers`, `accelerate`, `fsspec`) and adapting to different execution environments (local Windows, Google Colab, Kaggle). A pinned, known-good `requirements.txt` is essential for reproducibility.
- **The Power of Explicit Commands:** Relying on implicit or automatic framework behaviors (like auto-resuming from checkpoints) can be fragile. Explicitly commanding the desired behavior (e.g., `trainer.train(resume_from_checkpoint=True)`) is more robust.

---

## How to Run This Project

### 1. Setup

**Prerequisites:**
- Python 3.9+
- Git

**Installation:**

Clone the repository:

```bash
git clone https://github.com/winterr79/project-synapse.git
cd project-synapse
```

Create and activate a Python virtual environment:

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Running the Application

The main entry point is the Gradio application. This allows you to generate summaries and provide feedback.

Ensure you have a trained model in the models/ directory (e.g., t5-small-finetuned-cnn).

Run the app from the project root directory:

```bash
python src/app.py
```

Open the local URL (e.g., http://127.0.0.1:7860) in your browser.

### 3. The Continual Learning Loop

To improve the model based on your collected feedback:

Use the web app to generate summaries and save your corrections. This will populate data/feedback_data.csv.

Run the continual learning script:

```bash
python src/05_continual_learning.py
```

This will create a new, improved model version in the models/ directory (e.g., t5-small-finetuned-cnn-v2).

Update src/app.py to include the new model path in the dropdown to test its performance.

## Project Structure

```bash
project-synapse/
├── data/
│   ├── cnn_dailymail_3.0.0/          # (Ignored) Raw dataset
│   ├── cnn_dailymail_tokenized/      # (Ignored) Processed dataset
│   └── feedback_data.csv             # User corrections for continual learning
├── models/
│   └── t5-small-finetuned-cnn/       # (Ignored) The fine-tuned model checkpoints
├── src/
│   ├── 01_data_ingestion.py          # Downloads raw data
│   ├── 02_data_preprocessing.py      # Tokenizes data for training
│   ├── 03_model_training.py          # Main script for fine-tuning the base model
│   ├── 04_inference.py               # Command-line script for testing inference
│   ├── 05_continual_learning.py      # Script to re-train on feedback data
│   ├── app.py                        # The Gradio web application
│   └── utils.py                      # Reusable utility functions (e.g., summarize_text)
├── .gitignore
├── README.md
└── requirements.txt
```