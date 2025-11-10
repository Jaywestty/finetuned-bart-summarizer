# 📰 News Text Summarizer (Fine-Tuned BART-base)

## 📘 Project Overview

This project fine-tunes the **BART-base** model from Hugging Face Transformers to automatically summarize lengthy news articles into concise, factual highlights. The summarizer aims to preserve key information while enhancing readability, making it ideal for users who need **quick insights** from large volumes of news data.

The model was trained and evaluated on the **CNN/DailyMail dataset**, a widely used benchmark for abstractive summarization tasks. Training was performed on Google Colab using GPU acceleration.

---

## 🎯 Objective

To develop an abstractive text summarizer that:

* Generates human-like news summaries.
* Maintains factual accuracy and coherence.
* Runs efficiently on limited compute resources (e.g., Colab free-tier).

---

## 🧠 Model Architecture

* **Base Model:** `facebook/bart-base`

### Why BART?

* Performs strongly on **abstractive summarization** tasks.
* Balances summarization quality with computational efficiency.
* Requires minimal architectural modification for fine-tuning.

---

## 🗂 Dataset

* **Dataset Used:** CNN/DailyMail
* **Provider:** Hugging Face Datasets

Due to Colab’s limited GPU resources, a **subset** of the dataset was used:

* **Training samples:** 8,000
* **Validation samples:** 800
* **Test samples:** 800

Each record includes:

* `article`: The full news text.
* `highlights`: The reference summary used for supervision.

---

## ⚙️ Preprocessing

Preprocessing followed the BART-specific requirements:

* **Tokenization** of both article and summary text.
* Padding and truncation to a fixed sequence length.
* Conversion into input-label pairs for Seq2Seq fine-tuning.

---

## 🧩 Training Setup

* **Environment:** Google Colab (T4 GPU)
* **Frameworks:** PyTorch, Hugging Face Transformers, Datasets
* **Evaluation Metric:** ROUGE (1, 2, Lsum)

| Parameter | Value |
| :--- | :--- |
| Model | BART-base |
| Epochs | 3 |
| Batch Size | 8 |
| Learning Rate | 3e-5 |
| Evaluation Metric | ROUGE (1, 2, Lsum) |

Training was monitored via loss curves, with model checkpoints saved after each epoch.

---

## 🏆 Best Model Performance

After analysis, **epoch 2** produced the best balance between training stability and evaluation metrics.

| Metric | Score |
| :--- | :--- |
| Training Loss | 1.60 |
| Validation Loss | 2.23 |
| ROUGE-1 | 24.6 |
| ROUGE-2 | 9.5 |
| ROUGE-Lsum | 22.48 |

The epoch 2 checkpoint was selected for deployment.

---

## 🔍 Evaluation

The model was evaluated on the test subset using the **ROUGE metric** suite:

* **ROUGE-1:** Measures unigram overlap (content fidelity).
* **ROUGE-2:** Measures bigram overlap (fluency).
* **ROUGE-Lsum:** Measures longest common subsequence (structure and readability).

Results indicate that the fine-tuned model generalizes well without overfitting.

---

## 💡 Example Inference

The following Python code snippet demonstrates how to load and use the fine-tuned model for inference:

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# Ensure you replace "path_to_your_finetuned_model" with the actual path or Hugging Face repository ID
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("path_to_your_finetuned_model")

text = """The president met with global leaders today to discuss climate change initiatives and sustainable energy policies, emphasizing the urgent need for international cooperation to meet the 2030 emissions targets. This high-stakes summit, held in Geneva, concluded with a joint statement pledging billions of dollars towards green infrastructure projects across the developing world."""
inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)

# Generate summary
summary_ids = model.generate(
    inputs["input_ids"], 
    max_length=150, 
    min_length=40, 
    length_penalty=2.0,
    num_beams=4, # Added a common optimization for better summary quality
    early_stopping=True
)

print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

## 🚀 Deployment

The final model can be:

* Uploaded to **Hugging Face Hub** for public access via API.
* Integrated into a web app using **FastAPI** or **Streamlit**.
* Containerized with **Docker** for production use.

---

## 🧾 Project Structure
