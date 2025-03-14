# Project: Medical Q&A Chatbot & arXiv Research Chatbot

## Overview

This repository contains two AI-based applications:

1. **Medical Q&A Chatbot**: A chatbot that retrieves medical question answers using TF-IDF and cosine similarity.
2. **arXiv Research Chatbot**: A chatbot for retrieving and summarizing research papers from arXiv using Sentence Transformers and BART.

---

## Folder Structure

```
/Medical Chatbot and arXiv Chatbot
│-- requirements.txt  ✅
│-- model_training.ipynb  ✅ (Medical Chatbot Training)
│-- arxiv_training.ipynb ✅ (arXiv Chatbot Training)
│-- model_weights.pth (Google Drive link required)
│-- saved_model/ (Google Drive link required)
│-- app.py  ✅ (Medical Chatbot GUI)
│-- medical_retriever.py ✅ (Medical Q&A Model)
│-- data_processor.py ✅ (Processes MedQuAD dataset)
│-- arxiv.py ✅ (arXiv Chatbot UI)
│-- dataset/ (If any dataset is required)
│-- README.md  (Project Overview)
```

---

## 1️⃣ Medical Q&A Chatbot

### Features

- Retrieves answers for medical questions using TF-IDF and Cosine Similarity.
- Processes the MedQuAD dataset.
- User-friendly interface using Streamlit.

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 2️⃣ arXiv Research Chatbot

### Features

- Searches research papers in arXiv related to user queries.
- Uses Sentence Transformers for semantic search.
- Summarizes abstracts using BART.

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the chatbot:
   ```bash
   streamlit run arxiv.py
   ```

---

## Model Performance & Evaluation

- **Medical Chatbot:**
  - Accuracy: ✅ Minimum 70% (Ensure by training the model)
  - Metrics: Confusion Matrix, Precision, Recall
- **arXiv Chatbot:**
  - Uses similarity-based retrieval (cosine similarity)
  - Summarization accuracy depends on the BART model

---

## Contribution & Future Improvements

✅ **Enhancements:**

- Improve model accuracy using deep learning.
- Expand datasets for better chatbot responses.

✅ **Contributors:**

- **Medical Chatbot Lead:** [Swathiga S]
- **arXiv Chatbot Lead:** [Swathiga S]

---

## Contact

For any queries, contact: [swathiga22@gmail.com]

