Sentiment Analyzer Pro

This interactive app allows users to analyze the sentiment of multiple reviews or sentences. Built using  Transformers and Gradio, 
it leverages a fine-tuned language model to classify input as **Positive** or **Negative** with confidence scores and visual feedback.

## 🚀 Features

- 🧾 Batch review analysis (multi-line input)
- 📊 Sentiment distribution bar chart
- 🟢🔴 Emoji-enhanced output
- 👍👎 User feedback buttons with logging

## 💻 How to Use

1. Paste one or more reviews in the textbox.
2. Click "Analyze" to see predictions for each review.
3. Use the thumbs-up/thumbs-down buttons to give feedback.

## 🛠️ Tech Stack

- `transformers` (for model pipeline)
- `gradio` (for web UI)
- `matplotlib` (for sentiment chart)
- `torch` (model backend)

## 📂 Model
The sentiment model is stored locally in the `sentiment_model/` folder. It was fine-tuned using Hugging Face Transformers.

## ✨ Demo

Live on Hugging Face Spaces #comming soon

#note
This model is larger than expected because it uses a deep learning approach with transformer-based models (like BERT) trained on a large set of Amazon reviews. These models are very powerful and accurate, but they also come with many parameters, which makes the file size much bigger. While simpler methods could work for basic sentiment analysis or entity recognition, this project focuses on higher accuracy and better performance on real-world text — even if that means a heavier model.
