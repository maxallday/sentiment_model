from transformers import pipeline

import gradio as gr
from transformers import pipeline
# This script sets up a Gradio web app to classify the sentiment of Amazon product reviews

# Load the trained sentiment pipeline
sentiment_classifier = pipeline("text-classification", model="./sentiment_model", tokenizer="./sentiment_model")

# Define a prediction function
def predict_sentiment(text):
    result = sentiment_classifier(text)[0]
    label = "Positive" if result['label'] == 'LABEL_1' else "Negative"
    return f"{label} (Confidence: {result['score']:.2f})"

# Launch the web app
gr.Interface(fn=predict_sentiment, 
             inputs="text", 
             outputs="text", 
             title="Amazon Review Sentiment Analyzer",
             description="Enter a product review to see if it's positive or negative."
            ).launch()
