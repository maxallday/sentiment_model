import tensorflow as tf
import scipy # uses scipy to compute eigenvalues for some reason 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from scipy.sparse.linalg import eigs
from transformers import pipeline
import evaluate
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
# This script demonstrates how to fine-tune a pre-trained DistilBERT model for sentiment analysis using the Amazon Polarity dataset.
# The model will classify product reviews as either positive or negative.
# It uses the Hugging Face Transformers library for model handling and the Datasets library for data loading.       
# Load the Amazon Polarity dataset, which contains product reviews labeled as positive or negative.
dataset = load_dataset("amazon_polarity", trust_remote_code=True)
# Load the pre-trained tokenizer for DistilBERT
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example['content'], padding='max_length', truncation=True)

#Load pre-trained tokenizer and model
#We'll use DistilBERT, a lightweight BERT model


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

#Define evaluation metric (accuracy)
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


#setup training configuration
training_args  = TrainingArguments(
    output_dir = './result',
    eval_strategy = "epoch",
    logging_strategy = "epoch",
    save_strategy = "epoch",
    logging_steps=50,
    logging_dir='./logs',
    per_device_train_batch_size= 16,
    per_device_eval_batch_size= 16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True

)

#Initialize the Trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"].shuffle(seed=45).select(range(500)), #limit size for quick training
    eval_dataset= tokenized_datasets["test"].select(range(1000)),
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)
#Train the model

print(" Starting training")
trainer.train()
print(" Training done — saving model...")
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

trainer.evaluate()

# Plot training and validation accuracy

# ✅ Suggested replacement
train_acc, eval_acc = [], []
train_loss, eval_loss = [],[ ]
for record in trainer.state.log_history:
    if 'train_accuracy' in record:
        train_acc.append(record['train_accuracy'])
    if 'eval_accuracy' in record:
        eval_acc.append(record['eval_accuracy'])
    if 'train_loss' in record:
        train_loss.append(record['train_loss'])
    if 'eval_loss' in record:
        eval_loss.append(record['eval_loss'])    

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_acc, label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Logging Steps')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(eval_acc, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Logging Steps')
plt.ylabel('Accuracy')
plt.legend()
plt.suptitle('Model Accuracy Over Training Steps')

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Logging Steps')     
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(eval_loss, label='Validation Loss')
plt.title('Validation Loss')
plt.xlabel('Logging Steps')
plt.ylabel('Loss')
plt.legend()
plt.suptitle('Model Loss Over Training Steps')



plt.tight_layout()
plt.show()

# Save the model
#model.save_pretrained("./sentiment_model")
#tokenizer.save_pretrained("./sentiment_model")





# Predict on custom examples using the fine-tuned model
sentiment_classifier = pipeline("text-classification", model="./sentiment_model", tokenizer="./sentiment_model")

sample_reviews = [
    "This product is incredible, I’m so impressed!",
    "Absolutely terrible experience. Waste of money.",
    "It works fine, but nothing special."
]

predictions = sentiment_classifier(sample_reviews)

# Output predictions
for review, pred in zip(sample_reviews, predictions):
    label = "Positive" if pred['label'] == 'LABEL_1' else "Negative"
    print(f"Review: {review}")
    print(f"Prediction: {label} (confidence: {pred['score']:.2f})\n")




import gradio as gr

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


